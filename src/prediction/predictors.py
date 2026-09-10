"""Classical and learned short-horizon predictors sharing one safe contract."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import math

import numpy as np

from src.core.contracts import PredictedState, PredictionInput, TrackingState


ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_FEATURE_SCHEMA = (
    "x", "y", "vx", "vy", "log_cov_x", "log_cov_y", "confidence",
    "snr", "measurement_valid", "prediction_only", "dt",
)


def _invalid(name: str, horizon: float, reason: str, started: float) -> PredictedState:
    return PredictedState(False, horizon, predictor_name=name, failure_reason=reason,
                          inference_latency_ms=(perf_counter() - started) * 1000)


class NoPrediction:
    version = "none-2.0.0"
    predictor_name = "none"

    def __init__(self, config, streams=None):
        self.config = config

    def predict(self, request: PredictionInput) -> PredictedState:
        started = perf_counter(); state = request.current_state
        if not state.valid:
            return _invalid(self.predictor_name, request.prediction_horizon_s, "INVALID_CURRENT_STATE", started)
        return PredictedState(True, request.prediction_horizon_s, state.x, state.y, state.vx, state.vy,
                              covariance=state.image_covariance, predictor_name=self.predictor_name,
                              predictor_version=self.version, inference_latency_ms=None)

    def metadata(self):
        return {"name": self.predictor_name, "version": self.version}


class ConstantVelocityPredictor(NoPrediction):
    version = "constant-velocity-1.0.0"
    predictor_name = "cv"

    def predict(self, request: PredictionInput) -> PredictedState:
        started = perf_counter(); state = request.current_state; tau = request.prediction_horizon_s
        if not state.valid:
            return _invalid(self.predictor_name, tau, "INVALID_CURRENT_STATE", started)
        covariance = None if state.image_covariance is None else np.asarray(state.image_covariance, dtype=float).copy()
        if covariance is not None:
            covariance += np.eye(2) * self.config.estimation.process_noise_px_s2 * max(tau, 0) ** 2
        return PredictedState(True, tau, state.x + state.vx * tau, state.y + state.vy * tau,
                              state.vx, state.vy, covariance=covariance,
                              predictor_name=self.predictor_name, predictor_version=self.version,
                              inference_latency_ms=(perf_counter() - started) * 1000)


class ConstantAccelerationPredictor(NoPrediction):
    version = "constant-acceleration-1.0.0"
    predictor_name = "ca"

    @staticmethod
    def estimated_acceleration(history: tuple[TrackingState, ...]) -> tuple[float, float]:
        samples = [state for state in history if state.valid]
        if len(samples) < 2:
            return 0.0, 0.0
        values = []
        window = samples[-6:]
        for previous, current in zip(window[:-1], window[1:]):
            dt = current.timestamp - previous.timestamp
            if dt > 1e-9:
                values.append(((current.vx - previous.vx) / dt, (current.vy - previous.vy) / dt))
        if not values:
            return 0.0, 0.0
        return tuple(float(v) for v in np.median(np.asarray(values), axis=0))

    def predict(self, request: PredictionInput) -> PredictedState:
        started = perf_counter(); state = request.current_state; tau = request.prediction_horizon_s
        if not state.valid:
            return _invalid(self.predictor_name, tau, "INVALID_CURRENT_STATE", started)
        ax, ay = self.estimated_acceleration(request.state_history)
        return PredictedState(True, tau, state.x + state.vx * tau + 0.5 * ax * tau * tau,
                              state.y + state.vy * tau + 0.5 * ay * tau * tau,
                              state.vx + ax * tau, state.vy + ay * tau, ax=ax, ay=ay,
                              covariance=state.image_covariance, predictor_name=self.predictor_name,
                              predictor_version=self.version, inference_latency_ms=(perf_counter() - started) * 1000)


def temporal_features(request: PredictionInput, window: int) -> np.ndarray:
    history = request.state_history[-window:]
    quality = request.measurement_quality_history[-len(history):]
    rows = []
    for index, state in enumerate(history):
        covariance = state.image_covariance
        if covariance is None and state.covariance is not None and np.shape(state.covariance)[0] >= 2:
            covariance = np.asarray(state.covariance)[:2, :2]
        diag = np.diag(covariance) if covariance is not None else np.array([1e4, 1e4])
        q = quality[index] if index < len(quality) else {}
        dt = 0.0 if index == 0 else max(0.0, state.timestamp - history[index - 1].timestamp)
        rows.append((state.x, state.y, state.vx, state.vy,
                     math.log(max(float(diag[0]), 1e-8)), math.log(max(float(diag[1]), 1e-8)),
                     float(state.confidence or 0.0), math.log1p(max(0.0, float(q.get("image_snr_estimate", 0.0)))),
                     float(bool(q.get("measurement_valid", state.measurement_used))),
                     float(state.prediction_only), dt))
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=20.0, neginf=-20.0)


class NeuralPredictor(ConstantVelocityPredictor):
    kind = "gru"
    predictor_name = "gru"
    version = "neural-temporal-1.0.0"

    def __init__(self, config, streams=None):
        super().__init__(config, streams)
        self.settings = config.temporal; self.model = None; self.checkpoint = None
        self.load_error = None; self.fallback_count = 0
        self._load()

    def _load(self):
        configured = {"gru": self.settings.gru_model_path, "lstm": self.settings.lstm_model_path,
                      "gru_residual_cv": self.settings.gru_residual_model_path}.get(self.kind, self.settings.model_path)
        path = Path(configured)
        if not path.is_absolute(): path = ROOT / path
        if not path.exists():
            self.load_error = f"checkpoint not found: {path}"; return
        try:
            from src.ml.models import build_temporal_model, require_torch
            torch, _ = require_torch(); checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("kind") != self.kind:
                raise ValueError(f"checkpoint kind {checkpoint.get('kind')} does not match {self.kind}")
            model = build_temporal_model(checkpoint["kind"], checkpoint["input_feature_count"],
                                         checkpoint["hidden_size"], checkpoint["num_layers"])
            model.load_state_dict(checkpoint["model_state"]); model.eval()
            self.model, self.checkpoint = model, checkpoint
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _fallback(self, request: PredictionInput, reason: str, started: float) -> PredictedState:
        self.fallback_count += 1
        fallback = (ConstantVelocityPredictor(self.config).predict(request) if self.settings.fallback == "cv"
                    else NoPrediction(self.config).predict(request))
        return PredictedState(**{**fallback.__dict__, "predictor_name": self.predictor_name,
                                 "predictor_version": self.version,
                                 "model_id": None if self.checkpoint is None else self.checkpoint.get("model_id"),
                                 "inference_latency_ms": (perf_counter() - started) * 1000,
                                 "fallback_used": True, "failure_reason": reason})

    def predict(self, request: PredictionInput) -> PredictedState:
        started = perf_counter(); current = request.current_state; tau = request.prediction_horizon_s
        if not current.valid: return _invalid(self.predictor_name, tau, "INVALID_CURRENT_STATE", started)
        if self.model is None or self.checkpoint is None:
            return self._fallback(request, "MODEL_UNAVAILABLE", started)
        window = int(self.checkpoint["history_frames"])
        if len(request.state_history) < window or not all(state.valid for state in request.state_history[-window:]):
            return self._fallback(request, "INSUFFICIENT_HISTORY", started)
        try:
            features = temporal_features(request, window)
            mean = np.asarray(self.checkpoint["normalizer"]["mean"], dtype=np.float32)
            std = np.asarray(self.checkpoint["normalizer"]["std"], dtype=np.float32)
            features = (features - mean) / std
            from src.ml.models import require_torch
            torch, _ = require_torch()
            with torch.inference_mode():
                output = self.model(torch.from_numpy(features[None]).float(),
                                    torch.tensor([tau], dtype=torch.float32))[0].cpu().numpy().astype(float)
            if output.shape != (4,) or not np.isfinite(output).all():
                return self._fallback(request, "INVALID_OUTPUT", started)
            if self.checkpoint.get("target_mode") == "residual_cv":
                base = ConstantVelocityPredictor(self.config).predict(request)
                predicted = np.asarray(base.pixel) + output[:2]
            else:
                predicted = np.asarray(current.pixel) + output[:2]
            displacement = predicted - np.asarray(current.pixel)
            if float(np.linalg.norm(displacement)) > self.settings.max_displacement_px:
                return self._fallback(request, "OUT_OF_RANGE_PREDICTION", started)
            log_min, log_max = self.checkpoint["log_variance_bounds"]
            variance = np.exp(np.clip(output[2:], log_min, log_max)) * float(self.checkpoint.get("uncertainty_calibration", 1.0))
            sigma = np.sqrt(variance)
            if float(np.max(sigma)) > self.settings.max_sigma_px:
                return self._fallback(request, "HIGH_UNCERTAINTY", started)
            velocity = displacement / max(tau, 1e-6)
            weight = float(np.clip(1.0 / (1.0 + float(np.mean(variance)) / 25.0), 0, 1))
            return PredictedState(True, tau, float(predicted[0]), float(predicted[1]),
                                  float(velocity[0]), float(velocity[1]), covariance=np.diag(variance),
                                  predictor_name=self.predictor_name, predictor_version=self.version,
                                  model_id=self.checkpoint["model_id"],
                                  inference_latency_ms=(perf_counter() - started) * 1000,
                                  uncertainty_weight=weight)
        except Exception:
            return self._fallback(request, "INVALID_OUTPUT", started)

    def metadata(self):
        return {"name": self.predictor_name, "version": self.version,
                "model_id": None if self.checkpoint is None else self.checkpoint.get("model_id"),
                "weights_hash": None if self.checkpoint is None else self.checkpoint.get("weights_hash"),
                "fallback_count": self.fallback_count, "load_error": self.load_error}


class GRUPredictor(NeuralPredictor):
    kind = "gru"; predictor_name = "gru"


class LSTMPredictor(NeuralPredictor):
    kind = "lstm"; predictor_name = "lstm"


class GRUResidualCVPredictor(NeuralPredictor):
    kind = "gru_residual_cv"; predictor_name = "gru_residual_cv"
