"""Production measurement correctors with explicit, observable safety fallback."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import math

import numpy as np

from src.core.contracts import CorrectionFailureReason, Measurement
from src.ml.features import auxiliary_features, fixed_roi, normalize_features


ROOT = Path(__file__).resolve().parents[2]


class NoCorrection:
    version = "none-1.0.0"

    def __init__(self, config, streams=None):
        self.config = config

    def correct(self, frame: np.ndarray, measurement: Measurement, context=None) -> Measurement:
        return replace(measurement, classical_pixel=measurement.pixel, correction_algorithm="none")

    def metadata(self):
        return {"algorithm": "none", "version": self.version, "fallback_count": 0}


class CNNResidualCorrector:
    """ROI CNN -> residual position and bounded diagonal measurement covariance."""
    version = "cnn-residual-correction-1.0.0"

    def __init__(self, config, streams=None):
        self.config = config
        self.settings = config.cnn
        self.model = None
        self.checkpoint = None
        self.load_error: str | None = None
        self.fallback_count = 0
        self.inference_count = 0
        self._load()

    def _load(self):
        path = Path(self.settings.model_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            self.load_error = f"checkpoint not found: {path}"
            return
        try:
            from src.ml.models import build_spot_model, require_torch
            torch, _ = require_torch()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_spot_model(int(checkpoint["aux_feature_count"]), checkpoint["architecture"])
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.model, self.checkpoint = model, checkpoint
        except Exception as exc:  # safe optional dependency/model boundary
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _fallback(self, measurement: Measurement, started: float, reason: CorrectionFailureReason,
                  quality: dict | None = None) -> Measurement:
        self.fallback_count += 1
        merged = dict(measurement.quality)
        merged.update(quality or {})
        merged.update(cnn_fallback=True, cnn_failure_reason=reason.value)
        return replace(measurement, classical_pixel=measurement.pixel, correction=(0.0, 0.0),
                       correction_algorithm="cnn_residual", correction_model_id=self.settings.model_id,
                       correction_latency_ms=(perf_counter() - started) * 1000,
                       correction_fallback=True, correction_failure_reason=reason, quality=merged)

    def correct(self, frame: np.ndarray, measurement: Measurement, context=None) -> Measurement:
        started = perf_counter()
        if not measurement.valid:
            return replace(measurement, classical_pixel=None, correction_algorithm="cnn_residual",
                           correction_model_id=self.settings.model_id)
        if self.model is None or self.checkpoint is None:
            return self._fallback(measurement, started, CorrectionFailureReason.MODEL_UNAVAILABLE,
                                  {"cnn_load_error": self.load_error})
        try:
            pre_started = perf_counter()
            roi, origin, roi_clipped = fixed_roi(frame, measurement.pixel, int(self.checkpoint["roi_size_px"]))
            aux = auxiliary_features(measurement, origin, int(self.checkpoint["roi_size_px"]))
            if int(self.checkpoint["aux_feature_count"]) == 0:
                aux = np.zeros((0,), dtype=np.float32)
            else:
                aux = normalize_features(aux, self.checkpoint["normalizer"])
                if float(np.max(np.abs(aux))) > self.settings.ood_z_threshold:
                    return self._fallback(measurement, started, CorrectionFailureReason.OOD_WARNING,
                                          {"cnn_max_feature_z": float(np.max(np.abs(aux)))})
            preprocessing_ms = (perf_counter() - pre_started) * 1000
            from src.ml.models import require_torch
            torch, _ = require_torch()
            inference_started = perf_counter()
            with torch.inference_mode():
                image = torch.from_numpy(roi[None, None]).float()
                features = None if aux.size == 0 else torch.from_numpy(aux[None]).float()
                output = self.model(image, features)[0].cpu().numpy().astype(float)
            inference_ms = (perf_counter() - inference_started) * 1000
            if output.shape != (4,) or not np.isfinite(output).all():
                return self._fallback(measurement, started, CorrectionFailureReason.INVALID_OUTPUT,
                                      {"cnn_preprocessing_ms": preprocessing_ms, "cnn_inference_ms": inference_ms})
            correction = output[:2]
            if float(np.linalg.norm(correction)) > self.settings.max_correction_px:
                return self._fallback(measurement, started, CorrectionFailureReason.EXCESSIVE_CORRECTION,
                                      {"cnn_raw_correction_px": correction.tolist()})
            minimum = math.log(self.settings.sigma_min_px ** 2)
            maximum = math.log(self.settings.sigma_max_px ** 2)
            log_variance = np.clip(output[2:], minimum, maximum)
            variance = np.exp(log_variance) * float(self.checkpoint.get("uncertainty_calibration", 1.0))
            sigma = np.sqrt(variance)
            if float(np.max(sigma)) >= self.settings.sigma_max_px:
                return self._fallback(measurement, started, CorrectionFailureReason.HIGH_UNCERTAINTY,
                                      {"cnn_predicted_sigma_px": sigma.tolist()})
            if self.settings.uncertainty_mode == "learned":
                covariance = np.diag(variance)
            elif self.settings.uncertainty_mode == "fixed":
                covariance = np.eye(2) * self.config.estimation.measurement_noise_px2
            else:
                quality = max(float(measurement.confidence or 0.05), 0.05)
                covariance = np.eye(2) * self.config.estimation.measurement_noise_px2 / (quality * quality)
            corrected = np.asarray(measurement.pixel, dtype=float) + correction
            total_ms = (perf_counter() - started) * 1000
            if total_ms > self.settings.max_total_latency_ms:
                return self._fallback(measurement, started, CorrectionFailureReason.INFERENCE_TIMEOUT,
                                      {"cnn_total_latency_ms": total_ms})
            self.inference_count += 1
            confidence = float(np.clip((measurement.confidence or 0.5) * math.exp(-float(np.mean(sigma)) / 10), 0, 1))
            quality = dict(measurement.quality)
            quality.update(cnn_preprocessing_ms=preprocessing_ms, cnn_inference_ms=inference_ms,
                           cnn_postprocessing_ms=max(0.0, total_ms - preprocessing_ms - inference_ms),
                           cnn_predicted_sigma_px=sigma.tolist(), cnn_roi_clipped=roi_clipped,
                           cnn_uncertainty_mode=self.settings.uncertainty_mode, cnn_fallback=False)
            return replace(measurement, pixel=tuple(float(v) for v in corrected),
                           classical_pixel=tuple(float(v) for v in measurement.pixel),
                           correction=tuple(float(v) for v in correction), covariance=covariance,
                           confidence=confidence, correction_algorithm="cnn_residual",
                           correction_model_id=self.checkpoint["model_id"],
                           correction_model_version=self.checkpoint["model_version"],
                           correction_latency_ms=total_ms, correction_fallback=False,
                           correction_failure_reason=None, quality=quality)
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            return self._fallback(measurement, started, CorrectionFailureReason.INVALID_ROI,
                                  {"cnn_exception": f"{type(exc).__name__}: {exc}"})

    def metadata(self):
        fields = ("model_id", "model_version", "architecture", "weights_hash", "dataset_id",
                  "dataset_version", "uncertainty_calibration")
        meta = {} if self.checkpoint is None else {name:self.checkpoint.get(name) for name in fields}
        return {"algorithm": "cnn_residual", "version": self.version, "model": meta,
                "model_available": self.model is not None, "load_error": self.load_error,
                "fallback_count": self.fallback_count, "inference_count": self.inference_count,
                "uncertainty_mode": self.settings.uncertainty_mode}
