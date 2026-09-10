"""Open-loop replay and closed-loop PAT experiments for temporal prediction."""
from __future__ import annotations

import copy
import math
import numpy as np

from src.core.contracts import PredictionInput, TrackingState
from src.core.engine import SimulationEngine
from src.core.metrics import stats
from src.core.registry import REGISTRY, Stage, validate_config


DEFAULT_PREDICTORS = ("none", "cv", "ca", "gru", "lstm", "gru_residual_cv")


def record_state_sequence(config, duration_s=3.0) -> list[dict]:
    data = validate_config(config).model_dump(mode="json")
    data.update(pipeline_preset="CUSTOM", estimator="akf_r", predictor="none", controller="pid",
                prediction_horizon_s=0.0)
    resolved = validate_config(data)
    engine = SimulationEngine(config=resolved, retain_frames=0)
    rows = []
    for _ in range(max(2, math.ceil(duration_s * resolved.fps))):
        _, telemetry = engine.step(publish=False, annotate=False)
        vector = telemetry.get("estimator_state")
        truth = telemetry.get("ground_truth_pixel")
        if vector is None or truth is None:
            continue
        image_covariance = telemetry.get("estimator_image_covariance")
        state = TrackingState(
            x=float(vector[0]), y=float(vector[1]), vx=float(vector[2]), vy=float(vector[3]),
            timestamp=float(telemetry["timestamp"]), valid=True,
            covariance=None if image_covariance is None else np.asarray(image_covariance),
            image_covariance=None if image_covariance is None else np.asarray(image_covariance),
            confidence=telemetry.get("measurement_confidence"),
            measurement_used=bool(telemetry.get("estimator_measurement_used", False)),
            prediction_only=bool(telemetry.get("estimator_prediction_only", False)),
            estimator_name="akf_r",
        )
        quality = dict(telemetry.get("measurement_quality") or {})
        quality.update(image_snr_estimate=telemetry.get("image_snr_estimate"),
                       measurement_valid=telemetry.get("measurement_valid", False))
        rows.append(dict(state=state, truth=np.asarray(truth, dtype=float), quality=quality,
                         measurement_valid=bool(telemetry.get("measurement_valid", False))))
    return rows


def _summary(errors, latencies, fallback, angular, dropout):
    values = stats(errors)
    return dict(
        samples=len(errors), rmse_px=math.sqrt(float(np.mean(np.square(errors)))) if errors else None,
        mean_error_px=values["mean"], median_px=values["median"], p95_px=values["p95"],
        maximum_px=values["max"], angular_rmse_rad=math.sqrt(float(np.mean(np.square(angular)))) if angular else None,
        dropout_prediction_rmse_px=math.sqrt(float(np.mean(np.square(dropout)))) if dropout else None,
        mean_latency_ms=stats(latencies)["mean"], p95_latency_ms=stats(latencies)["p95"],
        fallback_count=fallback,
    )


def compare_predictors_open_loop(sequence: list[dict], config, predictors=DEFAULT_PREDICTORS,
                                 horizon_s=0.05, history_frames=None) -> dict:
    resolved = validate_config(config)
    implementations = {name: REGISTRY[Stage.PREDICTOR][name](resolved, None) for name in predictors}
    rows = {name: [] for name in predictors}
    history_frames = history_frames or resolved.temporal.history_frames
    timestamps = np.asarray([row["state"].timestamp for row in sequence])
    for index, row in enumerate(sequence):
        desired_time = row["state"].timestamp + horizon_s
        future_index = int(np.searchsorted(timestamps, desired_time, side="left"))
        if future_index >= len(sequence):
            continue
        if abs(float(timestamps[future_index]) - desired_time) <= 1e-9:
            future_truth = sequence[future_index]["truth"]
        else:
            previous_index = future_index - 1
            if previous_index < 0:
                continue
            span = float(timestamps[future_index] - timestamps[previous_index])
            if span <= 0:
                continue
            weight = float((desired_time - timestamps[previous_index]) / span)
            future_truth = ((1.0 - weight) * sequence[previous_index]["truth"] +
                            weight * sequence[future_index]["truth"])
        actual_horizon = float(horizon_s)
        history = tuple(item["state"] for item in sequence[max(0, index-history_frames+1):index+1])
        qualities = tuple(item["quality"] for item in sequence[max(0, index-history_frames+1):index+1])
        request = PredictionInput(row["state"], history, qualities, tuple(state.timestamp for state in history),
                                  actual_horizon, {"focal_length_px": resolved.camera.focal_length_px})
        for name, implementation in implementations.items():
            prediction = implementation.predict(request)
            error = None if not prediction.valid else float(np.linalg.norm(np.asarray(prediction.pixel)-future_truth))
            rows[name].append(dict(error_px=error, angular_error_rad=None if error is None else error/resolved.camera.focal_length_px,
                                   latency_ms=prediction.inference_latency_ms or 0.0,
                                   fallback=prediction.fallback_used, failure_reason=prediction.failure_reason,
                                   dropout=bool(row["state"].prediction_only), horizon_s=actual_horizon))
    summaries = {}
    for name, values in rows.items():
        valid = [value for value in values if value["error_px"] is not None]
        summaries[name] = _summary(
            [value["error_px"] for value in valid], [value["latency_ms"] for value in values],
            sum(value["fallback"] for value in values), [value["angular_error_rad"] for value in valid],
            [value["error_px"] for value in valid if value["dropout"]],
        )
    return dict(kind="PREDICTOR_OPEN_LOOP_REPLAY", horizon_s=horizon_s,
                predictors=list(predictors), summaries=summaries, rows=rows)


def run_predictor_replay(config, duration_s=3.0, predictors=DEFAULT_PREDICTORS,
                         horizon_s=0.05, history_frames=None):
    sequence = record_state_sequence(config, duration_s)
    return compare_predictors_open_loop(sequence, config, predictors, horizon_s, history_frames)


def predictor_horizon_sweep(config, horizons=(0.02, 0.05, 0.1, 0.2),
                            predictors=DEFAULT_PREDICTORS, duration_s=3.0):
    sequence = record_state_sequence(config, duration_s)
    return dict(kind="PREDICTOR_HORIZON_SWEEP", points=[
        compare_predictors_open_loop(sequence, config, predictors, float(horizon)) for horizon in horizons
    ])


def predictor_history_sweep(config, histories=(5, 10, 20, 30), predictors=("gru", "lstm"),
                            horizon_s=0.05, duration_s=3.0):
    sequence = record_state_sequence(config, duration_s)
    return dict(kind="PREDICTOR_HISTORY_SWEEP", points=[
        {"history_frames": int(history), **compare_predictors_open_loop(
            sequence, config, predictors, horizon_s, int(history))}
        for history in histories
    ])


def closed_loop_predictor_benchmark(config, predictors=("none", "cv", "ca", "gru", "lstm"),
                                    seeds=(41, 42), duration_s=3.0, horizon_s=0.05):
    base = validate_config(config).model_dump(mode="json")
    output = {}
    for name in predictors:
        per_seed = []
        for index, seed in enumerate(seeds):
            data = copy.deepcopy(base)
            data.update(seed=int(seed), run_index=index, pipeline_preset="CUSTOM", estimator="akf_r",
                        predictor=name, controller="pid" if name == "none" else "ff_pid",
                        prediction_horizon_s=float(horizon_s))
            resolved = validate_config(data)
            engine = SimulationEngine(config=resolved, retain_frames=0)
            for _ in range(math.ceil(duration_s * resolved.fps)):
                engine.step(publish=False, annotate=False)
            per_seed.append(engine.metrics.summary())
        output[name] = dict(
            runs=len(per_seed),
            tracking_rmse_px=stats([row["rmse_tracking_error_px"] for row in per_seed if row["rmse_tracking_error_px"] is not None]),
            p95_tracking_error_px=stats([row["p95_tracking_error_px"] for row in per_seed if row["p95_tracking_error_px"] is not None]),
            locked_percentage=stats([row["locked_percentage"] for row in per_seed]),
            rate_saturation_percent=stats([row["rate_saturation_percent"] for row in per_seed]),
            control_effort=stats([row["mean_control_effort"] for row in per_seed]),
            predictor_fallback_count=sum(row["predictor_fallback_count"] for row in per_seed),
            lead_prediction_rmse_px=stats([row["lead_prediction_rmse_px"] for row in per_seed if row["lead_prediction_rmse_px"] is not None]),
        )
    return dict(kind="PREDICTOR_CLOSED_LOOP_PAIRED_SEEDS", seeds=list(seeds),
                horizon_s=horizon_s, results=output)
