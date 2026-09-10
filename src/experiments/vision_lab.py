"""Paired same-frame classical localization experiments and dataset export."""
from __future__ import annotations

import base64
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.core.engine import ROOT, SimulationEngine
from src.core.registry import REGISTRY, Stage, validate_config
from src.core.metrics import stats


DEFAULT_ALGORITHMS = ("binary_centroid", "weighted_centroid", "gradient_centroid", "gaussian_fit")
VISION_CONDITIONS = {
    "CLEAN": {"disturbance.gaussian_noise": False, "disturbance.blur": False,
              "disturbance.beacon.nominal_intensity": 245, "disturbance.distractors.enabled": False},
    "WEAK_BEACON": {"disturbance.beacon.nominal_intensity": 70, "camera.noise_sigma": 12},
    "HIGH_NOISE": {"camera.noise_sigma": 22},
    "BLUR": {"disturbance.sensor.defocus_sigma_px": 3.0},
    "ELLIPTICAL": {"disturbance.beacon.elliptical_ratio": 2.2},
    "BACKGROUND": {"disturbance.sensor.background_brightness": 45, "disturbance.sensor.background_gradient": 55},
    "GLARE": {"disturbance.glare.enabled": True, "disturbance.glare.intensity": 140},
    "DISTRACTOR": {"disturbance.distractors.enabled": True, "disturbance.distractors.count": 3,
                   "disturbance.distractors.intensity": 240},
    "CLIPPING": {"orbit.initial_hill_position_m": (0.0, 3.6, 2.6)},
    "SATURATION": {"disturbance.beacon.nominal_intensity": 255, "disturbance.beacon.saturation_level": 220},
    "DYNAMIC": {"disturbance.vibration.enabled": True, "disturbance.vibration.level": "HIGH",
                "disturbance.vibration.stochastic_sigma_rad": 0.0008},
    "COMBINED_STRESS": {"disturbance.beacon.nominal_intensity": 95, "camera.noise_sigma": 18,
                        "disturbance.sensor.defocus_sigma_px": 2.5, "disturbance.sensor.background_brightness": 35,
                        "disturbance.distractors.enabled": True, "disturbance.distractors.count": 2,
                        "disturbance.vibration.enabled": True, "disturbance.vibration.level": "HIGH",
                        "disturbance.vibration.stochastic_sigma_rad": 0.0005},
}


def _set_path(value: dict, path: str, replacement):
    target = value
    pieces = path.split(".")
    for piece in pieces[:-1]:
        target = target[piece]
    target[pieces[-1]] = replacement


def _encoded(image: np.ndarray | None):
    if image is None or image.size == 0:
        return None
    display = image
    if display.dtype != np.uint8:
        finite = np.nan_to_num(display, nan=0, posinf=0, neginf=0)
        maximum = float(finite.max()) if finite.size else 0
        display = np.uint8(np.clip(finite * (255 / maximum if maximum > 0 else 1), 0, 255))
    ok, data = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return None if not ok else "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def compare_frame(config, frame: np.ndarray, true_pixel: tuple[float, float] | None = None,
                  algorithms=DEFAULT_ALGORITHMS, benchmark_mode="FULL_FRAME", include_images=True):
    """Every method receives byte-identical pixels; truth is evaluation-only."""
    config = validate_config(config)
    source = np.asarray(frame).copy()
    checksum = hashlib.sha256(source.tobytes()).hexdigest()
    origin = (0, 0)
    algorithm_input = source
    if benchmark_mode == "ORACLE_ROI":
        if true_pixel is None:
            raise ValueError("ORACLE_ROI requires evaluation truth in the harness")
        radius = max(12, config.vision.roi_padding_px * 3)
        x0, y0 = max(0, int(true_pixel[0]) - radius), max(0, int(true_pixel[1]) - radius)
        x1, y1 = min(source.shape[1], int(true_pixel[0]) + radius + 1), min(source.shape[0], int(true_pixel[1]) + radius + 1)
        algorithm_input = source[y0:y1, x0:x1].copy()
        origin = (x0, y0)
    elif benchmark_mode != "FULL_FRAME":
        raise ValueError("benchmark_mode must be FULL_FRAME or ORACLE_ROI")
    results = []
    for name in algorithms:
        implementation = REGISTRY[Stage.VISION].get(name)
        if implementation is None:
            raise ValueError(f"Unavailable vision algorithm '{name}'")
        localizer = implementation(config, None)
        received = algorithm_input.copy()
        started = perf_counter()
        measurement = localizer.measure(received, 0.0)
        elapsed = (perf_counter() - started) * 1000
        if measurement.valid and benchmark_mode == "ORACLE_ROI":
            measurement = replace(measurement, pixel=(measurement.pixel[0] + origin[0], measurement.pixel[1] + origin[1]))
        error = None if true_pixel is None or not measurement.valid else float(np.linalg.norm(np.asarray(measurement.pixel) - true_pixel))
        debug = getattr(localizer, "last_debug", {})
        overlay = source.copy()
        if measurement.valid:
            cv2.drawMarker(overlay, tuple(np.rint(measurement.pixel).astype(int)), (30, 245, 100), cv2.MARKER_CROSS, 18, 2)
        if true_pixel is not None:
            true_int = tuple(np.rint(true_pixel).astype(int))
            cv2.drawMarker(overlay, true_int, (20, 70, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            if measurement.valid:
                cv2.line(overlay, true_int, tuple(np.rint(measurement.pixel).astype(int)), (0, 210, 255), 1)
            cv2.putText(overlay, "TRUTH: EVALUATION ONLY", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 70, 255), 1)
        results.append(dict(algorithm=name, input_checksum=hashlib.sha256(received.tobytes()).hexdigest(),
                            measurement=measurement.as_dict(), localization_error_px=error, latency_ms=elapsed,
                            mask_image=_encoded(debug.get("mask")) if include_images else None,
                            roi_image=_encoded(debug.get("roi")) if include_images else None,
                            overlay_image=_encoded(overlay) if include_images else None))
    return dict(benchmark_mode=benchmark_mode, source_checksum=checksum,
                input_checksum=hashlib.sha256(algorithm_input.tobytes()).hexdigest(),
                raw_image=_encoded(source) if include_images else None, truth_evaluation_only=true_pixel, results=results)


def _condition_config(config, condition: str, seed: int):
    data = copy.deepcopy(validate_config(config).model_dump(mode="json"))
    for path, value in VISION_CONDITIONS[condition].items():
        _set_path(data, path, value)
    data.update(seed=seed, run_index=0, fps=10, duration_s=1.0)
    data["camera"].update(width_px=160, height_px=120, focal_length_px=130)
    data["vision"]["algorithm"] = "centroid"
    data["estimator"] = "none"
    data["pipeline_preset"] = "CUSTOM"
    return validate_config(data)


def run_vision_benchmark(config, seeds=(41, 42), algorithms=DEFAULT_ALGORITHMS,
                         conditions=None, frames_per_seed=8, benchmark_mode="FULL_FRAME", store=None):
    conditions = tuple(conditions or VISION_CONDITIONS)
    rows = []
    for condition in conditions:
        if condition not in VISION_CONDITIONS:
            raise ValueError(f"Unknown vision condition '{condition}'")
        for seed in seeds:
            resolved = _condition_config(config, condition, int(seed))
            engine = SimulationEngine(config=resolved)
            for frame_index in range(frames_per_seed):
                engine.step(publish=True, annotate=False)
                frame, telemetry, _ = engine.raw_snapshot()
                truth = telemetry.get("ground_truth_pixel")
                comparison = compare_frame(resolved, frame, None if truth is None else tuple(truth), algorithms,
                                           benchmark_mode, include_images=False)
                for result in comparison["results"]:
                    rows.append(dict(condition=condition, seed=int(seed), frame_index=frame_index,
                                     algorithm=result["algorithm"], valid=result["measurement"]["valid"],
                                     error_px=result["localization_error_px"], latency_ms=result["latency_ms"],
                                     failure_reason=result["measurement"]["failure_reason"],
                                     quality=result["measurement"]["quality"],
                                     source_checksum=comparison["input_checksum"]))
    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for algorithm in algorithms:
            selected = [row for row in rows if row["condition"] == condition and row["algorithm"] == algorithm]
            errors = [row["error_px"] for row in selected if row["error_px"] is not None]
            latencies = [row["latency_ms"] for row in selected]
            summary[condition][algorithm] = dict(samples=len(selected), valid_measurement_percent=100 * len(errors) / len(selected) if selected else 0,
                                                 rmse_px=math.sqrt(float(np.mean(np.square(errors)))) if errors else None,
                                                 mae_px=float(np.mean(errors)) if errors else None,
                                                 median_px=stats(errors)["median"], p95_px=stats(errors)["p95"],
                                                 max_px=stats(errors)["max"], mean_latency_ms=stats(latencies)["mean"],
                                                 p95_latency_ms=stats(latencies)["p95"])
    result = dict(kind="CLASSICAL_VISION_BENCHMARK", benchmark_mode=benchmark_mode,
                  seeds=list(seeds), conditions=list(conditions), algorithms=list(algorithms), summary=summary, rows=rows)
    if store is not None:
        store.append_lab_result("CLASSICAL_VISION_BENCHMARK", validate_config(config), result)
    return result


def parameter_sweep(config, path: str, values, seeds=(41, 42), algorithms=DEFAULT_ALGORITHMS,
                    frames_per_seed=5, store=None):
    points = []
    for value in values:
        rows=[]
        for seed in seeds:
            data = validate_config(config).model_dump(mode="json")
            _set_path(data, path, value)
            data.update(seed=int(seed),run_index=0,fps=10,duration_s=1.0,pipeline_preset="CUSTOM")
            data["camera"].update(width_px=160,height_px=120,focal_length_px=130)
            resolved=validate_config(data); engine=SimulationEngine(config=resolved)
            for _ in range(frames_per_seed):
                engine.step(publish=True,annotate=False); frame,telemetry,_=engine.raw_snapshot()
                truth=telemetry.get("ground_truth_pixel")
                comparison=compare_frame(resolved,frame,None if truth is None else tuple(truth),algorithms,include_images=False)
                for result in comparison["results"]:
                    rows.append((result["algorithm"],result["localization_error_px"],result["latency_ms"],result["measurement"]["valid"]))
        summaries={}
        for algorithm in algorithms:
            selected=[row for row in rows if row[0]==algorithm];errors=[row[1] for row in selected if row[1] is not None]
            latencies=[row[2] for row in selected]
            summaries[algorithm]=dict(samples=len(selected),valid_measurement_percent=100*len(errors)/len(selected) if selected else 0,
                                      rmse_px=math.sqrt(float(np.mean(np.square(errors)))) if errors else None,
                                      mean_latency_ms=stats(latencies)["mean"],p95_latency_ms=stats(latencies)["p95"])
        points.append(dict(value=value, algorithms=summaries))
    output = dict(kind="VISION_PARAMETER_SWEEP", parameter=path, seeds=list(seeds), points=points)
    if store is not None:
        store.append_lab_result("VISION_PARAMETER_SWEEP", validate_config(config), output)
    return output


def export_dataset_sample(directory: str | Path, frame: np.ndarray, measurement, truth_pixel,
                          config, seed: int, trajectory_id: str, scenario_id: str):
    """Explicitly invoked Prompt-5 foundation; never called during normal tracking."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    sample_id = f"{scenario_id}-{trajectory_id}-{seed}"
    image_path = directory / f"{sample_id}.png"
    metadata_path = directory / f"{sample_id}.json"
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError("Could not export dataset image")
    metadata = dict(sample_id=sample_id, seed=seed, trajectory_id=trajectory_id, scenario_id=scenario_id,
                    simulator_version=config.simulation_schema_version, true_beacon_centre=list(truth_pixel),
                    classical_measurement=measurement.as_dict(), scenario_parameters=config.model_dump(mode="json"))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"image": str(image_path), "metadata": str(metadata_path)}
