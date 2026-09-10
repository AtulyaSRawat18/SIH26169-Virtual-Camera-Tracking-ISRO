"""Trajectory-safe temporal dataset built from deployed-style estimator outputs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from src.core.config import PHYSICS_MODEL_VERSION, SIMULATION_SCHEMA_VERSION
from src.core.engine import ROOT, SimulationEngine
from src.core.scenarios import resolve_config
from src.ml.dataset import DEFAULT_PRESETS, OOD_PRESET, trajectory_split
from src.prediction.predictors import TEMPORAL_FEATURE_SCHEMA


TEMPORAL_SPLIT_SEED = 42
TEMPORAL_DATASET_VERSION = "temporal-state-1.3.0"
TEMPORAL_PRESETS = ("SAT_SAT_NOMINAL",) + DEFAULT_PRESETS


def _definition_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def _state_row(telemetry: dict, previous_timestamp: float | None) -> np.ndarray:
    state = telemetry.get("estimator_state") or [0, 0, 0, 0]
    covariance = telemetry.get("estimator_image_covariance")
    diagonal = np.diag(np.asarray(covariance, dtype=float)) if covariance is not None else np.array([1e4, 1e4])
    timestamp = float(telemetry["timestamp"])
    return np.asarray((state[0], state[1], state[2], state[3],
                       np.log(max(float(diagonal[0]), 1e-8)), np.log(max(float(diagonal[1]), 1e-8)),
                       float(telemetry.get("measurement_confidence") or 0.0),
                       np.log1p(max(0.0, float(telemetry.get("image_snr_estimate") or 0.0))),
                       float(bool(telemetry.get("measurement_valid", False))),
                       float(bool(telemetry.get("estimator_prediction_only", False))),
                       0.0 if previous_timestamp is None else timestamp - previous_timestamp), dtype=np.float32)


def generate_temporal_dataset(output_root: str | Path = ROOT / "data" / "ml" / "temporal",
                              trajectories_per_scenario: int = 3, frames_per_trajectory: int = 70,
                              history_frames: int = 12, horizons_s=(0.02, 0.05, 0.1, 0.2),
                              fps: float = 100.0, seed: int = TEMPORAL_SPLIT_SEED,
                              use_cnn: bool = False, camera_width_px: int = 160,
                              camera_height_px: int = 120, focal_length_px: float = 130.0,
                              beacon_radius_px: int = 5, camera_noise_sigma: float = 8.0) -> dict:
    if history_frames < 2 or frames_per_trajectory <= history_frames:
        raise ValueError("Temporal dataset needs a usable history and future interval")
    definition = dict(version=TEMPORAL_DATASET_VERSION, presets=list(TEMPORAL_PRESETS),
                      trajectories_per_scenario=trajectories_per_scenario,
                      frames_per_trajectory=frames_per_trajectory, history_frames=history_frames,
                      horizons_s=list(horizons_s), fps=fps, seed=seed, use_cnn=use_cnn,
                      camera_width_px=camera_width_px, camera_height_px=camera_height_px,
                      focal_length_px=focal_length_px, beacon_radius_px=beacon_radius_px,
                      camera_noise_sigma=camera_noise_sigma)
    dataset_id = f"temporal-{_definition_id(definition)}"
    directory = Path(output_root) / dataset_id; directory.mkdir(parents=True, exist_ok=True)
    trajectory_records: dict[str, list[dict]] = {}; ood_ids: list[str] = []
    plan = [(preset, False) for preset in TEMPORAL_PRESETS] + [(OOD_PRESET, True)]
    for preset_index, (preset, is_ood) in enumerate(plan):
        for trajectory_index in range(trajectories_per_scenario):
            trajectory_seed = int(seed + 2027 * preset_index + 53 * trajectory_index)
            trajectory_id = f"{preset}-{trajectory_seed}"
            if is_ood: ood_ids.append(trajectory_id)
            overrides = {
                "seed": trajectory_seed, "run_index": 0, "fps": fps, "time_scale": 1.0,
                "duration_s": (frames_per_trajectory + 25) / fps,
                "camera": {"width_px": camera_width_px, "height_px": camera_height_px,
                           "focal_length_px": focal_length_px, "beacon_radius_px": beacon_radius_px,
                           "noise_sigma": camera_noise_sigma, "threshold": 40},
                "vision": {"algorithm": "gradient_centroid", "correction": "cnn_residual" if use_cnn else "none",
                           "cnn_correction": use_cnn, "threshold_strategy": "background_sigma",
                           "background_sigma_k": 3.0, "fixed_threshold": 40},
                "cnn": {"enabled": use_cnn}, "estimator": "akf_r", "predictor": "none", "controller": "pid",
                "prediction_horizon_s": 0.0,
            }
            if is_ood:
                overrides["disturbance"] = {"manoeuvre": {"enabled": True, "start_s": 0.35,
                                                           "velocity_impulse_m_s": [8, -5, 2]},
                                                "vibration": {"enabled": True, "level": "OOD",
                                                              "stochastic_sigma_rad": 0.0008}}
            config = resolve_config(preset, "BEST_CLASSICAL_ESTIMATOR", overrides)
            engine = SimulationEngine(config=config, retain_frames=0); rows = []; previous_timestamp = None
            for frame_index in range(frames_per_trajectory + 20):
                _, telemetry = engine.step(publish=False, annotate=False)
                if frame_index < 20: continue
                truth = telemetry.get("ground_truth_pixel")
                estimate = telemetry.get("estimated_pixel")
                if truth is None or estimate is None: continue
                rows.append(dict(timestamp=float(telemetry["timestamp"]), truth=np.asarray(truth, dtype=np.float32),
                                 feature=_state_row(telemetry, previous_timestamp), measurement_valid=telemetry["measurement_valid"],
                                 prediction_only=telemetry["estimator_prediction_only"], scenario=telemetry["scenario"]))
                previous_timestamp = float(telemetry["timestamp"])
            trajectory_records[trajectory_id] = rows
    split_manifest = trajectory_split(trajectory_records, ood_ids, seed)
    group_to_split = {group: split for split, groups in split_manifest.items() for group in groups}
    sequences, targets, cv_residuals, horizons, metadata = [], [], [], [], []
    for trajectory_id, rows in trajectory_records.items():
        if not rows: continue
        times = np.asarray([row["timestamp"] for row in rows])
        for end_index in range(history_frames - 1, len(rows) - 1):
            current = rows[end_index]
            for requested_horizon in horizons_s:
                desired_time = current["timestamp"] + float(requested_horizon)
                future_index = int(np.searchsorted(times, desired_time, side="left"))
                if future_index >= len(rows): continue
                if abs(float(times[future_index]) - desired_time) <= 1e-9:
                    future_truth = rows[future_index]["truth"]
                else:
                    previous_index = future_index - 1
                    if previous_index < 0: continue
                    span = float(times[future_index] - times[previous_index])
                    if span <= 0: continue
                    weight = float((desired_time - times[previous_index]) / span)
                    future_truth = ((1.0 - weight) * rows[previous_index]["truth"] +
                                    weight * rows[future_index]["truth"])
                actual_horizon = float(requested_horizon)
                if actual_horizon <= 0: continue
                sequence = np.stack([row["feature"] for row in rows[end_index - history_frames + 1:end_index + 1]])
                current_position = sequence[-1, :2]
                cv_position = current_position + sequence[-1, 2:4] * actual_horizon
                sequences.append(sequence); targets.append(future_truth - current_position)
                cv_residuals.append(future_truth - cv_position); horizons.append(actual_horizon)
                metadata.append(dict(sample_index=len(sequences) - 1, trajectory_id=trajectory_id,
                                     split=group_to_split[trajectory_id], scenario=current["scenario"],
                                     history_start_s=float(rows[end_index - history_frames + 1]["timestamp"]),
                                     history_end_s=float(current["timestamp"]), label_timestamp_s=float(desired_time),
                                     requested_horizon_s=float(requested_horizon), actual_horizon_s=actual_horizon,
                                     missing_measurements=int(sum(not row["measurement_valid"] for row in rows[end_index-history_frames+1:end_index+1])),
                                     future_ground_truth_evaluation_only=future_truth.astype(float).tolist()))
    if not sequences: raise RuntimeError("No temporal samples were generated")
    np.savez_compressed(directory / "arrays.npz", sequences=np.asarray(sequences, dtype=np.float32),
                        targets=np.asarray(targets, dtype=np.float32), cv_residuals=np.asarray(cv_residuals, dtype=np.float32),
                        horizons=np.asarray(horizons, dtype=np.float32))
    (directory / "samples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = dict(dataset_id=dataset_id, dataset_version=TEMPORAL_DATASET_VERSION,
                    created_at=datetime.now(timezone.utc).isoformat(), sample_count=len(sequences),
                    trajectory_count=len(trajectory_records), scenario_distribution=dict(Counter(row["scenario"] for row in metadata)),
                    split_counts=dict(Counter(row["split"] for row in metadata)), split_manifest=split_manifest,
                    seed_manifest=sorted({int(group.rsplit("-", 1)[1]) for group in trajectory_records}),
                    split_seed=seed, history_frames=history_frames, feature_schema=list(TEMPORAL_FEATURE_SCHEMA),
                    horizons_s=list(map(float, horizons_s)), upstream_estimator="akf_r",
                    upstream_correction_model="cnn-tiny-residual-v4" if use_cnn else None,
                    simulation_schema_version=SIMULATION_SCHEMA_VERSION, physics_model_version=PHYSICS_MODEL_VERSION,
                    generation_definition=definition, ood_definition="unseen aggressive manoeuvre and vibration magnitude")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"directory": str(directory), **manifest}


def load_temporal_dataset(directory: str | Path):
    directory = Path(directory)
    with np.load(directory / "arrays.npz") as source:
        arrays = {name: source[name].copy() for name in source.files}
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return arrays, samples, manifest
