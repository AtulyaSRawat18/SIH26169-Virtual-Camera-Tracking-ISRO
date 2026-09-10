"""Versioned, trajectory-safe simulator dataset for residual spot correction."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.core.config import PHYSICS_MODEL_VERSION, SIMULATION_SCHEMA_VERSION
from src.core.engine import ROOT, SimulationEngine
from src.core.scenarios import resolve_config
from src.ml.features import AUX_FEATURE_SCHEMA, auxiliary_features, fixed_roi


ML_DATA_SPLIT_SEED = 42
DATASET_VERSION = "spot-residual-1.2.0"
DEFAULT_PRESETS = (
    "SAT_SAT_STRESS", "GROUND_SAT_TURBULENT", "UAV_GROUND_WINDY",
    "UAV_UAV_MODERATE", "UAV_SAT_NOMINAL",
)
OOD_PRESET = "UAV_UAV_AGGRESSIVE"


def _stable_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def trajectory_split(groups: Iterable[str], ood_groups: Iterable[str] = (), seed: int = ML_DATA_SPLIT_SEED) -> dict:
    ood = set(ood_groups)
    ordinary = sorted(set(groups) - ood)
    rng = np.random.default_rng(seed)
    by_scenario: dict[str, list[str]] = {}
    for group in ordinary:
        by_scenario.setdefault(group.rsplit("-", 1)[0], []).append(group)
    split = {"train": [], "validation": [], "test": [], "ood_test": sorted(ood)}
    for scenario in sorted(by_scenario):
        members = by_scenario[scenario]
        rng.shuffle(members)
        if len(members) >= 3:
            validation = members[-2]
            testing = members[-1]
            split["train"].extend(members[:-2])
            split["validation"].append(validation)
            split["test"].append(testing)
        elif len(members) == 2:
            split["train"].append(members[0])
            split["test"].append(members[1])
        else:
            split["train"].extend(members)
    return split


def _trajectory_config(preset: str, seed: int, ood: bool):
    overrides = {
        "seed": seed, "run_index": 0, "fps": 20, "time_scale": 1.0, "duration_s": 4.0,
        "camera": {"width_px": 160, "height_px": 120, "focal_length_px": 130,
                   "beacon_radius_px": 5, "noise_sigma": 10, "threshold": 45},
        "vision": {"algorithm": "gradient_centroid", "correction": "none", "cnn_correction": False,
                   "threshold_strategy": "background_sigma", "background_sigma_k": 3.0,
                   "fixed_threshold": 45, "roi_padding_px": 7},
        "estimator": "akf_r", "predictor": "none", "controller": "pid",
    }
    if ood:
        overrides.update({
            "camera": {**overrides["camera"], "noise_sigma": 20, "beacon_radius_px": 3},
            "disturbance": {"glare": {"enabled": True, "intensity": 105, "radius_fraction": 0.22},
                            "beacon": {"nominal_intensity": 165, "elliptical_ratio": 2.8,
                                       "radius_variation_fraction": 0.35},
                            "sensor": {"defocus_sigma_px": 2.5, "background_gradient": 55}},
        })
    return resolve_config(preset, "BEST_CLASSICAL_LOCALIZER", overrides)


def generate_spot_dataset(output_root: str | Path = ROOT / "data" / "ml" / "spot",
                          trajectories_per_scenario: int = 4, frames_per_trajectory: int = 24,
                          roi_size_px: int = 32, seed: int = ML_DATA_SPLIT_SEED) -> dict:
    if trajectories_per_scenario < 1 or frames_per_trajectory < 2:
        raise ValueError("Dataset requires trajectories and at least two frames per trajectory")
    definition = dict(version=DATASET_VERSION, presets=list(DEFAULT_PRESETS), ood_preset=OOD_PRESET,
                      trajectories_per_scenario=trajectories_per_scenario,
                      frames_per_trajectory=frames_per_trajectory, roi_size_px=roi_size_px, seed=seed)
    dataset_id = f"spot-{_stable_id(definition)}"
    directory = Path(output_root) / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []
    auxiliaries: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    sample_metadata: list[dict] = []
    hard_negatives: list[dict] = []
    trajectory_ids: list[str] = []
    ood_ids: list[str] = []
    preset_plan = [(name, False) for name in DEFAULT_PRESETS] + [(OOD_PRESET, True)]
    for preset_index, (preset, is_ood) in enumerate(preset_plan):
        for trajectory_index in range(trajectories_per_scenario):
            trajectory_seed = int(seed + 1009 * preset_index + 37 * trajectory_index)
            trajectory_id = f"{preset}-{trajectory_seed}"
            trajectory_ids.append(trajectory_id)
            if is_ood:
                ood_ids.append(trajectory_id)
            config = _trajectory_config(preset, trajectory_seed, is_ood)
            engine = SimulationEngine(config=config, retain_frames=0)
            for frame_index in range(frames_per_trajectory + 5):
                engine.step(publish=True, annotate=False)
                if frame_index < 5:
                    continue
                frame, telemetry, measurement = engine.raw_snapshot()
                truth = telemetry.get("ground_truth_pixel")
                if measurement is None or not measurement.valid or truth is None:
                    continue
                error = np.asarray(truth, dtype=float) - np.asarray(measurement.pixel, dtype=float)
                association_failure = bool(telemetry.get("incorrect_target_selection", False) or np.linalg.norm(error) > 12.0)
                if association_failure:
                    hard_negatives.append(dict(trajectory_id=trajectory_id, frame=frame_index,
                                               seed=trajectory_seed, scenario=preset,
                                               reason="TARGET_ASSOCIATION", error_px=float(np.linalg.norm(error))))
                    continue
                roi, origin, clipped = fixed_roi(frame, measurement.pixel, roi_size_px)
                images.append(roi)
                auxiliaries.append(auxiliary_features(measurement, origin, roi_size_px))
                residuals.append(error.astype(np.float32))
                sample_metadata.append(dict(
                    sample_index=len(images) - 1, trajectory_id=trajectory_id, seed=trajectory_seed,
                    scenario=config.scenario.value, scenario_preset=preset, frame=frame_index,
                    timestamp=telemetry["timestamp"], split=None, ood=is_ood,
                    production_roi=True, oracle_roi=False, roi_origin=list(origin), roi_clipped=clipped,
                    true_beacon_centre=list(map(float, truth)), classical_estimate=list(map(float, measurement.pixel)),
                    disturbance_parameters=config.disturbance.model_dump(mode="json"),
                    simulation_schema_version=SIMULATION_SCHEMA_VERSION,
                    physics_model_version=PHYSICS_MODEL_VERSION,
                    camera_calibration_version=config.estimation.camera_calibration_version,
                ))
    if not images:
        raise RuntimeError("No valid localization samples were generated")
    split_manifest = trajectory_split(trajectory_ids, ood_ids, seed)
    group_to_split = {group: split for split, groups in split_manifest.items() for group in groups}
    for row in sample_metadata:
        row["split"] = group_to_split[row["trajectory_id"]]
    np.savez_compressed(directory / "arrays.npz", images=np.asarray(images, dtype=np.float32),
                        auxiliary=np.asarray(auxiliaries, dtype=np.float32),
                        residual=np.asarray(residuals, dtype=np.float32))
    (directory / "samples.json").write_text(json.dumps(sample_metadata, indent=2), encoding="utf-8")
    counts = Counter(row["split"] for row in sample_metadata)
    scenarios = Counter(row["scenario"] for row in sample_metadata)
    manifest = dict(dataset_id=dataset_id, dataset_version=DATASET_VERSION,
                    created_at=datetime.now(timezone.utc).isoformat(), sample_count=len(images),
                    trajectory_count=len(set(trajectory_ids)), scenario_distribution=dict(scenarios),
                    split_counts=dict(counts), split_manifest=split_manifest,
                    seed_manifest=sorted(set(row["seed"] for row in sample_metadata)),
                    split_seed=seed, roi_size_px=roi_size_px, feature_schema=list(AUX_FEATURE_SCHEMA),
                    simulation_schema_version=SIMULATION_SCHEMA_VERSION,
                    physics_model_version=PHYSICS_MODEL_VERSION,
                    camera_calibration_version="pinhole-v1", hard_negative_count=len(hard_negatives),
                    hard_negatives=hard_negatives, generation_definition=definition)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"directory": str(directory), **manifest}


def load_spot_dataset(directory: str | Path) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    directory = Path(directory)
    with np.load(directory / "arrays.npz") as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return values, samples, manifest
