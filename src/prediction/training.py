"""Comparable GRU/LSTM training on estimator-state histories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from src.core.engine import ROOT
from src.ml.models import build_temporal_model, parameter_count, require_torch


@dataclass(frozen=True)
class TemporalTrainingConfig:
    model_id: str
    kind: str = "gru"
    model_version: str = "1.0.0"
    hidden_size: int = 24
    num_layers: int = 1
    batch_size: int = 64
    learning_rate: float = 0.002
    epochs: int = 20
    early_stopping_patience: int = 5
    weight_decay: float = 1e-5
    training_seed: int = 42
    log_variance_min: float = math.log(0.2 ** 2)
    log_variance_max: float = math.log(100.0 ** 2)


def load_temporal_dataset(directory: str | Path):
    directory = Path(directory)
    with np.load(directory / "arrays.npz") as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return values, samples, manifest


def _indices(samples, split):
    return np.asarray([index for index, row in enumerate(samples) if row["split"] == split], dtype=int)


def _normalizer(sequence, indices):
    train = sequence[indices].reshape(-1, sequence.shape[-1]).astype(np.float64)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _predict(model, sequence, horizons, batch_size, log_min, log_max, torch):
    mean, variance = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequence), batch_size):
            x = torch.from_numpy(sequence[start:start + batch_size]).float()
            tau = torch.from_numpy(horizons[start:start + batch_size]).float()
            output = model(x, tau)
            mean.append(output[:, :2].cpu().numpy())
            variance.append(torch.exp(output[:, 2:].clamp(log_min, log_max)).cpu().numpy())
    if not mean:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(mean), np.concatenate(variance)


def _metrics(prediction, target, variance, none_error, cv_error):
    if len(target) == 0:
        return {"samples": 0, "rmse_px": None}
    difference = prediction - target
    error = np.linalg.norm(difference, axis=1)
    variance = np.clip(variance, 1e-8, None)
    mahalanobis = np.sqrt(np.sum(difference * difference / variance, axis=1))
    nll = 0.5 * np.sum(difference * difference / variance + np.log(variance), axis=1)
    return dict(
        samples=int(len(target)),
        rmse_px=float(np.sqrt(np.mean(error ** 2))),
        mean_error_px=float(np.mean(error)),
        median_px=float(np.median(error)),
        p95_px=float(np.percentile(error, 95)),
        maximum_px=float(np.max(error)),
        none_rmse_px=float(np.sqrt(np.mean(np.sum(none_error * none_error, axis=1)))),
        cv_rmse_px=float(np.sqrt(np.mean(np.sum(cv_error * cv_error, axis=1)))),
        nll=float(np.mean(nll)),
        mean_predicted_sigma_px=float(np.mean(np.sqrt(variance))),
        coverage_1sigma_percent=float(100 * np.mean(mahalanobis <= 1)),
        coverage_2sigma_percent=float(100 * np.mean(mahalanobis <= 2)),
        coverage_3sigma_percent=float(100 * np.mean(mahalanobis <= 3)),
    )


def train_temporal_model(dataset_directory: str | Path, config: TemporalTrainingConfig,
                         models_root: str | Path = ROOT / "models" / "temporal") -> dict:
    if config.kind not in {"gru", "lstm", "gru_residual_cv"}:
        raise ValueError("kind must be gru, lstm or gru_residual_cv")
    torch, _ = require_torch()
    random.seed(config.training_seed)
    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    arrays, samples, manifest = load_temporal_dataset(dataset_directory)
    train_indices, validation_indices = _indices(samples, "train"), _indices(samples, "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Dataset needs non-empty trajectory-safe train and validation splits")
    normalizer = _normalizer(arrays["sequences"], train_indices)
    sequence = ((arrays["sequences"] - np.asarray(normalizer["mean"], dtype=np.float32)) /
                np.asarray(normalizer["std"], dtype=np.float32))
    target = arrays["cv_residuals"] if config.kind == "gru_residual_cv" else arrays["targets"]
    model = build_temporal_model(config.kind, sequence.shape[2], config.hidden_size, config.num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    rng = np.random.default_rng(config.training_seed)
    history, best_score, best_state, stale = [], math.inf, None, 0

    def loss_for(output, expected):
        log_variance = output[:, 2:].clamp(config.log_variance_min, config.log_variance_max)
        error = output[:, :2] - expected
        return 0.5 * torch.mean(torch.sum(error * error / torch.exp(log_variance) + log_variance, dim=1))

    for epoch in range(config.epochs):
        model.train()
        order = train_indices.copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), config.batch_size):
            selected = order[start:start + config.batch_size]
            x = torch.from_numpy(sequence[selected]).float()
            tau = torch.from_numpy(arrays["horizons"][selected]).float()
            expected = torch.from_numpy(target[selected]).float()
            optimizer.zero_grad()
            output = model(x, tau)
            loss = loss_for(output, expected)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction, variance = _predict(
            model, sequence[validation_indices], arrays["horizons"][validation_indices],
            config.batch_size, config.log_variance_min, config.log_variance_max, torch,
        )
        validation = _metrics(
            prediction, target[validation_indices], variance,
            arrays["targets"][validation_indices], arrays["cv_residuals"][validation_indices],
        )
        scheduler.step(validation["nll"])
        history.append(dict(
            epoch=epoch + 1, training_loss=float(np.mean(losses)), validation_loss=validation["nll"],
            position_rmse_px=validation["rmse_px"], median_error_px=validation["median_px"],
            p95_error_px=validation["p95_px"], uncertainty_nll=validation["nll"],
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        ))
        if validation["nll"] < best_score - 1e-6:
            best_score, stale = validation["nll"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state or final_state)
    val_prediction, val_variance = _predict(
        model, sequence[validation_indices], arrays["horizons"][validation_indices],
        config.batch_size, config.log_variance_min, config.log_variance_max, torch,
    )
    ratio = (val_prediction - target[validation_indices]) ** 2 / np.clip(val_variance, 1e-8, None)
    calibration = float(np.clip(np.mean(ratio), 0.1, 25.0))
    weights_hash = _state_hash(model)
    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint = dict(
        model_type="TEMPORAL_PREDICTOR", model_id=config.model_id, model_version=config.model_version,
        kind=config.kind, target_mode="residual_cv" if config.kind == "gru_residual_cv" else "direct_displacement",
        model_state=model.state_dict(), input_feature_count=int(sequence.shape[2]),
        hidden_size=config.hidden_size, num_layers=config.num_layers,
        history_frames=int(sequence.shape[1]), normalizer=normalizer,
        log_variance_bounds=[config.log_variance_min, config.log_variance_max],
        uncertainty_calibration=calibration, weights_hash=weights_hash,
        dataset_id=manifest["dataset_id"], dataset_version=manifest["dataset_version"],
        split_manifest=manifest["split_manifest"], training_config=asdict(config), created_at=created_at,
    )
    models_root = Path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    best_path = models_root / f"{config.model_id}-best.pt"
    final_path = models_root / f"{config.model_id}-final.pt"
    torch.save(checkpoint, best_path)
    torch.save({**checkpoint, "model_state": final_state}, final_path)
    reports, horizon_reports, scenario_reports = {}, {}, {}
    for split in ("train", "validation", "test", "ood_test"):
        selected = _indices(samples, split)
        prediction, variance = _predict(
            model, sequence[selected], arrays["horizons"][selected], config.batch_size,
            config.log_variance_min, config.log_variance_max, torch,
        )
        reports[split] = _metrics(
            prediction, target[selected], variance * calibration,
            arrays["targets"][selected], arrays["cv_residuals"][selected],
        )
    test_indices = _indices(samples, "test")
    for horizon in sorted(set(float(v) for v in arrays["horizons"][test_indices])):
        selected = test_indices[np.isclose(arrays["horizons"][test_indices], horizon)]
        prediction, variance = _predict(
            model, sequence[selected], arrays["horizons"][selected], config.batch_size,
            config.log_variance_min, config.log_variance_max, torch,
        )
        horizon_reports[f"{horizon:.3f}"] = _metrics(
            prediction, target[selected], variance * calibration,
            arrays["targets"][selected], arrays["cv_residuals"][selected],
        )
    for split in ("test", "ood_test"):
        for scenario in sorted({row["scenario"] for row in samples if row["split"] == split}):
            selected = np.asarray([
                index for index, row in enumerate(samples)
                if row["split"] == split and row["scenario"] == scenario
            ], dtype=int)
            prediction, variance = _predict(
                model, sequence[selected], arrays["horizons"][selected], config.batch_size,
                config.log_variance_min, config.log_variance_max, torch,
            )
            scenario_reports[f"{split}:{scenario}"] = _metrics(
                prediction, target[selected], variance * calibration,
                arrays["targets"][selected], arrays["cv_residuals"][selected],
            )
    latency = []
    for index in test_indices[:50]:
        started = perf_counter()
        _predict(model, sequence[index:index + 1], arrays["horizons"][index:index + 1], 1,
                 config.log_variance_min, config.log_variance_max, torch)
        latency.append((perf_counter() - started) * 1000)
    metadata = dict(
        model_id=config.model_id, model_version=config.model_version, kind=config.kind,
        weights_hash=weights_hash, dataset_id=manifest["dataset_id"],
        parameter_count=parameter_count(model), model_size_bytes=best_path.stat().st_size,
        history_frames=int(sequence.shape[1]), feature_schema=manifest["feature_schema"],
        horizons_s=manifest["horizons_s"], upstream_estimator=manifest["upstream_estimator"],
        upstream_correction_model=manifest["upstream_correction_model"],
        uncertainty_calibration=calibration, mean_latency_ms=float(np.mean(latency)),
        p95_latency_ms=float(np.percentile(latency, 95)), checkpoint=str(best_path), created_at=created_at,
    )
    registry_path = ROOT / "models" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    registry["temporal"] = [entry for entry in registry.get("temporal", [])
                            if entry.get("model_id") != config.model_id] + [metadata]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    report = dict(kind="TEMPORAL_MODEL_BENCHMARK", model=metadata,
                  split_results=reports, horizon_results=horizon_reports,
                  scenario_results=scenario_reports, training_history=history)
    (models_root / f"{config.model_id}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
