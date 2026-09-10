"""Reproducible training/evaluation for the heteroscedastic residual spot CNN."""
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
from src.ml.dataset import load_spot_dataset
from src.ml.features import fit_normalizer, normalize_features
from src.ml.models import build_spot_model, parameter_count, require_torch


@dataclass(frozen=True)
class CNNTrainingConfig:
    model_id: str = "cnn-tiny-residual-v4"
    model_version: str = "1.0.0"
    architecture: str = "cnn_tiny"
    use_aux_features: bool = True
    batch_size: int = 32
    learning_rate: float = 0.002
    epochs: int = 18
    early_stopping_patience: int = 5
    weight_decay: float = 1e-5
    training_seed: int = 42
    log_variance_min: float = math.log(0.15 ** 2)
    log_variance_max: float = math.log(20.0 ** 2)


def _set_seeds(seed: int, torch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _indices(samples: list[dict], split: str) -> np.ndarray:
    return np.asarray([index for index, row in enumerate(samples) if row["split"] == split], dtype=int)


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _metrics(prediction: np.ndarray, target: np.ndarray, variance: np.ndarray) -> dict:
    if len(target) == 0:
        return {"samples": 0, "rmse_px": None, "p95_px": None, "nll": None}
    error_vector = prediction - target
    errors = np.linalg.norm(error_vector, axis=1)
    classical = np.linalg.norm(target, axis=1)
    variance = np.clip(variance, 1e-8, None)
    nll = 0.5 * np.sum(error_vector * error_vector / variance + np.log(variance), axis=1)
    mahalanobis = np.sqrt(np.sum(error_vector * error_vector / variance, axis=1))
    return dict(samples=int(len(errors)), rmse_px=float(np.sqrt(np.mean(errors ** 2))),
                mae_px=float(np.mean(errors)), median_px=float(np.median(errors)),
                p95_px=float(np.percentile(errors, 95)), maximum_px=float(np.max(errors)),
                classical_rmse_px=float(np.sqrt(np.mean(classical ** 2))),
                false_correction_percent=float(100 * np.mean(errors > classical)),
                nll=float(np.mean(nll)), mean_predicted_sigma_px=float(np.mean(np.sqrt(variance))),
                coverage_1sigma_percent=float(100 * np.mean(mahalanobis <= 1)),
                coverage_2sigma_percent=float(100 * np.mean(mahalanobis <= 2)),
                coverage_3sigma_percent=float(100 * np.mean(mahalanobis <= 3)))


def _predict(model, images, auxiliary, batch_size, log_min, log_max, torch):
    predictions, variances = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start:start + batch_size, None]).float()
            aux = None if auxiliary is None else torch.from_numpy(auxiliary[start:start + batch_size]).float()
            output = model(x, aux)
            predictions.append(output[:, :2].cpu().numpy())
            variances.append(torch.exp(output[:, 2:].clamp(log_min, log_max)).cpu().numpy())
    if not predictions:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(predictions), np.concatenate(variances)


def train_spot_model(dataset_directory: str | Path, config: CNNTrainingConfig = CNNTrainingConfig(),
                     models_root: str | Path = ROOT / "models" / "cnn") -> dict:
    torch, _ = require_torch(); _set_seeds(config.training_seed, torch)
    arrays, samples, manifest = load_spot_dataset(dataset_directory)
    train_indices = _indices(samples, "train"); validation_indices = _indices(samples, "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Dataset needs non-empty trajectory-safe train and validation splits")
    normalizer = fit_normalizer(arrays["auxiliary"][train_indices])
    auxiliary = normalize_features(arrays["auxiliary"], normalizer) if config.use_aux_features else None
    model = build_spot_model(0 if auxiliary is None else auxiliary.shape[1], config.architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    rng = np.random.default_rng(config.training_seed)
    history, best_state, best_score, stale = [], None, math.inf, 0

    def loss_for(output, target):
        log_variance = output[:, 2:].clamp(config.log_variance_min, config.log_variance_max)
        error = output[:, :2] - target
        return 0.5 * torch.mean(torch.sum(error * error / torch.exp(log_variance) + log_variance, dim=1))

    for epoch in range(config.epochs):
        model.train(); order = train_indices.copy(); rng.shuffle(order); losses = []
        for start in range(0, len(order), config.batch_size):
            selected = order[start:start + config.batch_size]
            image = torch.from_numpy(arrays["images"][selected, None]).float()
            aux = None if auxiliary is None else torch.from_numpy(auxiliary[selected]).float()
            target = torch.from_numpy(arrays["residual"][selected]).float()
            optimizer.zero_grad(); output = model(image, aux); loss = loss_for(output, target)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        prediction, variance = _predict(model, arrays["images"][validation_indices],
                                        None if auxiliary is None else auxiliary[validation_indices],
                                        config.batch_size, config.log_variance_min, config.log_variance_max, torch)
        validation = _metrics(prediction, arrays["residual"][validation_indices], variance)
        validation_loss = validation["nll"]
        scheduler.step(validation_loss)
        history.append(dict(epoch=epoch + 1, training_loss=float(np.mean(losses)), validation_loss=validation_loss,
                            position_rmse_px=validation["rmse_px"], median_error_px=validation["median_px"],
                            p95_error_px=validation["p95_px"], uncertainty_nll=validation["nll"],
                            learning_rate=float(optimizer.param_groups[0]["lr"])))
        if validation_loss < best_score - 1e-6:
            best_score, stale = validation_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state or final_state)
    val_prediction, val_variance = _predict(model, arrays["images"][validation_indices],
                                            None if auxiliary is None else auxiliary[validation_indices],
                                            config.batch_size, config.log_variance_min, config.log_variance_max, torch)
    val_error = val_prediction - arrays["residual"][validation_indices]
    calibration = float(np.clip(np.mean(val_error * val_error / np.clip(val_variance, 1e-8, None)), 0.1, 25.0))
    validation_metrics = _metrics(val_prediction, arrays["residual"][validation_indices], val_variance * calibration)
    weights_hash = _state_hash(model)
    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint = dict(model_type="CNN_RESIDUAL", model_id=config.model_id, model_version=config.model_version,
                      architecture=config.architecture, model_state=model.state_dict(),
                      aux_feature_count=0 if auxiliary is None else int(auxiliary.shape[1]),
                      normalizer=normalizer, roi_size_px=int(manifest["roi_size_px"]),
                      log_variance_bounds=[config.log_variance_min, config.log_variance_max],
                      uncertainty_calibration=calibration, weights_hash=weights_hash,
                      dataset_id=manifest["dataset_id"], dataset_version=manifest["dataset_version"],
                      split_manifest=manifest["split_manifest"], training_config=asdict(config),
                      validation_metrics=validation_metrics, training_history=history,
                      created_at=created_at)
    models_root = Path(models_root); models_root.mkdir(parents=True, exist_ok=True)
    best_path = models_root / f"{config.model_id}-best.pt"
    final_path = models_root / f"{config.model_id}-final.pt"
    torch.save(checkpoint, best_path)
    final_checkpoint = dict(checkpoint, model_state=final_state, checkpoint_kind="final")
    torch.save(final_checkpoint, final_path)
    test_reports = {}
    for split in ("train", "validation", "test", "ood_test"):
        selected = _indices(samples, split)
        prediction, variance = _predict(model, arrays["images"][selected],
                                        None if auxiliary is None else auxiliary[selected], config.batch_size,
                                        config.log_variance_min, config.log_variance_max, torch)
        test_reports[split] = _metrics(prediction, arrays["residual"][selected], variance * calibration)
    scenario_reports = {}
    for split in ("test", "ood_test"):
        for scenario in sorted({row["scenario"] for row in samples if row["split"] == split}):
            selected = np.asarray([
                index for index, row in enumerate(samples)
                if row["split"] == split and row["scenario"] == scenario
            ], dtype=int)
            prediction, variance = _predict(
                model, arrays["images"][selected],
                None if auxiliary is None else auxiliary[selected], config.batch_size,
                config.log_variance_min, config.log_variance_max, torch,
            )
            scenario_reports[f"{split}:{scenario}"] = _metrics(
                prediction, arrays["residual"][selected], variance * calibration,
            )
    timing_indices = _indices(samples, "test")[:min(50, len(_indices(samples, "test")))]
    latency = []
    for index in timing_indices:
        started = perf_counter()
        _predict(model, arrays["images"][index:index + 1], None if auxiliary is None else auxiliary[index:index + 1],
                 1, config.log_variance_min, config.log_variance_max, torch)
        latency.append((perf_counter() - started) * 1000)
    metadata = dict(model_id=config.model_id, model_version=config.model_version, architecture=config.architecture,
                    weights_hash=weights_hash, dataset_id=manifest["dataset_id"],
                    training_config_hash=hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
                    training_seed=config.training_seed, split_seed=manifest["split_seed"], created_at=created_at,
                    roi_size_px=manifest["roi_size_px"], parameter_count=parameter_count(model),
                    model_file_size_bytes=best_path.stat().st_size, uncertainty_calibration=calibration,
                    validation_metrics=validation_metrics, mean_inference_latency_ms=float(np.mean(latency)) if latency else None,
                    p95_inference_latency_ms=float(np.percentile(latency, 95)) if latency else None,
                    checkpoint_path=str(best_path))
    registry_path = models_root.parent / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}
    registry["models"] = [item for item in registry["models"] if item.get("model_id") != config.model_id] + [metadata]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    report = dict(kind="CNN_TRAINING_AND_EVALUATION", model=metadata, dataset=manifest,
                  splits=test_reports, scenario_results=scenario_reports, training_history=history)
    (models_root / f"{config.model_id}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
