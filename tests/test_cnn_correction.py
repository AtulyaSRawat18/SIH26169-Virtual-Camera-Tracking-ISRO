import numpy as np
import pytest

from src.core.config import ExperimentConfig
from src.core.contracts import CorrectionFailureReason, Measurement
from src.core.scenarios import resolve_config
from src.estimation.filters import AdaptivePixelKF
from src.ml.correctors import CNNResidualCorrector
from src.ml.dataset import generate_spot_dataset, load_spot_dataset, trajectory_split
from src.ml.features import AUX_FEATURE_SCHEMA, fixed_roi
from src.ml.models import build_spot_model


def test_trajectory_split_is_deterministic_and_has_no_leakage():
    groups = [f"trajectory-{index}" for index in range(12)]
    first = trajectory_split(groups, ["trajectory-11"], 42)
    second = trajectory_split(reversed(groups), ["trajectory-11"], 42)
    assert first == second
    sets = [set(value) for value in first.values()]
    assert all(not (left & right) for index, left in enumerate(sets) for right in sets[index + 1:])
    assert first["ood_test"] == ["trajectory-11"]


def test_production_roi_is_deterministic_and_centred_on_classical_measurement():
    frame = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
    first, origin, clipped = fixed_roi(frame, (31.25, 29.75), 32)
    second, second_origin, _ = fixed_roi(frame, (31.25, 29.75), 32)
    assert np.array_equal(first, second)
    assert origin == second_origin and not clipped
    assert first.shape == (32, 32)


def test_small_dataset_preserves_truth_residual_seed_and_safe_splits(tmp_path):
    report = generate_spot_dataset(tmp_path, trajectories_per_scenario=1,
                                   frames_per_trajectory=6, seed=1337)
    arrays, samples, manifest = load_spot_dataset(report["directory"])
    assert arrays["images"].shape[1:] == (32, 32)
    assert arrays["auxiliary"].shape[1] == len(AUX_FEATURE_SCHEMA)
    assert manifest["split_seed"] == 1337 and manifest["dataset_version"]
    assert set(manifest["split_manifest"]) == {"train", "validation", "test", "ood_test"}
    for index, sample in enumerate(samples):
        expected = np.asarray(sample["true_beacon_centre"]) - np.asarray(sample["classical_estimate"])
        assert np.allclose(arrays["residual"][index], expected)
        assert sample["seed"] is not None and sample["trajectory_id"]


def test_cnn_shape_positive_variance_and_missing_model_fallback():
    torch = pytest.importorskip("torch")
    model = build_spot_model(len(AUX_FEATURE_SCHEMA))
    output = model(torch.zeros((2, 1, 32, 32)), torch.zeros((2, len(AUX_FEATURE_SCHEMA))))
    assert output.shape == (2, 4)
    assert torch.all(torch.exp(output[:, 2:]) > 0)
    data = resolve_config().model_dump(mode="json")
    data["pipeline_preset"] = "CUSTOM"
    data["vision"].update(correction="cnn_residual", cnn_correction=True)
    data["cnn"].update(enabled=True, model_path="models/cnn/does-not-exist.pt")
    corrector = CNNResidualCorrector(ExperimentConfig.model_validate(data))
    measurement = Measurement((20.5, 21.25), 0.0, covariance=np.eye(2))
    corrected = corrector.correct(np.zeros((64, 64, 3), dtype=np.uint8), measurement)
    assert corrected.pixel == measurement.pixel and corrected.classical_pixel == measurement.pixel
    assert corrected.correction_fallback
    assert corrected.correction_failure_reason == CorrectionFailureReason.MODEL_UNAVAILABLE


def test_learned_measurement_covariance_reaches_adaptive_filter():
    config = resolve_config()
    estimator = AdaptivePixelKF(config, None)
    measurement = Measurement((100.0, 100.0), 0.0, confidence=0.9,
                              covariance=np.diag([0.7, 3.2]))
    covariance, _ = estimator.measurement_covariance(measurement)
    assert covariance.shape == (2, 2)
    assert covariance[1, 1] > covariance[0, 0] > 0
