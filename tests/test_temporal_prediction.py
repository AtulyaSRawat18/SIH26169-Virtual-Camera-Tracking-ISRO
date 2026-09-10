import numpy as np
import pytest

from src.control.ff_pid import FeedForwardPIDController
from src.core.contracts import GimbalState, PredictedState, PredictionInput, TrackingState
from src.core.scenarios import resolve_config
from src.core.stages import PIDController
from src.ml.models import build_temporal_model
from src.prediction.dataset import generate_temporal_dataset
from src.prediction.predictors import (ConstantAccelerationPredictor, ConstantVelocityPredictor,
                                       GRUPredictor)


def request(states, horizon):
    return PredictionInput(states[-1], tuple(states), tuple({} for _ in states),
                           tuple(state.timestamp for state in states), horizon)


def test_cv_matches_analytic_constant_velocity_for_positive_negative_and_zero_rates():
    config = resolve_config()
    predictor = ConstantVelocityPredictor(config)
    state = TrackingState(x=10, y=20, vx=-4, vy=0, timestamp=2, valid=True)
    prediction = predictor.predict(request([state], 0.25))
    assert prediction.pixel == pytest.approx((9, 20))
    assert prediction.vx == -4 and prediction.vy == 0


def test_ca_uses_only_past_estimator_velocity_and_physical_time():
    config = resolve_config()
    states = [TrackingState(x=0, y=0, vx=index, vy=-2*index, timestamp=index*.1, valid=True)
              for index in range(4)]
    prediction = ConstantAccelerationPredictor(config).predict(request(states, 0.2))
    assert prediction.ax == pytest.approx(10)
    assert prediction.ay == pytest.approx(-20)
    assert prediction.x == pytest.approx(0 + 3*.2 + .5*10*.2**2)


def test_temporal_networks_share_shape_and_positive_uncertainty():
    torch = pytest.importorskip("torch")
    sequence = torch.zeros((3, 10, 11))
    horizon = torch.full((3,), .05)
    for kind in ("gru", "lstm", "gru_residual_cv"):
        output = build_temporal_model(kind, 11, hidden_size=12)(sequence, horizon)
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
        assert torch.all(torch.exp(output[:, 2:]) > 0)


def test_neural_missing_model_and_insufficient_history_use_visible_cv_fallback():
    data = resolve_config().model_dump(mode="json")
    data["temporal"].update(history_frames=5, gru_model_path="models/temporal/missing.pt")
    config = type(resolve_config()).model_validate(data)
    states = [TrackingState(x=10, y=10, vx=2, vy=-1, timestamp=.1*index, valid=True) for index in range(3)]
    prediction = GRUPredictor(config).predict(request(states, .1))
    assert prediction.valid and prediction.fallback_used
    assert prediction.failure_reason == "MODEL_UNAVAILABLE"
    assert prediction.pixel == pytest.approx((10.2, 9.9))


def test_ff_pid_zero_feedforward_matches_pid_and_uncertainty_reduces_feedforward():
    base = resolve_config().model_dump(mode="json")
    base["ff_pid"]["gain"] = 0
    zero = type(resolve_config()).model_validate(base)
    state = PredictedState(True, .05, x=330, y=235, vx=20, vy=-10, covariance=np.eye(2))
    ordinary = PIDController(zero, None).compute(state, GimbalState(0, 0), (320, 240), .1)
    augmented = FeedForwardPIDController(zero).compute(state, GimbalState(0, 0), (320, 240), .1)
    assert augmented == ordinary
    base["ff_pid"]["gain"] = 1
    config = type(resolve_config()).model_validate(base)
    controller = FeedForwardPIDController(config)
    low = controller.compute(state, GimbalState(0, 0), (state.x, state.y), .1)
    high_state = PredictedState(**{**state.__dict__, "covariance": np.eye(2)*1000})
    high = FeedForwardPIDController(config).compute(high_state, GimbalState(0, 0), (state.x, state.y), .1)
    assert abs(high.pan) < abs(low.pan)
    assert abs(low.pan) <= config.pid.max_rate_rad_s


def test_temporal_dataset_labels_use_future_time_and_trajectory_safe_split(tmp_path):
    report = generate_temporal_dataset(tmp_path, trajectories_per_scenario=1,
                                       frames_per_trajectory=20, history_frames=5,
                                       horizons_s=(.02, .05), fps=100, seed=202)
    samples = __import__("json").loads((__import__("pathlib").Path(report["directory"])/"samples.json").read_text())
    assert report["dataset_version"] and report["split_seed"] == 202
    groups = {split:set(values) for split,values in report["split_manifest"].items()}
    assert all(not (left & right) for index,left in enumerate(groups.values())
               for right in list(groups.values())[index+1:])
    assert all(row["label_timestamp_s"] > row["history_end_s"] >= row["history_start_s"] for row in samples)
    assert all(row["actual_horizon_s"] > 0 for row in samples)
