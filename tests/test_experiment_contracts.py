import math
import numpy as np
import pytest
from src.core.engine import SimulationEngine
from src.core.contracts import Measurement, TrackingState, GimbalState
from src.core.registry import Stage, validate_config
from src.core.metrics import Metrics, PATStateMachine, PATState
from src.core.config import LockConfig
from src.core.scenarios import resolve_config


def strip_runtime_timings(telemetry):
    telemetry.pop('processing_latency_ms')
    telemetry.pop('vision_latency_ms')
    telemetry.pop('estimator_latency_ms')
    telemetry['measurement_quality'].pop('processing_latency_ms',None)
    for field in ('effective_fps','mean_processing_latency_ms','p95_processing_latency_ms',
                  'mean_vision_latency_ms','p95_vision_latency_ms','mean_estimator_latency_ms','p95_estimator_latency_ms'):
        telemetry['metrics'].pop(field)


def test_repeatable_frames_telemetry_and_reset():
    a, b = SimulationEngine(), SimulationEngine()
    first = None
    for _ in range(8):
        fa, ta = a.step()
        fb, tb = b.step()
        assert np.array_equal(fa, fb)
        for telemetry in (ta,tb):
            strip_runtime_timings(telemetry)
        assert ta == tb
        if first is None:
            first = fa.copy(), ta
    a.reset()
    frame, telemetry = a.step()
    assert np.array_equal(frame, first[0])
    strip_runtime_timings(telemetry)
    assert telemetry == first[1]
    config = b.config_snapshot()
    config['seed'] += 1
    b.reset(config)
    assert not np.array_equal(frame, b.step()[0])


def test_centroid_measurement_contract():
    import cv2
    e = SimulationEngine()
    frame = np.zeros((480,640,3), dtype=np.uint8)
    cv2.circle(frame, (150,210), 8, (255,255,255), -1)
    m = e.stages[Stage.VISION].measure(frame, 2.5)
    assert m.valid and m.timestamp == 2.5
    assert np.allclose(m.pixel, [150,210], atol=.1)
    assert not e.stages[Stage.VISION].measure(np.zeros_like(frame), 3).valid


def test_kalman_missing_measurement_predicts_and_controller_is_finite():
    e = SimulationEngine()
    estimator = e.stages[Stage.ESTIMATOR]
    assert not estimator.update(Measurement(None,0), .1).valid
    for i in range(10):
        state = estimator.update(Measurement((320+i,240),i*.1), .1)
    predicted = estimator.update(Measurement(None,1), .1)
    assert predicted.valid and predicted.x > state.x
    assert predicted.covariance.shape == (4,4)
    command = e.stages[Stage.CONTROLLER].compute(predicted, GimbalState(0,0), (320,240), .1)
    assert np.isfinite([command.pan,command.tilt]).all()
    assert command.pan > 0
    for _ in range(100):
        before = e.actuator.pan_rad
        e.actuator.step(100,-100,.1)
        assert abs(e.actuator.pan_rad-before) <= e.config.pid.max_rate_rad_s*.1+1e-12
        assert abs(e.actuator.tilt_rate_rad_s) <= e.config.pid.max_rate_rad_s
        assert -1.2 <= e.actuator.tilt_rad <= 1.2


@pytest.mark.parametrize('stage,name', [('estimator','ukf'),('predictor','transformer'),('controller','reinforcement_learning'),('vision','cnn'),('motion_model','uav'),('reacquisition','scan')])
def test_unavailable_algorithm_rejected_without_reset(stage, name):
    e = SimulationEngine()
    e.step()
    config = e.config_snapshot()
    if stage == 'vision':
        config['vision']['algorithm'] = name
    else:
        config[stage] = name
    with pytest.raises(ValueError, match='Unavailable'):
        e.reset(config)
    assert e.frame_number == 1


def test_numeric_and_unimplemented_config_validation():
    e = SimulationEngine()
    for field, value in [('fps',0),('seed',-1),('time_scale',float('nan'))]:
        c = e.config_snapshot()
        c[field] = value
        with pytest.raises(ValueError):
            validate_config(c)
    c = e.config_snapshot()
    c['vision']['cnn_correction'] = True
    with pytest.raises(ValueError, match='CNN'):
        validate_config(c)


def test_metrics_known_errors_and_dropout():
    m = Metrics()
    for error, valid, locked in [(3,True,True),(4,True,False),(None,False,False)]:
        m.update(dict(true_tracking_error_px=error, measurement_valid=valid, locked=locked))
    s = m.summary()
    assert s['rmse_tracking_error_px'] == pytest.approx(math.sqrt(12.5))
    assert s['mean_tracking_error_px'] == 3.5
    assert s['max_tracking_error_px'] == 4
    assert s['measurement_dropout_count'] == 1
    assert s['locked_percentage'] == pytest.approx(100/3)


def test_lock_dwell_loss_and_recovery():
    lock = PATStateMachine(LockConfig(lock_error_threshold_px=5, lock_required_frames=2, acquisition_required_frames=2, max_missing_frames=2, reacquisition_required_frames=2))
    assert lock.update(False,None) == PATState.SEARCH
    assert lock.update(True,4) == PATState.ACQUIRE
    assert lock.update(True,4) == PATState.TRACK
    assert lock.update(True,4) == PATState.TRACK
    assert lock.update(True,4) == PATState.LOCKED
    assert lock.update(False,4) == PATState.LOCKED
    assert lock.update(False,4) == PATState.LOST
    assert lock.update(False,4) == PATState.REACQUIRE
    assert lock.update(True,4) == PATState.REACQUIRE
    assert lock.update(True,4) == PATState.TRACK
    assert lock.update(True,100) == PATState.LOST


def test_passthrough_closed_loop_and_no_fake_prediction():
    c = SimulationEngine().config_snapshot()
    c['estimator'] = 'none'
    e = SimulationEngine(config=c)
    for _ in range(210):
        _, t = e.step()
    assert t['tracking_error_px'] < 5
    assert t['estimated_pixel'] == t['opencv_pixel']
    assert t['predicted_pixel'] is None and t['kalman_pixel'] is None
    assert not e.stages[Stage.ESTIMATOR].update(Measurement(None,10),.1).valid


def test_ground_truth_does_not_replace_missing_measurements():
    class BlindVision:
        def measure(self, frame, timestamp):
            return Measurement(None,timestamp)
    e = SimulationEngine(config=resolve_config(overrides={"lock":{"start_locked":False}}))
    e.stages[Stage.VISION] = BlindVision()
    _, t = e.step()
    assert t['ground_truth_pixel'] is not None
    assert t['estimated_pixel'] is None
    assert t['controller_output'] == [0,0]
    assert not t['locked']


def test_disturbance_toggles_and_timestamps():
    e = SimulationEngine()
    c = e.config_snapshot()
    c['disturbance'].update(blur=False,gaussian_noise=False)
    e.reset(c)
    frame = np.zeros((20,20,3),dtype=np.uint8)
    frame[10,10] = 255
    assert np.array_equal(e.stages[Stage.DISTURBANCE].image(frame,0),frame)
    _, t = e.step()
    assert t['timestamp'] == pytest.approx(1/30)
    assert t['simulation_time_s'] == pytest.approx(2)
