import numpy as np
import pytest

from src.core.contracts import Measurement
from src.core.engine import SimulationEngine
from src.core.registry import REGISTRY,Stage,validate_config
from src.estimation.filters import AdaptivePixelKF,AngularEKF,AngularUKF,PixelCVKF,sigma_points,weighted_sigma_statistics
from src.estimation.geometry import CameraCalibration,angles_to_pixel,angular_measurement_jacobian,pixel_to_angles
from src.experiments.estimator_lab import compare_estimators_open_loop,record_measurement_sequence


def calibration(): return CameraCalibration(520,510,320,240,640,480)


def test_pixel_angle_round_trips_and_sign_convention():
    c=calibration()
    for pixel in ((320,240),(101.2,40.7),(620,450)):
        assert np.allclose(angles_to_pixel(pixel_to_angles(pixel,c),c),pixel,atol=1e-10)
    angles=np.array([.2,-.15])
    assert np.allclose(pixel_to_angles(angles_to_pixel(angles,c),c),angles,atol=1e-10)
    assert angles_to_pixel((0,.1),c)[1]>c.cy


def test_ekf_analytic_jacobian_matches_finite_difference():
    c=calibration(); state=np.array([.12,-.08,.01,.02]); analytic=angular_measurement_jacobian(state,c)
    numerical=np.zeros((2,4)); epsilon=1e-7
    for index in range(4):
        plus=state.copy();minus=state.copy();plus[index]+=epsilon;minus[index]-=epsilon
        numerical[:,index]=(angles_to_pixel(plus[:2],c)-angles_to_pixel(minus[:2],c))/(2*epsilon)
    assert np.allclose(analytic,numerical,rtol=1e-6,atol=1e-6)


def test_kf_prediction_and_dropout_covariance_growth():
    config=SimulationEngine().config; kf=PixelCVKF(config,None)
    kf.update(Measurement((100.,80.),0),.1)
    kf.x[2:]=[4,-3]
    first=kf.update(Measurement(None,.1),.5); trace1=np.trace(first.covariance)
    assert first.pixel==pytest.approx((102,78.5)) and first.prediction_only
    second=kf.update(Measurement(None,.6),.5)
    assert np.trace(second.covariance)>trace1


def test_ukf_sigma_points_reconstruct_simple_gaussian():
    mean=np.array([1.,2.,3.,4.]);cov=np.diag([.1,.2,.3,.4])
    points,wm,wc,_=sigma_points(mean,cov,.3,2,0)
    reconstructed,reconstructed_cov=weighted_sigma_statistics(points,wm,wc,np.zeros((4,4)))
    assert np.allclose(reconstructed,mean) and np.allclose(reconstructed_cov,cov)
    assert np.allclose(reconstructed_cov,reconstructed_cov.T)


def test_adaptive_r_uses_real_quality_and_bounds():
    config=SimulationEngine().config; akf=AdaptivePixelKF(config,None)
    strong=Measurement((10.,10.),0,confidence=.98,image_snr_estimate=40,quality={"candidate_ambiguity":0,"border_clipped":False,"saturation_fraction":0})
    weak=Measurement((10.,10.),0,confidence=.1,image_snr_estimate=.3,fit_error=60,quality={"candidate_ambiguity":.8,"border_clipped":True,"saturation_fraction":.7})
    strong_r,strong_scale=akf.measurement_covariance(strong);weak_r,weak_scale=akf.measurement_covariance(weak)
    assert weak_scale>strong_scale and np.trace(weak_r)>np.trace(strong_r)
    assert config.estimation.adaptive_r_min_scale<=strong_scale<=config.estimation.adaptive_r_max_scale


def test_gating_rejects_extreme_outlier_without_truth():
    data=SimulationEngine().config_snapshot();data["estimation"]["gate_enabled"]=True
    kf=PixelCVKF(validate_config(data),None);kf.update(Measurement((100.,100.),0),.1)
    result=kf.update(Measurement((600.,450.),.1),.1)
    assert result.prediction_only and not result.measurement_used and kf.measurement_rejection_count==1


@pytest.mark.parametrize("name",("none","kf_cv","ekf_angular","ukf_angular","akf_r"))
def test_estimators_instantiate_and_return_common_state(name):
    config=SimulationEngine().config;estimator=REGISTRY[Stage.ESTIMATOR][name](config,None)
    result=estimator.update(Measurement((322.,238.),0,confidence=.8,image_snr_estimate=12),1/30)
    assert result.valid and result.pixel is not None and result.estimator_name in (name,"kf_cv")
    assert np.isfinite(result.pixel).all()


def test_measurement_replay_is_deterministic_and_shared():
    data=SimulationEngine().config_snapshot();data.update(fps=10,duration_s=.4);data["camera"].update(width_px=100,height_px=80,focal_length_px=75)
    sequence=record_measurement_sequence(data,.4)
    a=compare_estimators_open_loop(sequence,data,("kf_cv","ekf_angular","ukf_angular","akf_r"))
    b=compare_estimators_open_loop(sequence,data,("kf_cv","ekf_angular","ukf_angular","akf_r"))
    for name in a["estimators"]:
        assert a["estimators"][name]["summary"]["rmse_px"]==pytest.approx(b["estimators"][name]["summary"]["rmse_px"])
        assert [row["measurement"] for row in a["estimators"][name]["rows"]]==[sample.measurement.as_dict() for sample in sequence.samples]


@pytest.mark.parametrize("name",("ekf_angular","ukf_angular","akf_r"))
def test_new_estimator_closed_loop_without_controller_branching(name):
    data=SimulationEngine().config_snapshot();data["estimator"]=name;data["pipeline_preset"]="CUSTOM"
    engine=SimulationEngine(config=data)
    for _ in range(10): _,telemetry=engine.step()
    assert telemetry["estimated_pixel"] is not None and np.isfinite(telemetry["controller_output"]).all()
