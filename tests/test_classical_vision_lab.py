import inspect
import cv2
import numpy as np
import pytest

from src.core.engine import SimulationEngine
from src.core.registry import REGISTRY, Stage, validate_config
from src.experiments.storage import ExperimentStore
from src.experiments.vision_lab import compare_frame, run_vision_benchmark


ALGORITHMS=("binary_centroid","weighted_centroid","gradient_centroid","gaussian_fit")


def clean_case():
    config=SimulationEngine().config_snapshot()
    config["camera"].update(width_px=160,height_px=120,focal_length_px=130)
    config["vision"].update(threshold_strategy="background_sigma",min_component_area=3,max_component_area=1000)
    frame=np.zeros((120,160,3),dtype=np.uint8)
    yy,xx=np.indices((120,160),dtype=float)
    spot=8+225*np.exp(-.5*(((xx-70.35)/3.2)**2+((yy-49.65)/4.0)**2))
    frame[:]=np.uint8(np.clip(spot[...,None],0,255))
    return config,frame,(70.35,49.65)


@pytest.mark.parametrize("algorithm",ALGORITHMS)
def test_classical_algorithms_share_subpixel_contract(algorithm):
    config,frame,truth=clean_case(); localizer=REGISTRY[Stage.VISION][algorithm](validate_config(config),None)
    measurement=localizer.measure(frame,1.25)
    assert measurement.valid and measurement.timestamp==1.25
    assert all(isinstance(value,float) for value in measurement.pixel)
    assert np.linalg.norm(np.asarray(measurement.pixel)-truth)<1.0
    assert measurement.algorithm_name==algorithm
    assert measurement.image_snr_estimate is not None and measurement.quality["candidate_count"]==1
    assert list(inspect.signature(localizer.measure).parameters)==["frame","timestamp"]


def test_empty_and_zero_signal_are_invalid_without_nan():
    config,frame,_=clean_case(); empty=np.zeros_like(frame)
    for algorithm in ALGORITHMS:
        measurement=REGISTRY[Stage.VISION][algorithm](validate_config(config),None).measure(empty,0)
        assert not measurement.valid and measurement.failure_reason is not None
        assert measurement.pixel is None


def test_gaussian_fit_reports_residual_and_covariance_safely():
    config,frame,truth=clean_case(); result=REGISTRY[Stage.VISION]["gaussian_fit"](validate_config(config),None).measure(frame,0)
    assert np.linalg.norm(np.asarray(result.pixel)-truth)<.25
    assert result.fit_error is not None and np.isfinite(result.fit_error)
    assert result.covariance is None or (result.covariance.shape==(2,2) and np.isfinite(result.covariance).all())


def test_same_frame_comparison_and_oracle_roi_truth_separation():
    config,frame,truth=clean_case()
    full=compare_frame(config,frame,truth,ALGORITHMS,"FULL_FRAME",include_images=False)
    oracle=compare_frame(config,frame,truth,ALGORITHMS,"ORACLE_ROI",include_images=False)
    assert len({item["input_checksum"] for item in full["results"]})==1
    assert len({item["input_checksum"] for item in oracle["results"]})==1
    assert all(item["measurement"]["valid"] for item in oracle["results"])
    assert full["source_checksum"]==oracle["source_checksum"]


def test_small_paired_benchmark_is_repeatable_and_persists(tmp_path):
    store=ExperimentStore(tmp_path/"lab.sqlite3"); config=SimulationEngine().config
    first=run_vision_benchmark(config,seeds=(7,8),algorithms=("binary_centroid","weighted_centroid"),
                               conditions=("CLEAN",),frames_per_seed=2,store=store)
    second=run_vision_benchmark(config,seeds=(7,8),algorithms=("binary_centroid","weighted_centroid"),
                                conditions=("CLEAN",),frames_per_seed=2)
    assert [row["source_checksum"] for row in first["rows"]]==[row["source_checksum"] for row in second["rows"]]
    assert store.query_lab_results("CLASSICAL_VISION_BENCHMARK")[0]["result"]["kind"]=="CLASSICAL_VISION_BENCHMARK"
