from dataclasses import fields, replace
import numpy as np

from src.control.gimbal import GimbalPlant
from src.control.reacquisition import HybridSearch
from src.core.contracts import ActuatorLimits, PredictedState, SearchInput, TrackingState
from src.core.engine import SimulationEngine
from src.core.registry import REGISTRY, Stage
from src.experiments.acquisition_lab import run_acquisition_benchmark, run_search_replay


def compact_config():
    data=SimulationEngine().config_snapshot()
    data["camera"].update(width_px=100,height_px=80,focal_length_px=80)
    data.update(fps=10,duration_s=.6,pipeline_preset="CUSTOM")
    return data


def search_request(config,elapsed=.1):
    centre=(config.camera.width_px/2,config.camera.height_px/2)
    last=TrackingState(centre[0]+8,centre[1]-4,2,-1,0,True,np.eye(4))
    predicted=PredictedState(True,.05,last.x+1,last.y-1,last.vx,last.vy,covariance=np.diag([9,4,1,1]))
    plant=GimbalPlant(config.pid,config.disturbance.gimbal,config.fps)
    return SearchInput(last,predicted,predicted.covariance,plant.reported_state(),
                       ActuatorLimits(config.pid.max_rate_rad_s,
                                      config.disturbance.gimbal.max_acceleration_rad_s2,
                                      *config.disturbance.gimbal.position_limit_rad),
                       centre,config.camera.focal_length_px,elapsed,1/config.fps,elapsed,"REACQUIRE")


def test_search_contract_contains_no_truth():
    names={field.name for field in fields(SearchInput)}
    assert not names & {"truth","ground_truth","true_state","true_pixel"}
    assert {"last_known_state","predicted_state","uncertainty","current_gimbal_state"} <= names


def test_all_search_strategies_are_bounded_and_finite():
    engine=SimulationEngine(); request=search_request(engine.config,.8)
    for name,implementation in REGISTRY[Stage.REACQUISITION].items():
        command=implementation(engine.config,None).search(request)
        if command is None: continue
        assert np.isfinite([command.pan,command.tilt]).all()
        assert max(abs(command.pan),abs(command.tilt))<=engine.config.pid.max_rate_rad_s


def test_hybrid_search_has_deterministic_phases():
    engine=SimulationEngine(); strategy=HybridSearch(engine.config)
    hold=strategy.search(search_request(engine.config,.01))
    later=strategy.search(search_request(engine.config,1.4))
    assert hold.phase=="HYBRID_HOLD"
    assert later.phase.startswith("HYBRID_") and later.phase!="HYBRID_HOLD"


def test_dropout_activates_search_and_records_mode():
    data=compact_config()
    data["disturbance"]["beacon"]["hard_dropout_windows"]=[{"start_s":.1,"end_s":.5,"factor":0}]
    data["reacquisition"]="hybrid"
    engine=SimulationEngine(config=data); frames=[engine.step(annotate=False)[1] for _ in range(6)]
    active=[frame for frame in frames if frame["search_active"]]
    assert active and all(frame["search_mode"].startswith("HYBRID_") for frame in active)
    assert engine.metrics.summary()["search_duration_s"]>0


def test_replay_and_paired_benchmark_are_comparable():
    config=compact_config()
    replay=run_search_replay(config,("raster","spiral","hybrid"),.3)
    assert replay["truth_available_to_strategy"] is False
    assert all(len(value["trace"])==replay["frames"] for value in replay["strategies"].values())
    result=run_acquisition_benchmark(config,("hold","hybrid"),("short_dropout",),(41,),.6)
    assert result["same_actuator_model"] and len(result["rows"])==2
    assert result["ranking_policy"].startswith("No universal winner")
