from dataclasses import fields
import numpy as np

from src.control.controllers import solve_discrete_riccati
from src.control.pid import PIDAxis
from src.core.contracts import ControllerInput
from src.core.engine import SimulationEngine
from src.core.registry import REGISTRY, Stage
from src.experiments.controller_lab import run_controller_benchmark, run_controller_replay, static_step_response


def compact(data):
    data["camera"].update(width_px=120,height_px=90,focal_length_px=100)
    data.update(fps=10,duration_s=.3,pipeline_preset="CUSTOM")
    return data


def test_controller_contract_excludes_simulator_truth():
    names={field.name for field in fields(ControllerInput)}
    assert not names & {"truth","ground_truth","true_state","true_pixel"}
    assert {"current_state_estimate","predicted_state","actuator_limits","current_gimbal_state"} <= names


def test_pid_anti_windup_and_derivative_filter_are_bounded():
    axis=PIDAxis(10,10,1,.2,.1,derivative_filter_tau_s=.1)
    for _ in range(100): assert abs(axis.step(5,.01)) <= .2
    assert abs(axis.integral) <= .1 and axis.saturated
    assert np.isfinite(axis.filtered_derivative)


def test_riccati_solution_is_finite_and_stabilizing():
    a=np.array([[1,.1],[0,.9]]); b=np.array([[0],[.1]])
    _,gain=solve_discrete_riccati(a,b,np.eye(2),np.eye(1))
    assert np.all(np.isfinite(gain))
    assert max(abs(np.linalg.eigvals(a-b@gain))) < 1


def test_all_controllers_run_through_integrated_gimbal():
    base=compact(SimulationEngine().config_snapshot())
    for name in REGISTRY[Stage.CONTROLLER]:
        data={**base,"controller":name}
        engine=SimulationEngine(config=data)
        _,telemetry=engine.step(annotate=False)
        assert np.isfinite(telemetry["controller_output"]).all()
        assert telemetry["controller_name"] == name
        assert "controller_diagnostics" in telemetry


def test_replay_step_and_compact_benchmark_cover_comparisons():
    config=compact(SimulationEngine().config_snapshot())
    replay=run_controller_replay(config,("pid","lqr","mpc"),.3)
    assert replay["truth_available_to_controller"] is False
    assert all(len(value["trace"])==replay["frames"] for value in replay["controllers"].values())
    step=static_step_response(config,("pid","lqr"),duration_s=.3)
    assert set(step["controllers"])=={"pid","lqr"}
    benchmark=run_controller_benchmark(config,("pid","lqr"),("nominal",),(41,),.2)
    assert benchmark["same_actuator_model"] and len(benchmark["rows"])==2
    assert set(benchmark["pareto_front"]) <= {"pid","lqr"}
