import numpy as np
import pytest

from src.core.engine import SimulationEngine
from src.core.scenarios import resolve_config
from src.physics.cw import (AnalyticalClohessyWiltshire, ClohessyWiltshire,
                            bounded_motion_residual, cw_case_initial_state)


def test_closed_form_matches_small_step_rk4():
    n=.00113; position=np.array([10.,2.,3.]); velocity=np.array([.01,-.0226,.004])
    rk=ClohessyWiltshire(n,position.copy(),velocity.copy())
    exact=AnalyticalClohessyWiltshire(n,position.copy(),velocity.copy())
    for _ in range(200): rk.step(.1)
    exact_position,exact_velocity=exact.step(20)
    assert rk.position_m == pytest.approx(exact_position,abs=1e-9)
    assert rk.velocity_m_s == pytest.approx(exact_velocity,abs=1e-10)


def test_named_cw_cases_and_bounded_indicator():
    n=.00113
    bounded_position,bounded_velocity=cw_case_initial_state("bounded_ellipse",n)
    drift_position,drift_velocity=cw_case_initial_state("along_track_drift",n)
    assert bounded_motion_residual(bounded_position,bounded_velocity,n)==pytest.approx(0,abs=1e-14)
    assert abs(bounded_motion_residual(drift_position,drift_velocity,n))>1e-4


def test_analytical_preset_is_satellite_only_and_telemetry_is_explicit():
    config=resolve_config("SAT_CW_BOUNDED_ELLIPSE")
    _,telemetry=SimulationEngine(config=config).step(annotate=False)
    assert telemetry["motion_model"]=="cw_analytical"
    assert telemetry["cw_case"]=="bounded_ellipse"
    assert telemetry["cw_bounded_residual_m_s"]==pytest.approx(0,abs=1e-12)
    data=config.model_dump(mode="json"); data["scenario"]="uav_uav"
    with pytest.raises(ValueError,match="only for satellite_satellite"):
        SimulationEngine(config=data)
