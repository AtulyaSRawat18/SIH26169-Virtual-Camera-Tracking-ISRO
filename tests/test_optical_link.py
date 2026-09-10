import numpy as np
import pytest

from src.core.engine import SimulationEngine
from src.core.scenarios import resolve_config
from src.experiments.monte_carlo import run_monte_carlo
from src.experiments.optical_lab import pointing_error_sweep, range_sweep
from src.experiments.storage import ExperimentStore
from src.optics.link import GaussianOpticalLink


def enabled_config():
    data=resolve_config().model_dump(mode="json")
    data["optical_link"].update(enabled=True,range_mode="MANUAL_OVERRIDE",manual_range_m=1000,
                                beam_divergence_urad=100,receiver_sensitivity_dbm=-50)
    return type(resolve_config()).model_validate(data)


def evaluate(config,error=0,atmosphere=1):
    return GaussianOpticalLink(config.optical_link).evaluate(error,0,0,0,1000,atmosphere)


def test_zero_error_maximizes_factor_and_error_is_monotonic_finite():
    c=enabled_config(); factors=[evaluate(c,x).pointing_factor for x in (0,10e-6,50e-6,1e-3)]
    assert factors[0]==pytest.approx(1)
    assert all(np.isfinite(factors)) and all(a>=b for a,b in zip(factors,factors[1:]))


def test_atmosphere_efficiency_range_and_margin_arithmetic():
    c=enabled_config(); clear=evaluate(c,20e-6,1); hazy=evaluate(c,20e-6,.4)
    assert hazy.received_power_w < clear.received_power_w
    data=c.model_dump(mode="json"); data["optical_link"]["receiver_efficiency"]*=.5
    inefficient=evaluate(type(c).model_validate(data),20e-6)
    assert inefficient.received_power_w < clear.received_power_w
    near=range_sweep(c,[100],(0,))["rows"][0]; far=range_sweep(c,[10000],(0,))["rows"][0]
    assert far["received_power_dbm"] < near["received_power_dbm"]
    assert clear.link_margin_db == pytest.approx(clear.received_power_dbm-c.optical_link.receiver_sensitivity_dbm)


def test_deterministic_output_and_snr_domains_are_separate():
    c=enabled_config(); a=evaluate(c,25e-6); b=evaluate(c,25e-6)
    assert a==b and a.communication_snr_db is None and a.communication_snr_status=="NOT_CONFIGURED"
    engine=SimulationEngine(config=c); _,telemetry=engine.step(annotate=False)
    assert "image_snr_estimate" in telemetry and "communication_snr_db" in telemetry
    assert telemetry["optical_link_enabled"] and telemetry["optical_range_m"]==pytest.approx(1000)
    assert telemetry["optical_range_m"] != pytest.approx(np.linalg.norm(telemetry["target_world_m"]))


def test_auto_range_and_user_override_are_resolved():
    data=resolve_config().model_dump(mode="json"); data["optical_link"].update(enabled=True,range_mode="AUTO_FROM_SCENARIO",transmit_power_w=3.25)
    engine=SimulationEngine(config=data); _,telemetry=engine.step(annotate=False)
    assert telemetry["optical_range_m"]==pytest.approx(np.linalg.norm(telemetry["target_world_m"]))
    assert engine.config_snapshot()["optical_link"]["transmit_power_w"]==3.25


def test_sweeps_and_monte_carlo_store_optical_outputs(tmp_path):
    c=enabled_config(); sweep=pointing_error_sweep(c,[0,10,50])
    assert [r["pointing_factor"] for r in sweep["rows"]]==sorted([r["pointing_factor"] for r in sweep["rows"]],reverse=True)
    data=c.model_dump(mode="json"); data["camera"].update(width_px=100,height_px=80,focal_length_px=80); data.update(fps=10,duration_s=.2,pipeline_preset="CUSTOM")
    result=run_monte_carlo(data,runs=1,base_seed=41,duration_s=.2,store=ExperimentStore(tmp_path/"optical.sqlite3"))
    metrics=result["results"][0]["metrics"]
    assert metrics["optical_link_enabled"] and "mean_received_power_dbm" in metrics


def test_default_stable_keeps_link_disabled():
    engine=SimulationEngine(); _,telemetry=engine.step(annotate=False)
    assert not telemetry["optical_link_enabled"] and telemetry["link_state"]=="NOT_CONFIGURED"
