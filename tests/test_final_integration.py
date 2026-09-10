import time

import numpy as np

from src.core.engine import SimulationEngine
from src.core.scenarios import catalogue, resolve_config
from src.experiments.evidence import export_evidence
from src.experiments.monte_carlo import run_monte_carlo
from src.experiments.storage import ExperimentStore


DEMOS=("DEMO_1_NOMINAL_SAT_SAT","DEMO_2_HIGH_VIBRATION","DEMO_3_WEAK_BEACON_GROUND_SAT",
       "DEMO_4_DROPOUT_REACQUISITION","DEMO_5_AGGRESSIVE_UAV","DEMO_6_COMBINED_STRESS")


def compact(config):
    data=config.model_dump(mode="json"); data["camera"].update(width_px=96,height_px=72,focal_length_px=75)
    data.update(fps=10,duration_s=.3,time_scale=min(2,data["time_scale"])); return data


def test_final_defaults_and_pipelines_are_runnable():
    default=resolve_config()
    assert default.scenario_preset=="SAT_SAT_NOMINAL" and default.seed==42 and default.pipeline_preset=="DEFAULT_STABLE"
    baseline=resolve_config("SAT_SAT_NOMINAL","REFERENCE_BASELINE")
    validated=resolve_config("SAT_SAT_NOMINAL","CURRENT_BEST_VALIDATED")
    assert baseline.vision.algorithm=="centroid" and baseline.estimator=="kalman"
    assert validated.vision.algorithm=="centroid" and validated.estimator=="kalman"
    assert "candidate not promoted" in catalogue()["pipelines"]["CURRENT_BEST_VALIDATED"]["evidence"]


def test_all_saved_demo_presets_start_without_optional_module_failure():
    for name in DEMOS:
        config=compact(resolve_config(name,"DEFAULT_STABLE"))
        engine=SimulationEngine(config=config)
        _,telemetry=engine.step(annotate=False)
        assert telemetry["scenario_preset"]==name and telemetry["frame"]==0
        assert telemetry["lock_state"]=="LOCKED"
        assert telemetry["pat_events"][0]["reason"]=="demo starts with coarse alignment established"
        assert not telemetry["correction_fallback"] and not telemetry["predictor_fallback"]


def test_non_demo_acquisition_preset_still_starts_in_search():
    config=compact(resolve_config("GROUND_SAT_CLEAR","DEFAULT_STABLE"))
    assert config["lock"]["start_locked"] is False
    engine=SimulationEngine(config=config)
    assert engine.state_machine.state.value=="SEARCH"


def test_raw_sensor_isolated_from_annotations_and_evaluation_overlay():
    engine=SimulationEngine(config=compact(resolve_config("DEMO_1_NOMINAL_SAT_SAT")))
    annotated,telemetry=engine.step(annotate=True)
    raw,_,_=engine.raw_snapshot()
    assert not np.array_equal(raw,annotated)
    assert engine.raw_jpeg_snapshot().startswith(b"\xff\xd8") and engine.jpeg_snapshot().startswith(b"\xff\xd8")
    assert telemetry["ground_truth_pixel"] is not None


def test_pat_event_history_is_bounded_and_transition_timed():
    engine=SimulationEngine(config=compact(resolve_config("DEMO_1_NOMINAL_SAT_SAT")))
    telemetry=None
    for _ in range(8):
        _,telemetry=engine.step(annotate=False)
    assert telemetry["pat_events"] and telemetry["time_in_pat_state_s"]>=0
    assert len(telemetry["pat_events"])<=100 and {"timestamp_s","event","reason"}<=set(telemetry["pat_events"][0])


def test_live_parameter_change_reaches_physics_and_is_recorded():
    engine=SimulationEngine(config=compact(resolve_config("DEMO_1_NOMINAL_SAT_SAT")))
    event=engine.apply_config_change("disturbance.vibration.stochastic_sigma_rad",40e-6)
    _,telemetry=engine.step(annotate=False)
    assert engine.config.disturbance.vibration.stochastic_sigma_rad==40e-6
    assert telemetry["resolved_error_config"]["disturbance.vibration.stochastic_sigma_rad"]==40e-6
    assert telemetry["config_change_events"]==[event]


def test_headless_summary_storage_and_exports(tmp_path):
    store=ExperimentStore(tmp_path/"final.sqlite3")
    result=run_monte_carlo(compact(resolve_config()),2,42,.2,summary_only=True,store=store)
    assert result["aggregate"]["runs"]==2 and all(run["telemetry"] is None for run in result["results"])
    rows=store.query_runs(); csv_text,csv_type=export_evidence({"runs":rows},"csv_runs")
    html,html_type=export_evidence({"title":"Acceptance","runs":2},"html")
    assert "metric_rmse_tracking_error_px" in csv_text and csv_type=="text/csv"
    assert "<!doctype html>" in html and html_type=="text/html"


def test_short_headless_performance_is_measured_not_hard_coded(tmp_path):
    start=time.perf_counter()
    result=run_monte_carlo(compact(resolve_config()),5,42,.2,summary_only=True,store=ExperimentStore(tmp_path/"perf.sqlite3"))
    elapsed=time.perf_counter()-start
    assert result["aggregate"]["runs"]==5 and elapsed>0
    assert all(run["metrics"]["mean_processing_latency_ms"] is not None for run in result["results"])
