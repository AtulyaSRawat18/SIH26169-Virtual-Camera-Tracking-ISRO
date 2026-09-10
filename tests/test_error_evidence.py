import math

import numpy as np
import pytest

from src.core.engine import SimulationEngine
from src.core.metrics import Metrics
from src.core.scenarios import resolve_config
from src.experiments.evidence import (ERROR_REGISTRY, apply_level_preset, before_after,
                                      bootstrap_ci, cumulative_evidence, failure_distribution,
                                      metric_change, parameter_sweep, pareto_front,
                                      relevant_variables, validate_distributions,
                                      waterfall_from_record)
from src.experiments.monte_carlo import run_monte_carlo, sample_parameters
from src.experiments.storage import ExperimentStore


def compact(config=None):
    data=(config or resolve_config()).model_dump(mode="json")
    data["camera"].update(width_px=80,height_px=60,focal_length_px=65)
    data.update(fps=10,duration_s=.3,time_scale=2,pipeline_preset="CUSTOM")
    return data


def test_live_errors_use_truth_only_after_pipeline_and_have_valid_units():
    engine=SimulationEngine(config=compact())
    _,record=engine.step(annotate=False)
    truth=np.asarray(record["ground_truth_pixel"])
    measured=np.asarray(record["opencv_pixel"])
    assert record["error_budget"]["classical_measurement_error_px"] == pytest.approx(np.linalg.norm(measured-truth))
    assert record["error_budget"]["final_pointing_error_rad"] == record["pointing_error_rad"]
    assert all(item["unit"] in {"px","rad","rad/s"} for item in ERROR_REGISTRY.values())
    assert all({"error","stage","value","unit","status"} <= set(row) for row in record["error_waterfall"])


def test_waterfall_preserves_negative_cnn_improvement():
    rows=waterfall_from_record({"error_budget":{"classical_measurement_error_px":1.0,"measurement_error_px":2.0,
                                                 "estimator_error_px":1.5,"final_pointing_error_rad":2e-6}})
    cnn=next(row for row in rows if row["error"]=="e_cnn")
    assert cnn["change_from_previous"]==pytest.approx(1.0) and cnn["status"]=="DEGRADED"


def test_running_rmse_p95_and_stage_summaries_are_mathematical():
    metrics=Metrics()
    for value in (1.,2.,4.,8.):
        metrics.update({"true_tracking_error_px":value,"classical_detection_error_px":value,
                        "detection_error_px":value/2,"estimator_error_px":value/3,
                        "measurement_valid":True,"controller_output":[0,0]})
    summary=metrics.summary()
    assert summary["rmse_tracking_error_px"]==pytest.approx(math.sqrt(85/4))
    assert summary["p95_tracking_error_px"]==pytest.approx(np.percentile([1,2,4,8],95))
    assert summary["classical_localization_rmse_px"]==pytest.approx(math.sqrt(85/4))
    assert summary["error_stage_statistics"]["measurement"]["rmse"]==pytest.approx(math.sqrt(85/4))


def test_prediction_is_evaluated_only_when_horizon_timestamp_arrives():
    data=compact(); data.update(predictor="cv",prediction_horizon_s=.2)
    engine=SimulationEngine(config=data)
    records=[engine.step(annotate=False)[1] for _ in range(5)]
    assert records[0]["prediction_error_px"] is None
    assert any(record["prediction_error_px"] is not None for record in records[2:])


def test_actuator_and_pointing_error_definitions_are_consistent():
    data=compact(); data["optical_link"]["enabled"]=True
    engine=SimulationEngine(config=data); _,record=engine.step(annotate=False)
    expected=np.linalg.norm(np.asarray(record["controller_output"])-np.asarray([record["camera_pan_rate_rad_s"],record["camera_tilt_rate_rad_s"]]))
    assert record["actuator_rate_error_rad_s"]==pytest.approx(expected)
    expected_point=math.hypot(record["pointing_error_x_rad"],record["pointing_error_y_rad"])
    assert record["pointing_error_rad"]==pytest.approx(expected_point)
    if record["estimated_pixel"]:
        centre=np.asarray(record["camera_centre_pixel"]); estimate=np.asarray(record["estimated_pixel"])
        assert record["control_error_rad"]==pytest.approx(np.linalg.norm(estimate-centre)/engine.config.camera.focal_length_px)


def test_user_variables_presets_and_dynamic_change_are_resolved_and_timestamped():
    base=resolve_config("GROUND_SAT_CLEAR")
    high=apply_level_preset(base,"HIGH")
    assert high.disturbance.vibration.stochastic_sigma_rad==pytest.approx(22.5e-6)
    assert high.disturbance.atmosphere.cloud_attenuation<1
    assert "disturbance.atmosphere.turbulence_strength" in relevant_variables(high.scenario.value)
    engine=SimulationEngine(config=compact(base)); engine.step()
    event=engine.apply_config_change("camera.noise_sigma",17)
    _,telemetry=engine.step()
    assert event["before"]!=17 and event["after"]==17 and telemetry["config_change_events"][0]==event


def test_fixed_uniform_normal_sampling_is_reproducible_and_metadata_retained(tmp_path):
    config=resolve_config()
    specs={"camera.noise_sigma":{"distribution":"normal","mean":8,"std":2},
           "disturbance.vibration.stochastic_sigma_rad":{"distribution":"fixed","value":8e-6}}
    validate_distributions(config,specs)
    first=sample_parameters(config,specs,42,3); second=sample_parameters(config,specs,42,3)
    assert first[1]==second[1] and first[1]["disturbance.vibration.stochastic_sigma_rad"]==8e-6
    store=ExperimentStore(tmp_path/"sampling.sqlite3")
    result=run_monte_carlo(compact(),1,42,.1,specs,True,store)
    assert result["results"][0]["sampling_metadata"]["camera.noise_sigma"]["subsystem_seed"]
    assert store.query_runs()[0]["sampling_metadata"]["camera.noise_sigma"]["configured_distribution"]==specs["camera.noise_sigma"]


def test_paired_statistics_changes_and_pipelines_are_honest(tmp_path):
    a=compact(); b=compact(); b["estimator"]="none"
    result=before_after(a,b,2,42,.2,ExperimentStore(tmp_path/"paired.sqlite3"))
    assert result["pairs"]==2 and len(result["pair_records"])==2
    assert result["pipeline_A"]["estimator"]=="kalman" and result["pipeline_B"]["estimator"]=="none"
    assert result["paired_statistics"]["rmse_tracking_error_px"]["improvement_ci"]["method"]=="seeded_percentile_bootstrap"
    assert metric_change(10,8)["improvement_percent"]==20
    assert metric_change(80,90,True)["status"]=="IMPROVED"
    assert bootstrap_ci([1,2,3],seed=9)==bootstrap_ci([1,2,3],seed=9)


def test_sweep_multiple_seeds_failure_analytics_and_cumulative_compatibility(tmp_path):
    store=ExperimentStore(tmp_path/"evidence.sqlite3")
    sweep=parameter_sweep(compact(),"camera.noise_sigma",[0,10],2,42,.1,store)
    assert len(sweep["points"])==2 and all(point["runs"]==2 for point in sweep["points"])
    history=cumulative_evidence(store)
    assert history["runs"]==4 and history["compatible_only"]
    failures=failure_distribution([{"failure_cause":"DROPOUT","metrics":{"link_availability_percent":0}},
                                   {"failure_cause":None,"metrics":{}}])
    assert any(row["cause"]=="DROPOUT" and row["count"]==1 for row in failures)
    assert any(row["cause"]=="LINK_UNAVAILABLE" for row in failures)


def test_pareto_does_not_collapse_metrics_to_one_score():
    records=[{"rmse_tracking_error_px":1,"mean_processing_latency_ms":10,"control_effort_integral":2,"mean_reacquisition_time_s":2,"locked_percentage":90,"link_availability_percent":80},
             {"rmse_tracking_error_px":2,"mean_processing_latency_ms":5,"control_effort_integral":1,"mean_reacquisition_time_s":1,"locked_percentage":95,"link_availability_percent":90},
             {"rmse_tracking_error_px":4,"mean_processing_latency_ms":20,"control_effort_integral":3,"mean_reacquisition_time_s":3,"locked_percentage":80,"link_availability_percent":70}]
    result=pareto_front(records)
    assert result["efficient_indices"]==[0,1] and len(result["metric_directions"])==6
