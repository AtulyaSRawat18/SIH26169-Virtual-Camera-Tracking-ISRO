"""Live API for simulation telemetry and camera frames."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.core.engine import ROOT, SimulationEngine
from src.core.registry import capabilities, validate_config
from src.core.scenarios import public_catalogue, resolve_config
from src.experiments.storage import ExperimentStore
from src.experiments.monte_carlo import run_monte_carlo, compare_paired, aggregate
from src.experiments.vision_lab import (DEFAULT_ALGORITHMS as CLASSICAL_ALGORITHMS, compare_frame,
                                        export_dataset_sample, parameter_sweep, run_vision_benchmark)
from src.experiments.estimator_lab import (DEFAULT_ESTIMATORS, compare_estimators_open_loop, estimator_parameter_sweep,
                                           record_measurement_sequence, run_estimator_benchmark)
from src.experiments.cnn_lab import build_and_train_cnn, compare_cnn_same_frame
from src.ml.dataset import generate_spot_dataset
from src.ml.training import CNNTrainingConfig, train_spot_model
from src.prediction.dataset import generate_temporal_dataset
from src.prediction.training import TemporalTrainingConfig, train_temporal_model
from src.experiments.predictor_lab import (DEFAULT_PREDICTORS, closed_loop_predictor_benchmark,
                                           predictor_history_sweep, predictor_horizon_sweep,
                                           run_predictor_replay)
from src.experiments.controller_lab import (DEFAULT_CONTROLLERS, controller_parameter_sweep,
                                             run_controller_benchmark, run_controller_replay,
                                             static_step_response)
from src.experiments.acquisition_lab import (DEFAULT_STRATEGIES, acquisition_parameter_sweep,
                                              run_acquisition_benchmark, run_search_replay)
from src.experiments.optical_lab import (beam_divergence_sweep, compare_link_pipelines,
                                          elevation_sweep, pointing_error_sweep, range_sweep)
from src.experiments.evidence import (ERROR_REGISTRY, VARIABLE_REGISTRY, apply_level_preset,
                                      before_after, cumulative_evidence, export_evidence,
                                      failure_distribution, parameter_sweep, relevant_variables,
                                      validate_distributions, waterfall_from_record)


engine = SimulationEngine(config=resolve_config("DEMO_1_NOMINAL_SAT_SAT"))
store = ExperimentStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="SIH26169 Virtual Camera Tracking", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    telemetry = engine.telemetry_snapshot()
    return {"status": "ok", "frame": telemetry.get("frame", 0)}


@app.get("/api/experiment")
def experiment() -> dict:
    return {"config": engine.config_snapshot(), "algorithms": capabilities(), "catalogue": public_catalogue()}


@app.get("/api/scenarios")
def scenarios() -> dict:
    return public_catalogue()


@app.post("/api/scenarios/resolve")
def resolve_scenario(request: dict) -> dict:
    try:
        config = resolve_config(request.get("scenario_preset", "SAT_SAT_NOMINAL"),
                                request.get("pipeline_preset", "DEFAULT_STABLE"),
                                request.get("overrides"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"config": config.model_dump(mode="json")}


@app.post("/api/experiment/reset")
def reset_experiment(config: dict) -> dict:
    try:
        engine.reset(config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"config": engine.config_snapshot()}


@app.get("/api/experiment/metrics")
def experiment_metrics() -> dict:
    return engine.metrics_snapshot()


@app.get("/api/evidence/registry")
def evidence_registry() -> dict:
    config=engine.config_snapshot()
    return {"errors":ERROR_REGISTRY,"variables":relevant_variables(config["scenario"]),
            "all_variables":VARIABLE_REGISTRY,"simple_levels":["NOMINAL","LOW","MEDIUM","HIGH","STRESS"],
            "sampling_modes":["fixed","uniform","normal"]}


@app.get("/api/evidence/live")
def evidence_live() -> dict:
    telemetry=engine.telemetry_snapshot()
    return {"waterfall":waterfall_from_record(telemetry),"running":engine.metrics_snapshot().get("summary",{}),
            "resolved_config":telemetry.get("resolved_error_config",{}),"scenario":telemetry.get("scenario"),
            "seed":telemetry.get("seed"),"optical_link":telemetry.get("optical_link")}


@app.post("/api/evidence/preset")
def evidence_preset(request: dict) -> dict:
    try:
        config=apply_level_preset(request.get("config",engine.config_snapshot()),request.get("level","NOMINAL"))
        engine.reset(config)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {"config":config.model_dump(mode="json"),"resolved_variables":{
        path:engine._config_value(path) for path in relevant_variables(config.scenario.value)}}


@app.post("/api/evidence/change")
def evidence_change(request: dict) -> dict:
    try:
        path=request["path"]
        if path not in relevant_variables(engine.config.scenario.value):
            raise ValueError(f"Variable '{path}' is not relevant to active scenario")
        event=engine.apply_config_change(path,request["value"])
    except (ValueError,KeyError,TypeError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {"event":event,"config":engine.config_snapshot()}


@app.post("/api/evidence/sweep")
def evidence_sweep(request: dict) -> dict:
    try:
        validate_distributions(request.get("config",engine.config_snapshot()),request.get("distributions"))
        return parameter_sweep(request.get("config",engine.config_snapshot()),request["parameter"],request["values"],
                               int(request.get("runs_per_point",10)),int(request.get("base_seed",42)),
                               request.get("duration_s"),store)
    except (ValueError,KeyError,TypeError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/evidence/before-after")
def evidence_before_after(request: dict) -> dict:
    try:
        return before_after(request["config_a"],request["config_b"],int(request.get("runs",10)),
                            int(request.get("base_seed",42)),request.get("duration_s"),store)
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.get("/api/evidence/cumulative")
def evidence_cumulative(scenario: str | None=None) -> dict:
    return cumulative_evidence(store,scenario)


@app.get("/api/evidence/failures")
def evidence_failures(scenario: str | None=None) -> dict:
    filters={} if scenario is None else {"scenario":scenario}
    runs=store.query_runs(filters)
    worst=sorted(runs,key=lambda r:(r["metrics"].get("rmse_tracking_error_px") is not None,
                                    r["metrics"].get("rmse_tracking_error_px") or -1),reverse=True)[:5]
    return {"distribution":failure_distribution(runs),"worst_runs":[{
        "run_id":r["run_id"],"seed":r["run_seed"],"scenario":r["scenario"],"pipeline":r["algorithm_pipeline"],
        "resolved_config":r["resolved_config"],"metrics":r["metrics"],"failure_cause":r["failure_cause"]} for r in worst]}


@app.post("/api/evidence/export")
def evidence_export(request: dict):
    format=request.get("format","json")
    try:
        payload=request.get("payload")
        if payload is None:
            payload={"runs":store.query_runs({"batch_id":request["batch_id"]} if request.get("batch_id") else {})}
        content,mime=export_evidence(payload,format)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    extension="csv" if format.startswith("csv") else "json"
    return StreamingResponse(iter([content]),media_type=mime,headers={"Content-Disposition":f"attachment; filename=pat-evidence.{extension}"})


@app.get("/api/evidence/report",response_class=HTMLResponse)
def evidence_report(scenario: str | None=None):
    payload={"title":"SIH26169 PAT Evidence Report","generated_from":"compatible stored experiments",
             "live":evidence_live(),"cumulative":cumulative_evidence(store,scenario),
             "limitations":["software simulation testbed","simplified atmospheric and detector models",
                            "coarse-gimbal emphasis","synthetic ML data","no flight-hardware validation"]}
    content,_=export_evidence(payload,"html")
    return HTMLResponse(content)


@app.post("/api/experiments/replay")
def replay_experiment(request: dict) -> dict:
    runs=store.query_runs({"run_id":request.get("run_id")})
    if not runs: raise HTTPException(status_code=404,detail="Compatible stored run not found")
    run=runs[0]; config=run["resolved_config"].copy(); config["seed"]=run["base_seed"]; config["run_index"]=run["pair_index"] or 0
    try: engine.reset(config)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {"replayed_run_id":run["run_id"],"config":engine.config_snapshot(),"provenance":{
        "batch_id":run["batch_id"],"run_seed":run["run_seed"],"algorithm_versions":run["algorithm_versions"]}}


@app.get("/api/evidence/summaries")
def evidence_summaries() -> dict:
    """Return the small, checked-in benchmark summaries used by the dashboard."""
    evidence_root = Path(ROOT) / "evidence"
    files = {
        "vision": "classical_vision_benchmark_summary.json",
        "estimation": "estimator_benchmark_summary.json",
        "cnn": "cnn_spot_correction_summary.json",
        "temporal": "temporal_prediction_summary.json",
        "controller": "controller_benchmark_summary.json",
        "acquisition": "acquisition_benchmark_summary.json",
        "optical_link": "optical_link_summary.json",
        "error_evidence": "error_evidence_summary.json",
    }
    summaries: dict[str, dict] = {}
    for key, filename in files.items():
        path = evidence_root / filename
        if path.exists():
            summaries[key] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


@app.post("/api/experiments/monte-carlo")
def monte_carlo(request: dict) -> dict:
    try:
        result = run_monte_carlo(request.get("config", engine.config_snapshot()), runs=int(request.get("runs", 10)),
                                 base_seed=int(request.get("base_seed", 42)), duration_s=request.get("duration_s"),
                                 distributions=request.get("distributions"), summary_only=bool(request.get("summary_only", True)), store=store)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post("/api/experiments/compare")
def compare(request: dict) -> dict:
    try:
        return compare_paired(request["config_a"], request["config_b"], runs=int(request.get("runs",10)),
                              base_seed=int(request.get("base_seed",42)), duration_s=request.get("duration_s"),
                              summary_only=bool(request.get("summary_only",True)), store=store)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/experiments/history")
def history(scenario: str | None = None, estimator: str | None = None, include_incompatible: bool = False) -> dict:
    filters={k:v for k,v in {"scenario":scenario,"estimator":estimator}.items() if v is not None}
    runs=store.query_runs(filters,include_incompatible)
    return {"runs": runs, "aggregate": aggregate(runs) if runs else {"runs": 0}}


@app.post("/api/control/replay")
def controller_replay(request: dict) -> dict:
    try:
        return run_controller_replay(request.get("config",engine.config_snapshot()),
                                     tuple(request.get("controllers",DEFAULT_CONTROLLERS)),
                                     float(request.get("duration_s",3.0)))
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/control/step-response")
def controller_step_response(request: dict) -> dict:
    try:
        return static_step_response(request.get("config",engine.config_snapshot()),
                                    tuple(request.get("controllers",DEFAULT_CONTROLLERS)),
                                    float(request.get("step_rad",.025)),float(request.get("duration_s",3.0)))
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/control/benchmark")
def controller_benchmark(request: dict) -> dict:
    try:
        return run_controller_benchmark(request.get("config",engine.config_snapshot()),
                                        tuple(request.get("controllers",DEFAULT_CONTROLLERS)),
                                        tuple(request.get("conditions",("nominal","vibration","latency","saturation","dropout"))),
                                        tuple(request.get("seeds",(41,42))),float(request.get("duration_s",2.0)),store)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/control/sweep")
def controller_sweep(request: dict) -> dict:
    try:
        return controller_parameter_sweep(request.get("config",engine.config_snapshot()),request["parameter"],
                                          request["values"],request.get("controller","pid"),
                                          tuple(request.get("seeds",(41,))),float(request.get("duration_s",1.5)))
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/acquisition/replay")
def acquisition_replay(request: dict) -> dict:
    try:
        return run_search_replay(request.get("config",engine.config_snapshot()),
                                 tuple(request.get("strategies",DEFAULT_STRATEGIES)),
                                 float(request.get("duration_s",2.0)))
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/acquisition/benchmark")
def acquisition_benchmark(request: dict) -> dict:
    try:
        return run_acquisition_benchmark(request.get("config",engine.config_snapshot()),
                                         tuple(request.get("strategies",DEFAULT_STRATEGIES)),
                                         tuple(request.get("cases",("initial_offset","short_dropout","long_dropout","uncertain_prediction","distractors"))),
                                         tuple(request.get("seeds",(41,42))),float(request.get("duration_s",3.0)),store)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/acquisition/sweep")
def acquisition_sweep(request: dict) -> dict:
    try:
        return acquisition_parameter_sweep(request.get("config",engine.config_snapshot()),request["parameter"],
                                           request["values"],request.get("strategy","hybrid"),
                                           tuple(request.get("seeds",(41,))),float(request.get("duration_s",1.5)))
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/optical/pointing-sweep")
def optical_pointing_sweep(request: dict) -> dict:
    try: return pointing_error_sweep(request.get("config",engine.config_snapshot()),request.get("values_urad",[0,10,25,50,100]))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/optical/divergence-sweep")
def optical_divergence_sweep(request: dict) -> dict:
    try: return beam_divergence_sweep(request.get("config",engine.config_snapshot()),request.get("values_urad",[10,25,50,100]),request.get("pointing_errors_urad",[0,20,50]))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/optical/range-sweep")
def optical_range_sweep(request: dict) -> dict:
    try: return range_sweep(request.get("config",engine.config_snapshot()),request.get("values_m",[100,1000,10000,100000]),request.get("pointing_errors_urad",[0,20,50]))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/optical/elevation-sweep")
def optical_elevation_sweep(request: dict) -> dict:
    try: return elevation_sweep(request.get("config",engine.config_snapshot()),request.get("elevations_deg",[10,20,30,45,60,90]),float(request.get("pointing_error_urad",20)))
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.post("/api/optical/compare")
def optical_compare(request: dict) -> dict:
    try: return compare_link_pipelines(request.get("config",engine.config_snapshot()),request.get("config_a"),request.get("config_b"),tuple(request.get("seeds",[41,42])),float(request.get("duration_s",2)),store)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.get("/api/experiments/batches")
def batches() -> dict:
    return {"batches": store.list_batches()}


@app.post("/api/vision/compare-frame")
def vision_compare_frame(request: dict) -> dict:
    frame, telemetry, _ = engine.raw_snapshot()
    truth = telemetry.get("ground_truth_pixel")
    try:
        return compare_frame(engine.config, frame, None if truth is None else tuple(truth),
                             request.get("algorithms", CLASSICAL_ALGORITHMS),
                             request.get("benchmark_mode", "FULL_FRAME"), True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/vision/benchmark")
def vision_benchmark(request: dict) -> dict:
    try:
        return run_vision_benchmark(request.get("config", engine.config_snapshot()),
                                    seeds=tuple(request.get("seeds", [41, 42])),
                                    algorithms=tuple(request.get("algorithms", CLASSICAL_ALGORITHMS)),
                                    conditions=request.get("conditions"),
                                    frames_per_seed=int(request.get("frames_per_seed", 8)),
                                    benchmark_mode=request.get("benchmark_mode", "FULL_FRAME"), store=store)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/vision/sweep")
def vision_sweep(request: dict) -> dict:
    try:
        return parameter_sweep(request.get("config", engine.config_snapshot()), request["parameter"],
                               request["values"], seeds=tuple(request.get("seeds", [41, 42])),
                               algorithms=tuple(request.get("algorithms", CLASSICAL_ALGORITHMS)),
                               frames_per_seed=int(request.get("frames_per_seed", 5)), store=store)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/vision/export-sample")
def vision_export_sample(request: dict) -> dict:
    frame, telemetry, measurement = engine.raw_snapshot()
    truth = telemetry.get("ground_truth_pixel")
    if measurement is None or truth is None:
        raise HTTPException(status_code=409, detail="No exportable target sample is available")
    return export_dataset_sample(ROOT / "data" / "dataset_exports", frame, measurement, truth, engine.config,
                                 engine.config.seed, request.get("trajectory_id", "live"), engine.config.scenario_preset)


@app.post("/api/estimation/replay")
def estimator_replay(request: dict) -> dict:
    try:
        config = request.get("config", engine.config_snapshot())
        sequence = record_measurement_sequence(config, float(request.get("duration_s", 2.0)))
        return compare_estimators_open_loop(sequence, config, tuple(request.get("estimators", DEFAULT_ESTIMATORS)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/estimation/benchmark")
def estimator_benchmark(request: dict) -> dict:
    try:
        return run_estimator_benchmark(request.get("config", engine.config_snapshot()),
                                       seeds=tuple(request.get("seeds", [41, 42])),
                                       estimators=tuple(request.get("estimators", DEFAULT_ESTIMATORS)),
                                       conditions=tuple(request.get("conditions", [])) or None, store=store)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/estimation/sweep")
def estimator_sweep(request: dict) -> dict:
    try:
        config=request.get("config",engine.config_snapshot())
        sequence=record_measurement_sequence(config,float(request.get("duration_s",2.0)))
        result=estimator_parameter_sweep(sequence,config,request["parameter"],request["values"],
                                         tuple(request.get("estimators",["kf_cv","ekf_angular","ukf_angular"])))
        store.append_lab_result("ESTIMATOR_PARAMETER_SWEEP",validate_config(config),result)
        return result
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.get("/api/labs/history")
def lab_history(kind: str | None = None, include_incompatible: bool = False) -> dict:
    return {"results": store.query_lab_results(kind, include_incompatible)}


@app.post("/api/ai/compare-frame")
def ai_compare_frame(request: dict) -> dict:
    frame, telemetry, _ = engine.raw_snapshot()
    truth = telemetry.get("ground_truth_pixel")
    return compare_cnn_same_frame(engine.config, frame, truth, request.get("model_path"))


@app.post("/api/ai/dataset/generate")
def ai_generate_dataset(request: dict) -> dict:
    try:
        return generate_spot_dataset(trajectories_per_scenario=int(request.get("trajectories_per_scenario", 4)),
                                     frames_per_trajectory=int(request.get("frames_per_trajectory", 24)),
                                     roi_size_px=int(request.get("roi_size_px", 32)), seed=int(request.get("seed", 42)))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/ai/train")
def ai_train(request: dict) -> dict:
    try:
        dataset_directory = request.get("dataset_directory")
        if not dataset_directory:
            dataset_directory = generate_spot_dataset(trajectories_per_scenario=3, frames_per_trajectory=20)["directory"]
        training = CNNTrainingConfig(model_id=request.get("model_id", "cnn-tiny-residual-v4"),
                                     architecture=request.get("architecture", "cnn_tiny"),
                                     use_aux_features=bool(request.get("use_aux_features", True)),
                                     epochs=int(request.get("epochs", 18)), training_seed=int(request.get("seed", 42)))
        result = train_spot_model(dataset_directory, training)
        store.append_lab_result("CNN_TRAINING_BENCHMARK", engine.config, result)
        return result
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prediction/dataset/generate")
def prediction_generate_dataset(request: dict) -> dict:
    try:
        return generate_temporal_dataset(trajectories_per_scenario=int(request.get("trajectories_per_scenario", 3)),
                                         frames_per_trajectory=int(request.get("frames_per_trajectory", 70)),
                                         history_frames=int(request.get("history_frames", 12)),
                                         horizons_s=tuple(request.get("horizons_s", [0.02, 0.05, 0.1, 0.2])),
                                         fps=float(request.get("fps", 30)), seed=int(request.get("seed", 42)),
                                         use_cnn=bool(request.get("use_cnn", False)),
                                         camera_width_px=int(request.get("camera_width_px", 640)),
                                         camera_height_px=int(request.get("camera_height_px", 480)),
                                         focal_length_px=float(request.get("focal_length_px", 520)),
                                         beacon_radius_px=int(request.get("beacon_radius_px", 8)),
                                         camera_noise_sigma=float(request.get("camera_noise_sigma", 5)))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prediction/train")
def prediction_train(request: dict) -> dict:
    try:
        directory = request["dataset_directory"]
        kind = request.get("kind", "gru")
        model_id = request.get("model_id", {"gru":"gru-small-v1", "lstm":"lstm-small-v1",
                                             "gru_residual_cv":"gru-residual-cv-v1"}.get(kind, kind))
        training = TemporalTrainingConfig(model_id=model_id, kind=kind,
                                          hidden_size=int(request.get("hidden_size", 24)),
                                          epochs=int(request.get("epochs", 20)),
                                          training_seed=int(request.get("seed", 42)))
        result = train_temporal_model(directory, training)
        store.append_lab_result("TEMPORAL_MODEL_BENCHMARK", engine.config, result)
        return result
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prediction/replay")
def prediction_replay(request: dict) -> dict:
    try:
        result = run_predictor_replay(request.get("config", engine.config_snapshot()),
                                      float(request.get("duration_s", 3)),
                                      tuple(request.get("predictors", DEFAULT_PREDICTORS)),
                                      float(request.get("horizon_s", 0.05)), request.get("history_frames"))
        store.append_lab_result("PREDICTOR_OPEN_LOOP_REPLAY", engine.config, result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prediction/horizon-sweep")
def prediction_horizon(request: dict) -> dict:
    result = predictor_horizon_sweep(request.get("config", engine.config_snapshot()),
                                     tuple(request.get("horizons", [0.02, 0.05, 0.1, 0.2])),
                                     tuple(request.get("predictors", DEFAULT_PREDICTORS)),
                                     float(request.get("duration_s", 3)))
    store.append_lab_result("PREDICTOR_HORIZON_SWEEP", engine.config, result)
    return result


@app.post("/api/prediction/history-sweep")
def prediction_history(request: dict) -> dict:
    result = predictor_history_sweep(request.get("config", engine.config_snapshot()),
                                     tuple(request.get("histories", [5, 10, 12, 20, 30])),
                                     tuple(request.get("predictors", ["gru", "lstm"])),
                                     float(request.get("horizon_s", 0.05)), float(request.get("duration_s", 3)))
    store.append_lab_result("PREDICTOR_HISTORY_SWEEP", engine.config, result)
    return result


@app.post("/api/prediction/closed-loop")
def prediction_closed_loop(request: dict) -> dict:
    result = closed_loop_predictor_benchmark(request.get("config", engine.config_snapshot()),
                                             tuple(request.get("predictors", ["none", "cv", "ca", "gru", "lstm"])),
                                             tuple(request.get("seeds", [41, 42])),
                                             float(request.get("duration_s", 3)), float(request.get("horizon_s", 0.05)))
    store.append_lab_result("PREDICTOR_CLOSED_LOOP", engine.config, result)
    return result


def mjpeg_stream(raw: bool=False):
    while True:
        jpeg = engine.raw_jpeg_snapshot() if raw else engine.jpeg_snapshot()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / float(engine.config["fps"]))


@app.get("/api/camera.mjpeg")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera-raw.mjpeg")
def raw_camera_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(raw=True), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            telemetry = engine.telemetry_snapshot()
            if telemetry:
                await websocket.send_json(telemetry)
            await asyncio.sleep(1.0 / 20.0)
    except WebSocketDisconnect:
        return


frontend_dist = Path(ROOT) / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
