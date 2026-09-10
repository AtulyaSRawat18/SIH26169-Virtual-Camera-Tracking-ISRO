"""Prompt 8 acquisition/reacquisition replay and closed-loop evidence."""
from __future__ import annotations

import copy
import math
import numpy as np

from src.control.gimbal import GimbalPlant
from src.core.contracts import ActuatorLimits, PredictedState, SearchInput, TrackingState
from src.core.engine import SimulationEngine
from src.core.metrics import stats
from src.core.registry import REGISTRY, Stage, validate_config

DEFAULT_STRATEGIES=("hold","last_known","raster","spiral","predicted_point",
                    "covariance_search","predictive_covariance","hybrid")
DEFAULT_CASES=("initial_offset","short_dropout","long_dropout","uncertain_prediction","distractors")


def run_search_replay(config,strategies=DEFAULT_STRATEGIES,duration_s=2.0):
    config=validate_config(config); dt=1/config.fps; frames=math.ceil(duration_s*config.fps)
    centre=(config.camera.width_px/2,config.camera.height_px/2)
    last=TrackingState(centre[0]+30,centre[1]-12,12,-4,0,True,np.diag([9,4,2,2]))
    predicted=PredictedState(True,.05,last.x+4,last.y-2,last.vx,last.vy,
                             covariance=np.diag([36,9,4,4]),predictor_name="recorded")
    output={}
    for name in strategies:
        if name not in REGISTRY[Stage.REACQUISITION]: raise ValueError(f"Unavailable search strategy '{name}'")
        strategy=REGISTRY[Stage.REACQUISITION][name](config,None)
        plant=GimbalPlant(config.pid,config.disturbance.gimbal,config.fps); trace=[]; effort=0.
        for index in range(frames):
            t=(index+1)*dt
            request=SearchInput(last,predicted,predicted.covariance,plant.reported_state(),
                                ActuatorLimits(config.pid.max_rate_rad_s,
                                               config.disturbance.gimbal.max_acceleration_rad_s2,
                                               *config.disturbance.gimbal.position_limit_rad),
                                centre,config.camera.focal_length_px,t,dt,t,"REACQUIRE")
            command=strategy.search(request)
            if command is None: pan=tilt=0.; mode="NONE"; diagnostics={}
            else: pan,tilt,mode,diagnostics=command.pan,command.tilt,command.phase,command.diagnostics
            plant.step(type("Command",(),{"pan":pan,"tilt":tilt})(),dt)
            effort+=(pan*pan+tilt*tilt)*dt
            trace.append({"timestamp":t,"command_rad_s":[pan,tilt],"gimbal_rad":[plant.pan_rad,plant.tilt_rad],
                          "search_mode":mode,"diagnostics":diagnostics})
        output[name]={"trace":trace,"search_effort_integral":effort,
                      "metadata":strategy.metadata() if hasattr(strategy,"metadata") else {"name":name}}
    return {"mode":"SAME_INPUT_SEARCH_REPLAY","truth_available_to_strategy":False,
            "frames":frames,"strategies":output}


def _case(config,name):
    data=config.model_dump(mode="json"); data["pipeline_preset"]="CUSTOM"
    beacon=data["disturbance"]["beacon"]
    if name=="initial_offset": pass
    elif name=="short_dropout": beacon["hard_dropout_windows"]=[{"start_s":.35,"end_s":.65,"factor":0}]
    elif name=="long_dropout": beacon["hard_dropout_windows"]=[{"start_s":.35,"end_s":1.8,"factor":0}]
    elif name=="uncertain_prediction":
        beacon["hard_dropout_windows"]=[{"start_s":.35,"end_s":1.6,"factor":0}]
        data["estimation"]["process_noise_px_s2"]=max(30,data["estimation"]["process_noise_px_s2"])
    elif name=="distractors":
        beacon["hard_dropout_windows"]=[{"start_s":.35,"end_s":1.4,"factor":0}]
        data["disturbance"]["distractors"].update(enabled=True,count=5,intensity=230)
    else: raise ValueError(f"Unknown acquisition case '{name}'")
    return data


def run_acquisition_benchmark(config,strategies=DEFAULT_STRATEGIES,cases=DEFAULT_CASES,
                              seeds=(41,42),duration_s=3.0,store=None):
    config=validate_config(config); rows=[]
    for case in cases:
        base=_case(config,case)
        for seed in seeds:
            for name in strategies:
                data=copy.deepcopy(base); data.update(reacquisition=name,seed=int(seed),run_index=0)
                engine=SimulationEngine(config=data,retain_frames=0)
                if case=="initial_offset":
                    engine.gimbal.actuator.pan_rad=-.45
                for _ in range(math.ceil(duration_s*engine.config.fps)): engine.step(publish=False,annotate=False)
                m=engine.metrics.summary()
                rows.append({"strategy":name,"case":case,"seed":seed,
                             "time_to_first_lock_s":m["time_to_first_lock_s"],
                             "mean_reacquisition_time_s":m["mean_reacquisition_time_s"],
                             "reacquisition_success_percent":m["reacquisition_success_percent"],
                             "locked_percentage":m["locked_percentage"],
                             "false_lock_probability":m["false_lock_probability"],
                             "search_effort_integral":m["search_effort_integral"],
                             "search_duration_s":m["search_duration_s"],
                             "search_fallback_count":m["search_fallback_count"],
                             "dominant_failure_cause":m["dominant_failure_cause"]})
    summary={}
    for name in strategies:
        selected=[r for r in rows if r["strategy"]==name]
        summary[name]={key:stats([r[key] for r in selected if r.get(key) is not None])
                       for key in ("time_to_first_lock_s","mean_reacquisition_time_s","locked_percentage",
                                   "false_lock_probability","search_effort_integral","search_duration_s")}
        summary[name]["failure_count"]=sum(bool(r["dominant_failure_cause"]) for r in selected)
    result={"mode":"PAIRED_CLOSED_LOOP_SEARCH_MATRIX","strategies":list(strategies),
            "cases":list(cases),"seeds":list(seeds),"duration_s":duration_s,"rows":rows,
            "summary":summary,"truth_available_to_strategy":False,"same_actuator_model":True,
            "ranking_policy":"No universal winner; inspect success, time, false lock, and effort by regime."}
    if store is not None: result["result_id"]=store.append_lab_result("ACQUISITION_BENCHMARK",config,result)
    return result


def acquisition_parameter_sweep(config,parameter,values,strategy="hybrid",seeds=(41,),duration_s=1.5):
    allowed={"acquisition.scan_rate_rad_s","acquisition.hold_duration_s","acquisition.last_known_duration_s",
             "acquisition.covariance_sigma_scale","acquisition.innovation_gate_nis"}
    if parameter not in allowed: raise ValueError("Unsupported acquisition sweep parameter")
    output=[]
    for value in values:
        data=validate_config(config).model_dump(mode="json"); target=data
        for part in parameter.split(".")[:-1]: target=target[part]
        target[parameter.split(".")[-1]]=value
        result=run_acquisition_benchmark(data,(strategy,),DEFAULT_CASES,seeds,duration_s)
        output.append({"value":value,"summary":result["summary"][strategy]})
    return {"parameter":parameter,"strategy":strategy,"points":output}
