"""Prompt 7 controller replay, closed-loop benchmarking, sweeps, and Pareto evidence."""
from __future__ import annotations

import copy
import math
import numpy as np

from src.control.gimbal import GimbalPlant
from src.core.contracts import (ActuatorLimits, ControllerInput, PredictedState,
                                TrackingState)
from src.core.engine import SimulationEngine
from src.core.metrics import stats
from src.core.registry import REGISTRY, Stage, validate_config

DEFAULT_CONTROLLERS = ("pid", "ff_pid", "gain_scheduled_pid", "lqr", "mpc")
DEFAULT_CONDITIONS = ("nominal", "vibration", "latency", "saturation", "dropout")


def _request(config, state, predicted, gimbal, timestamp, dt):
    return ControllerInput(
        state, predicted, (config.camera.width_px / 2, config.camera.height_px / 2),
        gimbal.reported_state(),
        ActuatorLimits(config.pid.max_rate_rad_s,
                       config.disturbance.gimbal.max_acceleration_rad_s2,
                       *config.disturbance.gimbal.position_limit_rad),
        config.camera.focal_length_px, dt, timestamp, predicted.covariance,
        state.covariance, {"source": "recorded_estimator_replay"})


def run_controller_replay(config, controllers=DEFAULT_CONTROLLERS, duration_s=3.0):
    """Replay one estimator/predictor sequence through every controller and identical plants."""
    config = validate_config(config); dt = 1 / config.fps
    frames = max(2, math.ceil(duration_s * config.fps)); cx = config.camera.width_px / 2
    cy = config.camera.height_px / 2; focal = config.camera.focal_length_px
    results = {}
    for name in controllers:
        if name not in REGISTRY[Stage.CONTROLLER]:
            raise ValueError(f"Unavailable controller '{name}'")
        controller = REGISTRY[Stage.CONTROLLER][name](config, None)
        plant = GimbalPlant(config.pid, config.disturbance.gimbal, config.fps)
        trace=[]; previous=np.zeros(2); effort=smoothness=0.0; saturated=0; fallback=0
        for index in range(frames):
            t=(index+1)*dt
            # Recorded-estimator surrogate: two LOS frequencies plus a small step.
            ex=.035*math.sin(1.7*t)+(.018 if t>duration_s*.45 else 0)
            ey=.022*math.cos(1.1*t)
            vx=.035*1.7*math.cos(1.7*t); vy=-.022*1.1*math.sin(1.1*t)
            state=TrackingState(cx+focal*math.tan(ex), cy-focal*math.tan(ey),
                                focal*vx, -focal*vy, t, True, np.eye(4)*1.5)
            predicted=PredictedState(True, config.prediction_horizon_s,
                                     state.x+state.vx*config.prediction_horizon_s,
                                     state.y+state.vy*config.prediction_horizon_s,
                                     state.vx,state.vy,covariance=np.eye(4)*2,
                                     predictor_name="replay_cv")
            command=controller.compute(_request(config,state,predicted,plant,t,dt))
            applied=plant.step(command,dt); u=np.array([command.pan,command.tilt])
            effort+=float(u@u)*dt; smoothness+=float((u-previous)@(u-previous)); previous=u
            saturated+=int(command.saturated or plant.rate_saturated or plant.acceleration_saturated)
            fallback+=int(command.fallback_used)
            trace.append({"timestamp":t,"input_error_rad":[ex,ey],"command_rad_s":u.tolist(),
                          "applied_rad_s":[applied.pan,applied.tilt],
                          "gimbal_rad":[plant.pan_rad,plant.tilt_rad],
                          "diagnostics":command.diagnostics})
        results[name]={"trace":trace,"control_effort_integral":effort,
                       "control_smoothness_integral":smoothness,
                       "saturation_percent":100*saturated/frames,"fallback_count":fallback,
                       "metadata":controller.metadata() if hasattr(controller,"metadata") else {"name":name}}
    return {"mode":"SAME_INPUT_OPEN_LOOP_REPLAY","truth_available_to_controller":False,
            "frames":frames,"duration_s":duration_s,"controllers":results}


def static_step_response(config, controllers=DEFAULT_CONTROLLERS, step_rad=.025, duration_s=3.0):
    """Deterministic angular step against identical nonlinear camera/plant interfaces."""
    config=validate_config(config); dt=1/config.fps; frames=math.ceil(duration_s*config.fps)
    cx,cy=config.camera.width_px/2,config.camera.height_px/2; focal=config.camera.focal_length_px
    output={}
    for name in controllers:
        controller=REGISTRY[Stage.CONTROLLER][name](config,None)
        plant=GimbalPlant(config.pid,config.disturbance.gimbal,config.fps)
        errors=[]; commands=[]; settled=None; peak=0.0
        for index in range(frames):
            t=(index+1)*dt; error=step_rad-plant.pan_rad; peak=max(peak,plant.pan_rad)
            x=cx+focal*math.tan(error)
            state=TrackingState(x,cy,0,0,t,True,np.eye(4))
            predicted=PredictedState(True,0,x,cy,0,0,covariance=np.eye(4),predictor_name="step")
            command=controller.compute(_request(config,state,predicted,plant,t,dt))
            plant.step(command,dt); errors.append(error); commands.append(command.pan)
            if settled is None and index>2 and abs(error)<max(abs(step_rad)*.02,1e-4): settled=t
        steady=float(np.mean(np.abs(errors[-max(2,frames//10):])))
        output[name]={"rise_time_s":next((i*dt for i,e in enumerate(errors) if abs(e)<=.1*abs(step_rad)),None),
                      "settling_time_s":settled,"overshoot_percent":max(0,(peak-step_rad)/abs(step_rad)*100),
                      "steady_state_error_rad":steady,"peak_command_rad_s":max(abs(x) for x in commands),
                      "error_rad":errors,"command_rad_s":commands}
    return {"mode":"STATIC_ANGULAR_STEP","step_rad":step_rad,"controllers":output}


def _condition(config, name):
    data=config.model_dump(mode="json"); data["pipeline_preset"]="CUSTOM"
    if name=="vibration": data["disturbance"]["vibration"].update(enabled=True,level="MODERATE",stochastic_sigma_rad=.0002)
    elif name=="latency": data["disturbance"]["gimbal"]["command_latency_s"]=.08
    elif name=="saturation":
        data["pid"]["max_rate_rad_s"]=min(data["pid"]["max_rate_rad_s"],.035)
        data["disturbance"]["gimbal"]["max_acceleration_rad_s2"]=.08
    elif name=="dropout": data["disturbance"]["sensor"]["frame_dropout_probability"]=.18
    elif name!="nominal": raise ValueError(f"Unknown controller condition '{name}'")
    return data


def _pareto(rows):
    points=[]
    for row in rows:
        error=row["rmse_tracking_error_px"]; effort=row["control_effort_integral"]
        dominated=any(other is not row and other["rmse_tracking_error_px"]<=error and
                      other["control_effort_integral"]<=effort and
                      (other["rmse_tracking_error_px"]<error or other["control_effort_integral"]<effort)
                      for other in rows)
        if not dominated: points.append(row["controller"])
    return points


def run_controller_benchmark(config, controllers=DEFAULT_CONTROLLERS, conditions=DEFAULT_CONDITIONS,
                             seeds=(41,42), duration_s=2.0, store=None):
    config=validate_config(config); rows=[]
    for condition in conditions:
        base=_condition(config,condition)
        for seed in seeds:
            for name in controllers:
                data=copy.deepcopy(base); data.update(controller=name,seed=int(seed),run_index=0)
                engine=SimulationEngine(config=data,retain_frames=0)
                for _ in range(math.ceil(duration_s*engine.config.fps)): engine.step(publish=False,annotate=False)
                metrics=engine.metrics.summary()
                rows.append({"controller":name,"condition":condition,"seed":seed,**metrics})
    summary={}
    for name in controllers:
        selected=[r for r in rows if r["controller"]==name]
        summary[name]={key:stats([r[key] for r in selected if r.get(key) is not None])
                       for key in ("rmse_tracking_error_px","p95_tracking_error_px","acquisition_time_s",
                                   "locked_percentage","control_effort_integral","control_smoothness_integral",
                                   "rate_saturation_percent","mean_controller_latency_ms")}
        summary[name]["failure_count"]=sum(bool(r.get("dominant_failure_cause")) for r in selected)
    aggregate_rows=[{"controller":name,
                     "rmse_tracking_error_px":summary[name]["rmse_tracking_error_px"]["mean"] or math.inf,
                     "control_effort_integral":summary[name]["control_effort_integral"]["mean"] or math.inf}
                    for name in controllers]
    result={"mode":"CLOSED_LOOP_COMMON_RANDOM_NUMBERS","controllers":list(controllers),
            "conditions":list(conditions),"seeds":list(seeds),"duration_s":duration_s,
            "rows":rows,"summary":summary,"pareto_front":_pareto(aggregate_rows),
            "truth_available_to_controller":False,"same_actuator_model":True}
    if store is not None: result["result_id"]=store.append_lab_result("CONTROLLER_BENCHMARK",config,result)
    return result


def controller_parameter_sweep(config, parameter, values, controller="pid", seeds=(41,), duration_s=1.5):
    if parameter not in {"pid.kp","pid.ki","pid.kd","ff_pid.gain","mpc.prediction_horizon_steps",
                         "mpc.control_effort_weight","lqr.control_effort_weight"}:
        raise ValueError("Unsupported controller sweep parameter")
    rows=[]
    for value in values:
        data=validate_config(config).model_dump(mode="json"); target=data
        parts=parameter.split(".")
        for part in parts[:-1]: target=target[part]
        target[parts[-1]]=value; data.update(controller=controller,pipeline_preset="CUSTOM")
        result=run_controller_benchmark(data,(controller,),("nominal",),seeds,duration_s)
        item=result["summary"][controller]
        rows.append({"value":value,"rmse_tracking_error_px":item["rmse_tracking_error_px"]["mean"],
                     "control_effort_integral":item["control_effort_integral"]["mean"],
                     "locked_percentage":item["locked_percentage"]["mean"]})
    return {"parameter":parameter,"controller":controller,"rows":rows}
