"""Prompt 9 optical consequence sweeps and paired pipeline comparisons."""
from __future__ import annotations

import copy
import math
from src.core.engine import SimulationEngine
from src.core.metrics import stats
from src.core.registry import validate_config
from src.optics.link import GaussianOpticalLink


def _enabled(config):
    data=validate_config(config).model_dump(mode="json")
    data["optical_link"]["enabled"]=True; data["pipeline_preset"]="CUSTOM"
    return validate_config(data)


def _isolated(config,pointing_error_rad=0.0,range_m=None,divergence_urad=None,atmosphere=1.0):
    data=_enabled(config).model_dump(mode="json")
    if range_m is not None: data["optical_link"].update(range_mode="MANUAL_OVERRIDE",manual_range_m=float(range_m))
    if divergence_urad is not None: data["optical_link"]["beam_divergence_urad"]=float(divergence_urad)
    model=GaussianOpticalLink(validate_config(data).optical_link)
    selected_range=data["optical_link"]["manual_range_m"] if data["optical_link"]["range_mode"]=="MANUAL_OVERRIDE" else 1000.0
    return model.evaluate(pointing_error_rad,0,0,0,selected_range,atmosphere).as_dict()


def pointing_error_sweep(config,values_urad):
    rows=[]
    for value in values_urad:
        result=_isolated(config,float(value)*1e-6)
        rows.append({"pointing_error_urad":value,"pointing_factor":result["pointing_factor"],
                     "pointing_loss_db":result["pointing_loss_db"],"received_power_dbm":result["received_power_dbm"],
                     "link_margin_db":result["link_margin_db"],"link_state":result["link_state"]})
    return {"sweep":"POINTING_ERROR","rows":rows,"monotonic_expectation":"pointing factor must not increase"}


def beam_divergence_sweep(config,values_urad,pointing_errors_urad=(0,20,50)):
    rows=[]
    for divergence in values_urad:
        for error in pointing_errors_urad:
            result=_isolated(config,error*1e-6,divergence_urad=divergence)
            rows.append({"beam_divergence_urad":divergence,"pointing_error_urad":error,
                         "geometric_capture_factor":result["geometric_capture_factor"],
                         "pointing_factor":result["pointing_factor"],"received_power_dbm":result["received_power_dbm"],
                         "link_margin_db":result["link_margin_db"]})
    return {"sweep":"BEAM_DIVERGENCE","rows":rows,"interpretation":"Wider beams reduce pointing sensitivity but spread power over a larger spot."}


def range_sweep(config,values_m,pointing_errors_urad=(0,20,50)):
    rows=[]
    for range_m in values_m:
        for error in pointing_errors_urad:
            result=_isolated(config,error*1e-6,range_m=range_m)
            rows.append({"range_m":range_m,"pointing_error_urad":error,
                         "beam_radius_m":result["beam_radius_m"],"geometric_capture_factor":result["geometric_capture_factor"],
                         "received_power_dbm":result["received_power_dbm"],"link_margin_db":result["link_margin_db"]})
    return {"sweep":"RANGE","rows":rows}


def elevation_sweep(config,elevations_deg,pointing_error_urad=20):
    c=_enabled(config); base=c.disturbance.atmosphere.attenuation*c.disturbance.atmosphere.cloud_attenuation
    rows=[]
    for elevation in elevations_deg:
        airmass=min(4.0,1/max(.15,math.sin(math.radians(elevation))))
        atmosphere=base**airmass if c.disturbance.atmosphere.enabled else 1.0
        result=_isolated(c,pointing_error_urad*1e-6,atmosphere=atmosphere)
        rows.append({"elevation_deg":elevation,"airmass_proxy":airmass,"atmospheric_transmission":atmosphere,
                     "received_power_dbm":result["received_power_dbm"],"link_margin_db":result["link_margin_db"]})
    return {"sweep":"GROUND_SAT_ELEVATION","fidelity":"simplified airmass proxy","rows":rows}


def candidate_pipeline(config):
    data=_enabled(config).model_dump(mode="json")
    data.update(estimator="akf_r",predictor="gru",controller="ff_pid",prediction_horizon_s=.05,pipeline_preset="CUSTOM")
    data["vision"].update(algorithm="gradient_centroid",correction="cnn_residual",cnn_correction=True)
    data["cnn"]["enabled"]=True
    return validate_config(data)


def baseline_pipeline(config):
    data=_enabled(config).model_dump(mode="json")
    data.update(estimator="kalman",predictor="none",controller="pid",pipeline_preset="CUSTOM")
    data["vision"].update(algorithm="centroid",correction="none",cnn_correction=False)
    data["cnn"]["enabled"]=False
    return validate_config(data)


def compare_link_pipelines(config,config_a=None,config_b=None,seeds=(41,42),duration_s=2.0,store=None):
    a=baseline_pipeline(config) if config_a is None else _enabled(config_a)
    b=candidate_pipeline(config) if config_b is None else _enabled(config_b)
    groups={"BASELINE":[],"AI_CANDIDATE":[]}; rows=[]
    for seed in seeds:
        for label,current in (("BASELINE",a),("AI_CANDIDATE",b)):
            data=current.model_dump(mode="json"); data.update(seed=int(seed),run_index=0)
            engine=SimulationEngine(config=data,retain_frames=0)
            for _ in range(math.ceil(duration_s*engine.config.fps)): engine.step(publish=False,annotate=False)
            metrics=engine.metrics.summary(); groups[label].append(metrics)
            rows.append({"variant":label,"seed":seed,"metrics":metrics,"resolved_config":data})
    keys=("pointing_rmse_rad","p95_pointing_error_rad","mean_pointing_loss_db","mean_received_power_dbm",
          "p05_received_power_dbm","mean_link_margin_db","link_availability_percent","locked_percentage")
    summary={label:{key:stats([item[key] for item in values if item.get(key) is not None]) for key in keys}
             for label,values in groups.items()}
    result={"mode":"PAIRED_LINK_COMPARISON","seeds":list(seeds),"duration_s":duration_s,"summary":summary,"rows":rows,
            "same_physical_seed":True,"candidate_status":"EXPERIMENTAL_NOT_PROMOTED"}
    if store is not None: result["result_id"]=store.append_lab_result("OPTICAL_LINK_COMPARISON",a,result)
    return result
