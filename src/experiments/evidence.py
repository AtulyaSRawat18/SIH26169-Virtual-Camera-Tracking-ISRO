"""Evidence backbone for stage-wise errors, paired claims and controlled sweeps."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import copy
from html import escape
from io import StringIO
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.core.config import Scenario
from src.core.metrics import stats
from src.core.registry import validate_config
from src.experiments.monte_carlo import METRICS, compare_paired, run_monte_carlo
from src.experiments.storage import ExperimentStore


ERROR_REGISTRY = {
    "e_measurement": {"field": "classical_measurement_error_px", "unit": "px", "stage": "vision", "truth_only_evaluation": True},
    "e_cnn": {"field": "measurement_error_px", "unit": "px", "stage": "correction", "truth_only_evaluation": True},
    "e_estimation": {"field": "estimator_error_px", "unit": "px", "stage": "estimation", "truth_only_evaluation": True},
    "e_prediction": {"field": "prediction_error_px", "unit": "px", "stage": "prediction_at_horizon", "truth_only_evaluation": True},
    "e_control": {"field": "control_error_rad", "unit": "rad", "stage": "control", "truth_only_evaluation": False},
    "e_actuator": {"field": "actuator_rate_error_rad_s", "unit": "rad/s", "stage": "actuator", "truth_only_evaluation": False},
    "e_pointing": {"field": "final_pointing_error_rad", "unit": "rad", "stage": "optical_axis", "truth_only_evaluation": True},
}

ALL_SCENARIOS = tuple(x.value for x in Scenario)
ATMOSPHERIC_SCENARIOS = tuple(x for x in ALL_SCENARIOS if x != Scenario.SATELLITE_SATELLITE.value)
UAV_SCENARIOS = tuple(x for x in ALL_SCENARIOS if "uav" in x)

VARIABLE_REGISTRY = {
    "platform.angular_rate_rad_s": {"label": "Angular LOS rate", "category": "geometry_navigation", "unit": "rad/s", "scenarios": ALL_SCENARIOS},
    "platform.altitude_m": {"label": "Platform altitude", "category": "geometry_navigation", "unit": "m", "scenarios": UAV_SCENARIOS},
    "platform.speed_mps": {"label": "Platform speed", "category": "platform", "unit": "m/s", "scenarios": UAV_SCENARIOS},
    "disturbance.vibration.stochastic_sigma_rad": {"label": "Vibration amplitude", "category": "platform", "unit": "rad", "scenarios": ALL_SCENARIOS},
    "disturbance.attitude.white_noise_sigma_rad": {"label": "Attitude noise", "category": "platform", "unit": "rad", "scenarios": ALL_SCENARIOS},
    "disturbance.atmosphere.turbulence_strength": {"label": "Turbulence strength", "category": "environment", "unit": "fraction", "scenarios": ATMOSPHERIC_SCENARIOS},
    "disturbance.atmosphere.cloud_attenuation": {"label": "Cloud transmission", "category": "environment", "unit": "fraction", "scenarios": ATMOSPHERIC_SCENARIOS},
    "disturbance.atmosphere.beam_wander_sigma_rad": {"label": "Beam wander", "category": "optics", "unit": "rad", "scenarios": ATMOSPHERIC_SCENARIOS},
    "optical_link.beam_divergence_urad": {"label": "Beam divergence", "category": "optics", "unit": "urad", "scenarios": ALL_SCENARIOS},
    "disturbance.beacon.nominal_intensity": {"label": "Beacon intensity", "category": "beacon", "unit": "DN", "scenarios": ALL_SCENARIOS},
    "camera.noise_sigma": {"label": "Image noise sigma", "category": "camera_sensor", "unit": "DN", "scenarios": ALL_SCENARIOS},
    "disturbance.sensor.defocus_sigma_px": {"label": "Defocus blur sigma", "category": "camera_sensor", "unit": "px", "scenarios": ALL_SCENARIOS},
    "disturbance.gimbal.command_latency_s": {"label": "Gimbal latency", "category": "actuator", "unit": "s", "scenarios": ALL_SCENARIOS},
    "pid.max_rate_rad_s": {"label": "Gimbal maximum rate", "category": "actuator", "unit": "rad/s", "scenarios": ALL_SCENARIOS},
    "disturbance.beacon.random_dropout_probability": {"label": "Beacon dropout probability", "category": "failure_events", "unit": "probability", "scenarios": ALL_SCENARIOS},
    "disturbance.sensor.frame_dropout_probability": {"label": "Frame dropout probability", "category": "failure_events", "unit": "probability", "scenarios": ALL_SCENARIOS},
}

LEVEL_FACTORS = {"NOMINAL": 0.0, "LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "STRESS": 1.0}


def relevant_variables(scenario: str):
    return {path: meta for path, meta in VARIABLE_REGISTRY.items() if scenario in meta["scenarios"]}


def _get_path(data, path):
    target = data
    for part in path.split("."):
        target = target[part]
    return target


def _set_path(data, path, value):
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def validate_distributions(config, distributions):
    data = validate_config(config).model_dump(mode="json")
    result = {}
    for path, spec in (distributions or {}).items():
        if path not in VARIABLE_REGISTRY:
            raise ValueError(f"Unknown or non-controllable evidence variable '{path}'")
        _get_path(data, path)
        kind = spec.get("distribution")
        if kind == "fixed":
            if "value" not in spec: raise ValueError(f"Fixed distribution for {path} requires value")
        elif kind == "uniform":
            if spec.get("high") < spec.get("low"): raise ValueError(f"Uniform high must be >= low for {path}")
        elif kind == "normal":
            if spec.get("std", -1) < 0: raise ValueError(f"Normal std must be non-negative for {path}")
        else:
            raise ValueError(f"{path} must use fixed, uniform or normal sampling")
        result[path] = copy.deepcopy(spec)
    return result


def apply_level_preset(config, level):
    """Resolve simple UI labels into explicit, inspectable physical values."""
    if level not in LEVEL_FACTORS: raise ValueError(f"Unknown error level '{level}'")
    data = validate_config(config).model_dump(mode="json")
    f = LEVEL_FACTORS[level]
    data["disturbance"]["preset_level"] = level
    data["disturbance"]["vibration"]["enabled"] = f > 0
    data["disturbance"]["vibration"]["stochastic_sigma_rad"] = f * 30e-6
    data["camera"]["noise_sigma"] = 2 + f * 20
    data["disturbance"]["sensor"]["defocus_sigma_px"] = f * 2.0
    data["disturbance"]["beacon"]["random_dropout_probability"] = f * 0.12
    data["disturbance"]["gimbal"]["command_latency_s"] = f * 0.12
    if data["scenario"] != Scenario.SATELLITE_SATELLITE.value:
        data["disturbance"]["atmosphere"]["enabled"] = True
        data["disturbance"]["atmosphere"]["turbulence_strength"] = f
        data["disturbance"]["atmosphere"]["cloud_attenuation"] = 1 - 0.55 * f
        data["disturbance"]["atmosphere"]["beam_wander_sigma_rad"] = f * 25e-6
    return validate_config(data)


def waterfall_from_record(record):
    budget = record.get("error_budget") or {}
    rows = []
    previous = None
    for key in ("e_measurement", "e_cnn", "e_estimation", "e_prediction", "e_control", "e_actuator", "e_pointing"):
        definition = ERROR_REGISTRY[key]
        value = budget.get(definition["field"])
        if value is None: continue
        change = None if previous is None or definition["unit"] != previous[1] else value - previous[0]
        relative = None if change is None or abs(previous[0]) < 1e-12 else 100 * change / abs(previous[0])
        rows.append({"error": key, "stage": definition["stage"], "value": value, "unit": definition["unit"],
                     "change_from_previous": change, "relative_change_percent": relative,
                     "status": None if change is None else ("IMPROVED" if change < 0 else "DEGRADED" if change > 0 else "UNCHANGED")})
        previous = (value, definition["unit"])
    return rows


def bootstrap_ci(values: Iterable[float], confidence=0.95, samples=2000, seed=42):
    values = np.asarray(list(values), dtype=float)
    if not len(values): return {"low": None, "high": None, "confidence": confidence, "method": "seeded_percentile_bootstrap"}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return {"low": float(np.quantile(means, alpha)), "high": float(np.quantile(means, 1-alpha)),
            "confidence": confidence, "method": "seeded_percentile_bootstrap", "resamples": samples, "seed": seed}


def metric_change(a, b, higher_is_better=False):
    if a is None or b is None: return {"baseline": a, "candidate": b, "difference": None, "improvement_percent": None, "status": "UNAVAILABLE"}
    difference = b - a
    improvement = (difference if higher_is_better else -difference) / max(abs(a), 1e-12) * 100
    return {"baseline": a, "candidate": b, "difference": difference, "improvement_percent": improvement,
            "status": "IMPROVED" if improvement > 0 else "DEGRADED" if improvement < 0 else "UNCHANGED"}


HIGHER_BETTER = {"locked_percentage", "link_availability_percent", "measurement_availability_percent"}


def paired_statistics(pair_records, metric, seed=42):
    pairs = [(p["A"].get(metric), p["B"].get(metric)) for p in pair_records]
    pairs = [(a,b) for a,b in pairs if a is not None and b is not None]
    deltas = np.asarray([b-a for a,b in pairs], dtype=float)
    signed_improvement = deltas if metric in HIGHER_BETTER else -deltas
    std = float(np.std(signed_improvement, ddof=1)) if len(signed_improvement)>1 else 0.0
    return {"metric": metric, "pairs": len(pairs), "mean_paired_difference": float(np.mean(deltas)) if len(deltas) else None,
            "median_paired_difference": float(np.median(deltas)) if len(deltas) else None,
            "improvement_ci": bootstrap_ci(signed_improvement, seed=seed),
            "paired_effect_size_dz": (float(np.mean(signed_improvement))/std if std>0 else None),
            "direction": "higher_is_better" if metric in HIGHER_BETTER else "lower_is_better"}


def before_after(config_a, config_b, runs=10, base_seed=42, duration_s=None, store=None):
    comparison = compare_paired(config_a, config_b, runs, base_seed, duration_s, True, store)
    headline = ("rmse_tracking_error_px", "pointing_rmse_rad", "p95_pointing_error_rad", "locked_percentage",
                "mean_reacquisition_time_s", "controller_command_saturation_percent", "mean_processing_latency_ms",
                "mean_pointing_loss_db", "link_availability_percent")
    changes = {}
    paired = {}
    for metric in headline:
        changes[metric] = metric_change(comparison["A"].get(metric, {}).get("mean"), comparison["B"].get(metric, {}).get("mean"), metric in HIGHER_BETTER)
        paired[metric] = paired_statistics(comparison["pair_records"], metric, base_seed)
    comparison["changes"] = changes
    comparison["paired_statistics"] = paired
    comparison["scientific_label"] = "SELECTED_CANDIDATE_NOT_UNIVERSAL_WINNER"
    return comparison


def parameter_sweep(config, parameter, values, runs_per_point=10, base_seed=42, duration_s=None, store=None):
    distributions = validate_distributions(config, {parameter: {"distribution":"fixed", "value": values[0]}})
    del distributions
    points = []
    for point_index, value in enumerate(values):
        result = run_monte_carlo(config, runs_per_point, base_seed + point_index*10000, duration_s,
                                 {parameter:{"distribution":"fixed", "value":value}}, True, store)
        metric = result["aggregate"]["rmse_tracking_error_px"]
        raw = [r["metrics"]["rmse_tracking_error_px"] for r in result["results"] if r["metrics"]["rmse_tracking_error_px"] is not None]
        points.append({"value": value, "runs": runs_per_point, "tracking_rmse": metric,
                       "tracking_rmse_ci": bootstrap_ci(raw, seed=base_seed+point_index),
                       "p95_tracking_error": result["aggregate"]["p95_tracking_error_px"],
                       "lock_success_rate": result["aggregate"]["lock_success_rate"],
                       "link_availability": result["aggregate"]["link_availability_percent"],
                       "batch_id": result["batch_id"]})
    means = [p["tracking_rmse"]["mean"] for p in points]
    slope = None if len(points)<2 or np.ptp(np.asarray(values,dtype=float))==0 else float(np.polyfit(np.asarray(values,dtype=float), means, 1)[0])
    return {"parameter":parameter,"metadata":VARIABLE_REGISTRY[parameter],"points":points,
            "sensitivity_tracking_rmse_per_unit":slope,"seed_policy":"base_seed + point_index*10000; deterministic runs within point"}


def failure_distribution(runs):
    categories = Counter()
    for run in runs:
        cause = run.get("failure_cause") or (run.get("metrics") or {}).get("dominant_failure_cause") or "UNKNOWN"
        categories[cause] += 1
        if (run.get("metrics") or {}).get("link_availability_percent") == 0: categories["LINK_UNAVAILABLE"] += 1
    total = len(runs)
    return [{"cause":cause,"count":count,"percent":100*count/max(total,1)} for cause,count in categories.most_common()]


def pareto_front(records):
    """Return non-dominated run indices; minimize error/latency/effort, maximize lock/link."""
    keys = (("rmse_tracking_error_px",1),("mean_processing_latency_ms",1),("control_effort_integral",1),
            ("mean_reacquisition_time_s",1),("locked_percentage",-1),("link_availability_percent",-1))
    vectors=[]
    for record in records:
        m=record.get("metrics",record); vectors.append(tuple((math.inf if m.get(k) is None else sign*m[k]) for k,sign in keys))
    efficient=[]
    for i,v in enumerate(vectors):
        if not any(j!=i and all(a<=b for a,b in zip(other,v)) and any(a<b for a,b in zip(other,v)) for j,other in enumerate(vectors)):
            efficient.append(i)
    return {"metric_directions":{k:("minimize" if sign==1 else "maximize") for k,sign in keys},"efficient_indices":efficient}


def cumulative_evidence(store=None, scenario=None):
    store=store or ExperimentStore(); filters={} if scenario is None else {"scenario":scenario}
    runs=store.query_runs(filters,include_incompatible=False)
    grouped=defaultdict(list)
    for run in runs: grouped[(run["scenario"],json.dumps(run["algorithm_pipeline"],sort_keys=True))].append(run)
    groups=[]
    for (scenario_name,pipeline),items in grouped.items():
        values=[r["metrics"].get("rmse_tracking_error_px") for r in items if r["metrics"].get("rmse_tracking_error_px") is not None]
        groups.append({"scenario":scenario_name,"pipeline":json.loads(pipeline),"runs":len(items),"tracking_rmse":stats(values),
                       "tracking_rmse_ci":bootstrap_ci(values),"failures":failure_distribution(items)})
    return {"compatible_only":True,"runs":len(runs),"groups":groups,"pareto":pareto_front(runs) if runs else {"efficient_indices":[]}}


def export_evidence(payload, format="json"):
    if format=="json": return json.dumps(payload,indent=2,sort_keys=True),"application/json"
    if format in {"csv_runs","csv_summary"}:
        output=StringIO(); rows=payload.get("runs",payload if isinstance(payload,list) else [])
        flattened=[]
        for row in rows:
            base={k:row.get(k) for k in ("run_id","batch_id","pair_index","variant","scenario","scenario_preset","base_seed","run_seed","duration_s","failure_cause","completion_status")}
            base.update({f"metric_{k}":v for k,v in (row.get("metrics") or {}).items() if isinstance(v,(str,int,float,bool)) or v is None})
            flattened.append(base)
        fields=sorted({key for row in flattened for key in row})
        writer=csv.DictWriter(output,fieldnames=fields); writer.writeheader(); writer.writerows(flattened)
        return output.getvalue(),"text/csv"
    if format=="html":
        title=escape(str(payload.get("title","PAT Evidence Report")))
        body=escape(json.dumps(payload,indent=2,sort_keys=True))
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>"
                "body{font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#16202a}"
                "h1{font-size:28px}p{color:#526170}pre{white-space:pre-wrap;background:#f4f6f8;padding:18px;border-radius:8px}"
                "@media print{body{margin:0;max-width:none}}</style></head><body><h1>"+title+
                "</h1><p>Generated from compatible stored runs. Values are measured; missing evidence remains explicit.</p><pre>"+body+"</pre></body></html>"),"text/html"
    raise ValueError(f"Unsupported evidence export format '{format}'")
    if format=="markdown":
        lines=["# SIH26169 experiment evidence","",f"Generated from version-compatible records: {payload.get('runs','n/a')}",""]
        for group in payload.get("groups",[]): lines.extend([f"## {group['scenario']}","",f"Runs: {group['runs']}",f"Tracking RMSE: {group['tracking_rmse'].get('mean')}",""])
        return "\n".join(lines),"text/markdown"
    raise ValueError("format must be json or markdown")
