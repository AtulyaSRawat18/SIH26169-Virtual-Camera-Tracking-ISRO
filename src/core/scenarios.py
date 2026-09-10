"""Data-driven scenario and pipeline precedence resolution."""
import copy
import json
from pathlib import Path
from src.core.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = ROOT / "configs" / "scenarios.json"
GLOBAL_PATH = ROOT / "configs" / "integrated.json"

OPTICAL_DEFAULTS = {
    "satellite_satellite": {"enabled": False,"wavelength_nm":1550,"transmit_power_w":1.0,"beam_divergence_urad":25,"receiver_aperture_m":.15,"transmitter_efficiency":.78,"receiver_efficiency":.72,"receiver_sensitivity_dbm":-45,"range_mode":"AUTO_FROM_SCENARIO","atmosphere_mode":"FROM_SCENARIO"},
    "ground_satellite": {"enabled": False,"wavelength_nm":1550,"transmit_power_w":2.0,"beam_divergence_urad":35,"receiver_aperture_m":.20,"transmitter_efficiency":.72,"receiver_efficiency":.68,"receiver_sensitivity_dbm":-43,"range_mode":"AUTO_FROM_SCENARIO","atmosphere_mode":"FROM_SCENARIO"},
    "uav_ground": {"enabled": False,"wavelength_nm":1550,"transmit_power_w":.5,"beam_divergence_urad":150,"receiver_aperture_m":.08,"transmitter_efficiency":.68,"receiver_efficiency":.62,"receiver_sensitivity_dbm":-40,"range_mode":"AUTO_FROM_SCENARIO","atmosphere_mode":"FROM_SCENARIO"},
    "uav_uav": {"enabled": False,"wavelength_nm":1550,"transmit_power_w":.5,"beam_divergence_urad":200,"receiver_aperture_m":.06,"transmitter_efficiency":.65,"receiver_efficiency":.60,"receiver_sensitivity_dbm":-39,"range_mode":"AUTO_FROM_SCENARIO","atmosphere_mode":"FROM_SCENARIO"},
    "uav_satellite": {"enabled": False,"wavelength_nm":1550,"transmit_power_w":1.5,"beam_divergence_urad":50,"receiver_aperture_m":.18,"transmitter_efficiency":.70,"receiver_efficiency":.65,"receiver_sensitivity_dbm":-44,"range_mode":"AUTO_FROM_SCENARIO","atmosphere_mode":"FROM_SCENARIO"},
}


def _load(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def deep_merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def catalogue():
    return _load(SCENARIOS_PATH)


def _resolved_preset(data,name,seen=None):
    seen=set() if seen is None else seen
    if name in seen: raise ValueError(f"Cyclic scenario preset inheritance at '{name}'")
    seen.add(name); value=copy.deepcopy(data["presets"][name]); parent=value.pop("extends",None)
    if parent is None: return value
    if parent not in data["presets"]: raise ValueError(f"Unknown parent scenario preset '{parent}'")
    return deep_merge(_resolved_preset(data,parent,seen),value)


def resolve_config(preset="SAT_SAT_NOMINAL", pipeline="DEFAULT_STABLE", overrides=None):
    data = catalogue()
    if preset not in data["presets"]:
        raise ValueError(f"Unknown scenario preset '{preset}'")
    if pipeline not in data["pipelines"]:
        raise ValueError(f"Unknown pipeline preset '{pipeline}'")
    pipeline_data = data["pipelines"][pipeline]
    if pipeline_data["status"] != "runnable":
        raise ValueError(f"Pipeline '{pipeline}' is {pipeline_data['status']}")
    algorithm = {k:v for k,v in pipeline_data.items() if k not in ("status", "required", "evidence")}
    resolved = deep_merge(_load(GLOBAL_PATH), _resolved_preset(data,preset))
    scenario_name=resolved["scenario"]
    resolved["optical_link"]=deep_merge(OPTICAL_DEFAULTS[scenario_name],resolved.get("optical_link"))
    resolved = deep_merge(resolved, algorithm)
    resolved = deep_merge(resolved, overrides)
    resolved.update(scenario_preset=preset, pipeline_preset=pipeline,
                    scenario_version=data["scenario_version"])
    return ExperimentConfig.model_validate(resolved)


def public_catalogue():
    data = catalogue()
    return {"scenario_version": data["scenario_version"],
            "presets": [{"name": name, "scenario": value["scenario"]} for name,value in data["presets"].items()],
            "pipelines": data["pipelines"]}
