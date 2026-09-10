"""Recorded-measurement replay and paired open/closed-loop estimator studies."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import json
import numpy as np

from src.core.contracts import Measurement
from src.core.engine import SimulationEngine
from src.core.metrics import stats
from src.core.registry import REGISTRY, Stage, validate_config
from src.core.scenarios import resolve_config
from src.estimation.geometry import CameraCalibration, pixel_to_angles


DEFAULT_ESTIMATORS = ("none", "kf_cv", "ekf_angular", "ukf_angular", "akf_r")


@dataclass(frozen=True)
class ReplaySample:
    timestamp: float
    measurement: Measurement
    true_pixel_evaluation_only: tuple[float, float] | None
    dropout: bool = False

    def as_dict(self):
        return dict(timestamp=self.timestamp, measurement=self.measurement.as_dict(),
                    true_pixel_evaluation_only=None if self.true_pixel_evaluation_only is None else list(self.true_pixel_evaluation_only),
                    dropout=self.dropout)

    @classmethod
    def from_dict(cls, value):
        truth = value.get("true_pixel_evaluation_only")
        return cls(value["timestamp"], Measurement.from_dict(value["measurement"]),
                   None if truth is None else tuple(truth), bool(value.get("dropout", False)))


@dataclass(frozen=True)
class EstimatorReplaySequence:
    calibration: CameraCalibration
    samples: tuple[ReplaySample, ...]
    scenario_metadata: dict

    def as_dict(self):
        return dict(kind="EstimatorReplaySequence", calibration=self.calibration.as_dict(),
                    samples=[sample.as_dict() for sample in self.samples], scenario_metadata=self.scenario_metadata)

    @classmethod
    def from_dict(cls, value):
        calibration = CameraCalibration(**value["calibration"])
        return cls(calibration, tuple(ReplaySample.from_dict(sample) for sample in value["samples"]),
                   dict(value.get("scenario_metadata") or {}))

    def save(self, path: str | Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def record_measurement_sequence(config, duration_s: float | None = None) -> EstimatorReplaySequence:
    """Record one camera/vision realization; ground truth is stored only beside samples."""
    config = validate_config(config)
    engine = SimulationEngine(config=config)
    frames = max(1, math.ceil((duration_s or config.duration_s) * config.fps))
    samples = []
    for _ in range(frames):
        _, telemetry = engine.step(publish=False, annotate=False)
        measurement = engine.last_measurement
        truth = telemetry.get("ground_truth_pixel")
        samples.append(ReplaySample(measurement.timestamp, measurement,
                                    None if truth is None else tuple(truth),
                                    bool(telemetry.get("beacon_dropout") or telemetry.get("frame_dropout"))))
    return EstimatorReplaySequence(CameraCalibration.from_config(config), tuple(samples),
                                   dict(scenario=config.scenario.value, scenario_preset=config.scenario_preset,
                                        seed=config.seed, run_index=config.run_index,
                                        simulation_schema_version=config.simulation_schema_version,
                                        vision_algorithm=config.vision.algorithm,
                                        resolved_config=config.model_dump(mode="json")))


def replay_estimator(sequence: EstimatorReplaySequence, config, estimator_name: str):
    config_data = validate_config(config).model_dump(mode="json")
    config_data["estimator"] = estimator_name; config_data["pipeline_preset"] = "CUSTOM"
    resolved = validate_config(config_data)
    implementation = REGISTRY[Stage.ESTIMATOR].get(estimator_name)
    if implementation is None:
        raise ValueError(f"Unavailable estimator '{estimator_name}'")
    estimator = implementation(resolved, None)
    errors=[]; angular_errors=[]; dropout_errors=[]; nis_values=[]; nees_values=[]; traces=[]; latencies=[]
    coverage={"1sigma":0,"2sigma":0,"3sigma":0}; coverage_total=0; rows=[]
    previous_timestamp = None
    for sample in sequence.samples:
        dt = 1 / resolved.fps if previous_timestamp is None else max(1e-9, sample.timestamp - previous_timestamp)
        previous_timestamp = sample.timestamp
        estimate = estimator.update(sample.measurement, dt)
        truth = sample.true_pixel_evaluation_only
        error = angular_error = nees = normalized = None
        if truth is not None and estimate.valid:
            difference = np.asarray(estimate.pixel) - np.asarray(truth)
            error = float(np.linalg.norm(difference)); errors.append(error)
            angular_error = float(np.linalg.norm(pixel_to_angles(estimate.pixel, sequence.calibration) -
                                                 pixel_to_angles(truth, sequence.calibration)))
            angular_errors.append(angular_error)
            if sample.dropout: dropout_errors.append(error)
            if estimate.image_covariance is not None:
                try:
                    nees = float(difference @ np.linalg.solve(estimate.image_covariance, difference))
                    nees_values.append(nees); normalized = math.sqrt(max(0.0, nees)); coverage_total += 1
                    for label, radius in (("1sigma",1),("2sigma",2),("3sigma",3)):
                        coverage[label] += int(normalized <= radius)
                except np.linalg.LinAlgError:
                    pass
        if estimate.nis is not None: nis_values.append(estimate.nis)
        if estimate.covariance is not None: traces.append(float(np.trace(estimate.covariance)))
        if estimate.update_latency_ms is not None: latencies.append(estimate.update_latency_ms)
        rows.append(dict(timestamp=sample.timestamp, measurement=sample.measurement.as_dict(),
                         truth_evaluation_only=truth, estimate=None if not estimate.valid else list(estimate.pixel),
                         error_px=error, angular_error_rad=angular_error, nis=estimate.nis, nees_position=nees,
                         prediction_only=estimate.prediction_only, measurement_used=estimate.measurement_used,
                         uncertainty_ellipse=estimate.uncertainty_ellipse,
                         numerical_events=list(estimate.numerical_events), latency_ms=estimate.update_latency_ms))
    summary = dict(samples=len(sequence.samples), valid_estimates=len(errors),
                   rmse_px=math.sqrt(float(np.mean(np.square(errors)))) if errors else None,
                   mae_px=stats(errors)["mean"], median_px=stats(errors)["median"], p95_px=stats(errors)["p95"],
                   maximum_px=stats(errors)["max"], angular_rmse_rad=math.sqrt(float(np.mean(np.square(angular_errors)))) if angular_errors else None,
                   dropout_rmse_px=math.sqrt(float(np.mean(np.square(dropout_errors)))) if dropout_errors else None,
                   mean_nis=stats(nis_values)["mean"], mean_nees_position=stats(nees_values)["mean"],
                   mean_covariance_trace=stats(traces)["mean"], mean_latency_ms=stats(latencies)["mean"],
                   p95_latency_ms=stats(latencies)["p95"], measurement_rejection_count=estimator.measurement_rejection_count,
                   numerical_recovery_count=estimator.numerical_recovery_count,
                   uncertainty_coverage={key:(100*value/coverage_total if coverage_total else None) for key,value in coverage.items()})
    return dict(estimator=estimator_name, metadata=estimator.metadata(), summary=summary, rows=rows)


def compare_estimators_open_loop(sequence: EstimatorReplaySequence, config, estimators=DEFAULT_ESTIMATORS):
    results = {name: replay_estimator(sequence, config, name) for name in estimators}
    return dict(kind="OPEN_LOOP_ESTIMATOR_REPLAY", shared_measurement_samples=len(sequence.samples),
                sequence_metadata=sequence.scenario_metadata, estimators=results)


def estimator_parameter_sweep(sequence: EstimatorReplaySequence, config, parameter: str, values,
                              estimators=("kf_cv", "ekf_angular", "ukf_angular")):
    allowed = {"estimation.process_noise_px_s2", "estimation.angular_process_noise_rad_s2",
               "estimation.measurement_noise_px2"}
    if parameter not in allowed:
        raise ValueError(f"Estimator sweep parameter must be one of {sorted(allowed)}")
    points=[]
    for value in values:
        data=validate_config(config).model_dump(mode="json"); _set_path(data,parameter,float(value))
        comparison=compare_estimators_open_loop(sequence,data,estimators)
        points.append(dict(value=float(value),estimators={name:result["summary"] for name,result in comparison["estimators"].items()}))
    return dict(kind="ESTIMATOR_PARAMETER_SWEEP",parameter=parameter,shared_measurement_samples=len(sequence.samples),points=points)


ESTIMATOR_CONDITIONS = ("NOMINAL", "HIGH_MEASUREMENT_NOISE", "WEAK_BEACON", "HIGH_VIBRATION",
                        "MANOEUVRE", "SHORT_DROPOUT", "LONG_DROPOUT", "DISTRACTOR_OUTLIER",
                        "GROUND_SAT_STRESS", "UAV_AGGRESSIVE", "COMBINED_STRESS")


def _set_path(value: dict, path: str, replacement):
    target=value; parts=path.split(".")
    for part in parts[:-1]: target=target[part]
    target[parts[-1]]=replacement


def _benchmark_config(base, condition, seed):
    if condition == "GROUND_SAT_STRESS":
        data = resolve_config("GROUND_SAT_TURBULENT").model_dump(mode="json")
    elif condition == "UAV_AGGRESSIVE":
        data = resolve_config("UAV_UAV_AGGRESSIVE").model_dump(mode="json")
    else:
        data = copy.deepcopy(validate_config(base).model_dump(mode="json"))
    patches = {
        "HIGH_MEASUREMENT_NOISE":{"camera.noise_sigma":22},
        "WEAK_BEACON":{"disturbance.beacon.nominal_intensity":70,"camera.noise_sigma":12},
        "HIGH_VIBRATION":{"disturbance.vibration.enabled":True,"disturbance.vibration.level":"HIGH",
                          "disturbance.vibration.stochastic_sigma_rad":0.001},
        "MANOEUVRE":{"disturbance.manoeuvre.enabled":True,"disturbance.manoeuvre.start_s":0.4,
                     "disturbance.manoeuvre.velocity_impulse_m_s":(4,2,-1)},
        "SHORT_DROPOUT":{"disturbance.beacon.hard_dropout_windows":({"start_s":0.4,"end_s":0.7,"factor":0},)},
        "LONG_DROPOUT":{"disturbance.beacon.hard_dropout_windows":({"start_s":0.3,"end_s":1.2,"factor":0},)},
        "DISTRACTOR_OUTLIER":{"disturbance.distractors.enabled":True,"disturbance.distractors.count":3,
                              "disturbance.distractors.intensity":245},
        "COMBINED_STRESS":{"camera.noise_sigma":18,"disturbance.beacon.nominal_intensity":90,
                           "disturbance.vibration.enabled":True,"disturbance.vibration.level":"HIGH",
                           "disturbance.vibration.stochastic_sigma_rad":0.0008,
                           "disturbance.distractors.enabled":True,"disturbance.distractors.count":2,
                           "disturbance.sensor.defocus_sigma_px":2.5},
    }
    for path,value in patches.get(condition,{}).items(): _set_path(data,path,value)
    data.update(seed=int(seed),run_index=0,fps=10,duration_s=1.5,pipeline_preset="CUSTOM")
    data["camera"].update(width_px=160,height_px=120,focal_length_px=130)
    data["vision"]["algorithm"]="weighted_centroid"
    return validate_config(data)


def run_estimator_benchmark(config, seeds=(41,42), estimators=DEFAULT_ESTIMATORS,
                            conditions=ESTIMATOR_CONDITIONS, store=None):
    conditions=tuple(conditions or ESTIMATOR_CONDITIONS)
    rows=[]
    for condition in conditions:
        if condition not in ESTIMATOR_CONDITIONS: raise ValueError(f"Unknown estimator condition '{condition}'")
        for seed in seeds:
            resolved=_benchmark_config(config,condition,seed)
            sequence=record_measurement_sequence(resolved,1.5)
            comparison=compare_estimators_open_loop(sequence,resolved,estimators)
            for name,result in comparison["estimators"].items():
                rows.append(dict(condition=condition,seed=int(seed),estimator=name,**result["summary"]))
    summary={}
    for condition in conditions:
        summary[condition]={}
        for name in estimators:
            selected=[row for row in rows if row["condition"]==condition and row["estimator"]==name]
            summary[condition][name]={metric:stats([row[metric] for row in selected if row.get(metric) is not None])["mean"]
                                      for metric in ("rmse_px","p95_px","dropout_rmse_px","mean_nis","mean_nees_position",
                                                     "mean_covariance_trace","mean_latency_ms","measurement_rejection_count",
                                                     "numerical_recovery_count")}
    paired=[]
    for row in rows:
        if row["estimator"]=="kf_cv": continue
        baseline=next((item for item in rows if item["condition"]==row["condition"] and item["seed"]==row["seed"] and item["estimator"]=="kf_cv"),None)
        if baseline and row.get("rmse_px") is not None and baseline.get("rmse_px") is not None:
            paired.append(dict(condition=row["condition"],seed=row["seed"],estimator=row["estimator"],
                               rmse_delta_vs_kf_cv=row["rmse_px"]-baseline["rmse_px"]))
    output=dict(kind="ESTIMATOR_BENCHMARK",seeds=list(seeds),conditions=list(conditions),
                estimators=list(estimators),summary=summary,paired_differences=paired,rows=rows)
    if store is not None: store.append_lab_result("ESTIMATOR_BENCHMARK",validate_config(config),output)
    return output
