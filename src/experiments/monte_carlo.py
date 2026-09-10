"""Headless Monte Carlo and mandatory common-random-number comparison."""
import copy
import math
import numpy as np
import subprocess
from src.core.engine import SimulationEngine
from src.core.randomness import derive_seed
from src.core.registry import validate_config
from src.core.metrics import stats
from src.experiments.storage import ExperimentStore

METRICS=('rmse_tracking_error_px','mean_tracking_error_px','median_tracking_error_px','max_tracking_error_px','p95_tracking_error_px',
         'cv_localization_rmse_px','classical_localization_rmse_px','corrected_localization_rmse_px','estimator_rmse_px','prediction_rmse_px',
         'control_rmse_rad','actuator_rate_rmse_rad_s','measurement_availability_percent','locked_percentage','lock_losses',
         'acquisition_time_s','mean_reacquisition_time_s','false_lock_probability',
         'rate_saturation_percent','position_saturation_percent','mean_processing_latency_ms','p95_processing_latency_ms',
         'mean_vision_latency_ms','p95_vision_latency_ms','mean_estimator_latency_ms','p95_estimator_latency_ms',
         'mean_correction_latency_ms','p95_correction_latency_ms','mean_predictor_latency_ms','p95_predictor_latency_ms',
         'lead_prediction_rmse_px','correction_fallback_count','predictor_fallback_count',
         'mean_controller_latency_ms','control_effort_integral','control_smoothness_integral',
         'controller_command_saturation_percent','acceleration_saturation_percent','controller_fallback_count',
         'pointing_rmse_rad','p95_pointing_error_rad','mean_pointing_loss_db','median_pointing_loss_db',
         'p95_pointing_loss_db','mean_received_power_dbm','p05_received_power_dbm','mean_link_margin_db',
         'link_availability_percent','link_outage_percent','longest_link_outage_s','link_interruption_count',
         'mean_nis','p95_nis','mean_nees_position','p95_nees_position','mean_covariance_trace',
         'measurement_rejection_count','numerical_recovery_count')


def run_seed(base_seed,index): return derive_seed(base_seed,index,'run')%(2**32)


def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None


def _set_path(data,path,value):
    target=data
    parts=path.split('.')
    for part in parts[:-1]: target=target[part]
    target[parts[-1]]=value


def sample_parameters(config,distributions,base_seed,index):
    data=config.model_dump(mode='json'); sampled={}
    sampling_seed=derive_seed(base_seed,index,'parameter_sampling')
    rng=np.random.default_rng(sampling_seed)
    for path,spec in (distributions or {}).items():
        kind=spec['distribution']
        if kind=='fixed': value=spec['value']
        elif kind=='uniform': value=float(rng.uniform(spec['low'],spec['high']))
        elif kind=='normal': value=float(rng.normal(spec['mean'],spec['std']))
        elif kind=='choice': value=rng.choice(spec['values']).item()
        else: raise ValueError(f"Unsupported distribution '{kind}'")
        _set_path(data,path,value); sampled[path]=value
    data['seed']=base_seed; data['run_index']=index
    return validate_config(data),sampled


def aggregate(results):
    output={key:stats([r['metrics'][key] for r in results if r['metrics'].get(key) is not None]) for key in METRICS}
    n=len(results)
    output['runs']=n
    output['acquisition_success_rate']=sum(r['metrics']['time_to_first_detection_s'] is not None for r in results)/n if n else 0
    output['lock_success_rate']=sum(r['metrics']['time_to_first_lock_s'] is not None for r in results)/n if n else 0
    output['reacquisition_success_rate']=sum((r['metrics']['reacquisition_success_percent'] or 0)>0 for r in results)/n if n else 0
    output['false_lock_probability']=float(np.mean([r['metrics']['false_lock_probability'] for r in results])) if n else 0
    failures={}
    for run in results:
        cause=run['failure_cause']
        if cause: failures[cause]=failures.get(cause,0)+1
    output['failure_counts']=failures
    return output


def _execute(config,base_seed,index,duration_s,distributions,summary_only):
    resolved,sampled=sample_parameters(config,distributions,base_seed,index)
    engine=SimulationEngine(config=resolved,retain_frames=0 if summary_only else max(1200,math.ceil(duration_s*resolved.fps)))
    last=None
    for _ in range(math.ceil(duration_s*resolved.fps)): _,last=engine.step(publish=False,annotate=False)
    metrics=engine.metrics.summary(); metadata=engine.metadata(); algorithm_details=engine.algorithm_metadata()
    return dict(metrics=metrics,failure_cause=metrics['dominant_failure_cause'],run_seed=run_seed(base_seed,index),
                subsystem_seeds=metadata['subsystem_seeds'],sampled_inputs=sampled,resolved_config=resolved.model_dump(mode='json'),
                sampling_metadata={path:{"configured_distribution":spec,"effective_value":sampled[path],
                                         "subsystem_seed":derive_seed(base_seed,index,'parameter_sampling')}
                                   for path,spec in (distributions or {}).items()},
                algorithm_versions=metadata['algorithm_versions'],vision_metadata=algorithm_details['vision'],
                correction_metadata=algorithm_details['correction'],estimator_metadata=algorithm_details['estimator'],
                predictor_metadata=algorithm_details['predictor'],controller_metadata=algorithm_details['controller'],
                optical_metadata=metadata.get('optical_model'),
                measurement_quality=None if last is None else last.get('measurement_quality'),
                telemetry=None if summary_only else list(engine.metrics.frames),last=last)


def run_monte_carlo(config,runs=100,base_seed=42,duration_s=None,distributions=None,summary_only=True,store=None):
    if runs<1 or runs>10000: raise ValueError('runs must be between 1 and 10000')
    config=validate_config(config); duration_s=duration_s or config.duration_s; store=store or ExperimentStore()
    batch=store.create_batch(config,'MONTE_CARLO',base_seed,runs); results=[]
    pipeline={x:getattr(config,x) for x in ('motion_model','estimator','predictor','controller','reacquisition')}
    pipeline['vision']=config.vision.algorithm; pipeline['correction']=config.vision.correction
    for index in range(runs):
        outcome=_execute(config,base_seed,index,duration_s,distributions,summary_only)
        record=dict(outcome,batch_id=batch,pair_index=index,variant='A',git_commit_hash=git_commit(),simulation_schema_version=config.simulation_schema_version,
                    physics_model_version=config.physics_model_version,scenario_version=config.scenario_version,scenario=config.scenario.value,
                    scenario_preset=config.scenario_preset,algorithm_pipeline=pipeline,base_seed=base_seed,duration_s=duration_s,completion_status='COMPLETE')
        record.pop('last'); store.append_run(record); results.append(record)
    return dict(batch_id=batch,results=results,aggregate=aggregate(results))


def _physical_config(config):
    data=config.model_dump(mode='json')
    for key in ('pipeline_preset','vision','cnn','estimator','predictor','controller','reacquisition','kalman','pid','lock',
                'prediction_horizon_s','temporal','latency','ff_pid','gain_scheduled_pid','lqr','mpc','controller_lab'):
        data.pop(key,None)
    return data


def compare_paired(config_a,config_b,runs=10,base_seed=42,duration_s=None,summary_only=True,store=None):
    a,b=validate_config(config_a),validate_config(config_b)
    if _physical_config(a)!=_physical_config(b): raise ValueError('Paired comparison requires identical resolved physical/scenario configuration')
    duration_s=duration_s or a.duration_s; store=store or ExperimentStore(); batch=store.create_batch(a,'PAIRED_COMPARISON',base_seed,runs)
    groups={'A':[],'B':[]}
    for index in range(runs):
        outcomes={'A':_execute(a,base_seed,index,duration_s,None,summary_only),'B':_execute(b,base_seed,index,duration_s,None,summary_only)}
        if outcomes['A']['subsystem_seeds']!=outcomes['B']['subsystem_seeds']: raise RuntimeError('Paired RNG streams diverged')
        for variant,config in (('A',a),('B',b)):
            outcome=outcomes[variant]
            pipeline={x:getattr(config,x) for x in ('motion_model','estimator','predictor','controller','reacquisition')}
            pipeline.update(vision=config.vision.algorithm,correction=config.vision.correction)
            record=dict(outcome,batch_id=batch,pair_index=index,variant=variant,git_commit_hash=git_commit(),simulation_schema_version=config.simulation_schema_version,
                        physics_model_version=config.physics_model_version,scenario_version=config.scenario_version,scenario=config.scenario.value,
                        scenario_preset=config.scenario_preset,algorithm_pipeline=pipeline,base_seed=base_seed,duration_s=duration_s,completion_status='COMPLETE')
            record.pop('last'); store.append_run(record); groups[variant].append(record)
    deltas=[groups['B'][i]['metrics']['rmse_tracking_error_px']-groups['A'][i]['metrics']['rmse_tracking_error_px'] for i in range(runs)]
    return dict(batch_id=batch,A=aggregate(groups['A']),B=aggregate(groups['B']),paired_rmse_delta=stats(deltas),pairs=runs,
                pipeline_A=groups['A'][0]['algorithm_pipeline'],pipeline_B=groups['B'][0]['algorithm_pipeline'],
                pair_records=[{"pair_index":i,"A":groups['A'][i]['metrics'],"B":groups['B'][i]['metrics']} for i in range(runs)])
