import inspect
import json
import sqlite3
import numpy as np
import pytest
from src.core.contracts import GimbalState
from src.core.engine import SimulationEngine
from src.core.metrics import PATStateMachine,PATState,stats
from src.core.config import LockConfig
from src.core.registry import Stage,validate_config
from src.core.scenarios import resolve_config,public_catalogue
from src.experiments.monte_carlo import run_monte_carlo,compare_paired
from src.experiments.storage import ExperimentStore


def compact(config):
    data=config.model_dump(mode='json'); data['camera'].update(width_px=80,height_px=60,focal_length_px=65)
    data.update(fps=10,duration_s=.3,time_scale=min(data['time_scale'],2)); return data


def orientation(config,seed,t=.137):
    e=SimulationEngine(config=compact(config)); d=e.stages[Stage.DISTURBANCE]
    return d.camera_orientation(GimbalState(0,0),t,.1),e


def test_vibration_seed_reproduction_difference_and_stream_isolation():
    base=resolve_config('SAT_SAT_STRESS').model_dump(mode='json'); base['seed']=42
    a=SimulationEngine(config=compact(validate_config(base))); b=SimulationEngine(config=compact(validate_config(base)))
    da,db=a.stages[Stage.DISTURBANCE],b.stages[Stage.DISTURBANCE]
    sequence_a=[da.camera_orientation(GimbalState(0,0),i/30,1/30) for i in range(8)]
    sequence_b=[db.camera_orientation(GimbalState(0,0),i/30,1/30) for i in range(8)]
    assert sequence_a==sequence_b
    changed=compact(validate_config(base)); changed['seed']=43; c=SimulationEngine(config=changed)
    assert sequence_a != [c.stages[Stage.DISTURBANCE].camera_orientation(GimbalState(0,0),i/30,1/30) for i in range(8)]
    before=a.streams['dropout'].random(5)
    # An arbitrary number of camera noise draws cannot affect the dropout stream.
    b.streams['camera_noise'].random(5000)
    after=b.streams['dropout'].random(5)
    assert np.array_equal(before,after)
    assert a.streams.seeds['vibration']!=a.streams.seeds['dropout']


def test_boresight_and_physical_vibration_change_projection():
    raw=compact(resolve_config()); raw['disturbance']['gaussian_noise']=False; raw['disturbance']['blur']=False
    clean=SimulationEngine(config=raw); world=np.array([0.,0.,10.])
    baseline=clean.camera.project(world,GimbalState(0,0))
    raw['disturbance']['boresight'].update(enabled=True,pan_offset_rad=.02,tilt_offset_rad=-.01)
    biased=SimulationEngine(config=raw); axis=biased.stages[Stage.DISTURBANCE].camera_orientation(biased.gimbal.physical_state(),.1,.1)
    shifted=biased.camera.project(world,axis)
    assert not np.allclose(baseline[:2],shifted[:2])
    raw['disturbance']['boresight']['enabled']=False
    raw['disturbance']['vibration']={'enabled':True,'level':'LOW','pan':[{'amplitude_rad':.02,'frequency_hz':1,'phase_rad':0}],'tilt':[],'stochastic_sigma_rad':0}
    vibrating=SimulationEngine(config=raw); axis=vibrating.stages[Stage.DISTURBANCE].camera_orientation(vibrating.gimbal.physical_state(),.25,.1)
    assert not np.allclose(baseline[:2],vibrating.camera.project(world,axis)[:2])


def test_weak_beacon_dropout_truth_motion_and_real_distractors():
    raw=compact(resolve_config()); raw['disturbance'].update(gaussian_noise=False,blur=False)
    strong=SimulationEngine(config=raw); strong_frame,_=strong.step(annotate=False)
    raw['disturbance']['beacon'].update(nominal_intensity=70,hard_dropout_windows=[])
    weak=SimulationEngine(config=raw); weak_frame,_=weak.step(annotate=False)
    assert weak_frame.max()<strong_frame.max()
    raw['disturbance']['beacon'].update(nominal_intensity=255,hard_dropout_windows=[{'start_s':0,'end_s':10,'factor':0}])
    raw['disturbance']['distractors'].update(enabled=False,count=0)
    dropped=SimulationEngine(config=raw); positions=[]
    for _ in range(3):
        _,telemetry=dropped.step(); positions.append(telemetry['true_relative_position_m'])
        assert telemetry['beacon_dropout'] and not telemetry['measurement_valid']
        assert telemetry['failure_cause']=='DROPOUT'
    assert positions[0]!=positions[-1]
    raw['disturbance']['distractors'].update(enabled=True,count=1,intensity=230,radius_px=5)
    distractor=SimulationEngine(config=raw); frame,telemetry=distractor.step()
    assert telemetry['distractor_pixels'] and frame.max()>=220 and telemetry['measurement_valid']
    assert list(inspect.signature(distractor.stages[Stage.VISION].measure).parameters)==['frame','timestamp']


def test_pat_full_state_paths():
    machine=PATStateMachine(LockConfig(lock_error_threshold_px=5,lock_required_frames=2,acquisition_required_frames=2,
                                       loss_error_threshold_px=20,max_missing_frames=1,reacquisition_required_frames=2))
    assert [machine.update(*x) for x in [(False,None),(True,4),(True,4),(True,4),(True,4)]] == [PATState.SEARCH,PATState.ACQUIRE,PATState.TRACK,PATState.TRACK,PATState.LOCKED]
    assert machine.update(False,4)==PATState.LOST
    assert machine.update(False,4)==PATState.REACQUIRE
    assert machine.update(True,4)==PATState.REACQUIRE
    assert machine.update(True,4)==PATState.TRACK


def test_scenario_profiles_applicability_and_uav_parameters():
    sat=resolve_config('SAT_SAT_NOMINAL'); ground=resolve_config('GROUND_SAT_TURBULENT'); uav=resolve_config('UAV_GROUND_HIGH_ALT')
    assert not sat.disturbance.atmosphere.enabled
    assert ground.disturbance.atmosphere.enabled and ground.disturbance.atmosphere.turbulence_strength>0
    assert uav.platform.target_type=='uav' and uav.platform.target_altitude_m==1200 and uav.platform.speed_mps==28
    assert sat.motion_model=='cw' and ground.motion_model=='kinematic'
    assert len(public_catalogue()['presets'])>=10


def test_stats_pipeline_validation_and_pending_hybrid():
    s=stats([1,2,3,4]); assert s['mean']==2.5 and s['median']==2.5 and s['min']==1 and s['max']==4
    raw=compact(resolve_config()); raw['pipeline_order']=['controller','vision']
    with pytest.raises(ValueError,match='Illegal pipeline ordering'): validate_config(raw)
    raw=compact(resolve_config()); raw['predictor']='transformer'
    with pytest.raises(ValueError,match='Unavailable'): validate_config(raw)
    assert resolve_config('SAT_SAT_NOMINAL','DEFAULT_STABLE').estimator=='kalman'
    assert resolve_config('SAT_SAT_NOMINAL','BEST_CLASSICAL_LOCALIZER').vision.algorithm=='gradient_centroid'
    assert resolve_config('SAT_SAT_NOMINAL','BEST_CLASSICAL_ESTIMATOR').estimator=='ukf_angular'
    with pytest.raises(ValueError,match='evidence gate failed'): resolve_config('SAT_SAT_NOMINAL','RECOMMENDED_HYBRID')
    assert resolve_config('SAT_SAT_NOMINAL','AI_MEASUREMENT_HYBRID').vision.correction=='cnn_residual'
    assert resolve_config('SAT_SAT_NOMINAL','PREDICTIVE_GRU_EXPERIMENT').predictor=='gru'


def test_monte_carlo_paired_realizations_storage_and_compatibility(tmp_path):
    store=ExperimentStore(tmp_path/'experiments.sqlite3'); config=validate_config(compact(resolve_config()))
    result=run_monte_carlo(config,runs=3,base_seed=42,duration_s=.2,store=store)
    assert len(result['results'])==3 and result['aggregate']['runs']==3
    again=run_monte_carlo(config,runs=1,base_seed=42,duration_s=.2,store=store)
    assert result['results'][0]['run_seed']==again['results'][0]['run_seed']
    assert result['results'][0]['subsystem_seeds']==again['results'][0]['subsystem_seeds']
    alternative=config.model_dump(mode='json'); alternative['estimator']='none'; alternative['pipeline_preset']='CUSTOM'
    paired=compare_paired(config,alternative,runs=2,base_seed=42,duration_s=.2,store=store)
    assert paired['pairs']==2 and paired['A']['runs']==2 and paired['B']['runs']==2
    rows=store.query_runs(); assert len(rows)==8 and all(row['sampled_inputs'] is not None for row in rows)
    assert len(store.list_batches())==3
    incompatible=config.model_copy(update={'physics_model_version':'future-physics'})
    batch=store.create_batch(incompatible,'MONTE_CARLO',42,1)
    sample=result['results'][0].copy(); sample.update(batch_id=batch,run_id='incompatible',physics_model_version='future-physics')
    sample.pop('last',None); store.append_run(sample)
    assert all(row['run_id']!='incompatible' for row in store.query_runs())
    assert any(row['run_id']=='incompatible' for row in store.query_runs(include_incompatible=True))


def test_parameter_samples_are_retained(tmp_path):
    store=ExperimentStore(tmp_path/'sampled.sqlite3'); config=validate_config(compact(resolve_config('SAT_SAT_STRESS')))
    distribution={'disturbance.vibration.stochastic_sigma_rad':{'distribution':'uniform','low':.0001,'high':.0003}}
    result=run_monte_carlo(config,runs=2,base_seed=7,duration_s=.1,distributions=distribution,store=store)
    values=[x['sampled_inputs']['disturbance.vibration.stochastic_sigma_rad'] for x in result['results']]
    assert len(values)==2 and values[0]!=values[1]


def test_existing_baseline_still_centres_target():
    engine=SimulationEngine()
    for _ in range(210): _,telemetry=engine.step()
    assert telemetry['measurement_valid'] and telemetry['tracking_error_px']<8
    assert telemetry['estimator']=='kalman' and telemetry['predictor']=='none'


def test_glare_background_manoeuvre_and_gimbal_bias_are_physical():
    raw=compact(resolve_config()); raw['disturbance'].update(gaussian_noise=False,blur=False)
    raw['disturbance']['glare'].update(enabled=True,intensity=80,centre_fraction=[.5,.5])
    raw['disturbance']['sensor']['background_brightness']=20
    raw['disturbance']['manoeuvre'].update(enabled=True,start_s=0,velocity_impulse_m_s=[1,0,0])
    raw['disturbance']['gimbal']['static_bias_rad']=[.02,0]
    affected=SimulationEngine(config=raw); frame,t=affected.step(annotate=False)
    clean=SimulationEngine(config=compact(resolve_config())); _,baseline=clean.step(annotate=False)
    assert frame.mean()>0 and t['true_relative_position_m']!=baseline['true_relative_position_m']
    assert t['true_gimbal_axis_rad'][0]==pytest.approx(.02)
    assert t['true_optical_axis_rad'][0]==pytest.approx(.02)


def test_visual_annotation_is_decorative_and_paired_raw_inputs_match():
    raw=compact(resolve_config('SAT_SAT_STRESS'))
    live=SimulationEngine(config=raw); headless=SimulationEngine(config=raw)
    _,visual=live.step(annotate=True); _,plain=headless.step(annotate=False)
    assert visual['opencv_pixel']==plain['opencv_pixel'] and visual['ground_truth_pixel']==plain['ground_truth_pixel']
    a=SimulationEngine(config=raw); variant=dict(raw); variant['estimator']='none'; variant['pipeline_preset']='CUSTOM'; b=SimulationEngine(config=variant)
    for _ in range(3):
        _,ta=a.step(annotate=True); _,tb=b.step(annotate=False)
        assert ta['true_relative_position_m']==tb['true_relative_position_m']
        assert ta['error_budget']['vibration_rad']==tb['error_budget']['vibration_rad']
        assert ta['beacon_intensity']==tb['beacon_intensity']
        assert ta['distractor_pixels']==tb['distractor_pixels']
