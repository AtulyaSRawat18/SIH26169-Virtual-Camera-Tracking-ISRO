"""Shared live/headless experiment orchestrator."""
from __future__ import annotations
import math
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from src.control.gimbal import GimbalPlant
from src.core.camera import VirtualCamera
from src.core.contracts import (ActuatorLimits, ControllerInput, PredictionInput, SearchInput,
                                TrackingState)
from src.core.randomness import RandomStreams
from src.core.registry import Stage,validate_config,build_stages,algorithm_versions,algorithm_metadata
from src.core.metrics import Metrics,PATState,FailureCause
from src.core.scenarios import resolve_config
from src.physics.cw import bounded_motion_residual
from src.optics.link import GaussianOpticalLink

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs'/'integrated.json'


class SimulationEngine:
    def __init__(self,config_path=DEFAULT_CONFIG,config=None,retain_frames=1200):
        self.lock=threading.RLock(); self.stop_event=threading.Event(); self.thread=None
        self.retain_frames=retain_frames
        self.config=validate_config(config) if config is not None else resolve_config()
        self.reset()

    def reset(self,config=None,change_event=None):
        candidate=validate_config(config if config is not None else self.config)
        streams=RandomStreams(candidate.seed,candidate.run_index)
        stages=build_stages(candidate,streams)
        camera=VirtualCamera(candidate.camera,streams)
        gimbal=GimbalPlant(candidate.pid,candidate.disturbance.gimbal,candidate.fps)
        if candidate.lock.start_locked:
            motion=stages[Stage.MOTION]
            if hasattr(motion,"model"):
                hill=motion.model.position_m
                initial_world=np.array([hill[1],hill[2],candidate.orbit.nominal_camera_range_m+hill[0]],dtype=float)
            else:
                initial_world=np.asarray(motion.p,dtype=float)
            gimbal.initialize_attitude(math.atan2(initial_world[0],initial_world[2]),
                                       math.atan2(initial_world[1],math.hypot(initial_world[0],initial_world[2])))
        optical_link=GaussianOpticalLink(candidate.optical_link)
        from src.core.metrics import PATStateMachine
        with self.lock:
            self.config,self.streams,self.stages,self.camera,self.gimbal,self.optical_link=candidate,streams,stages,camera,gimbal,optical_link
            self.metrics,self.state_machine=Metrics(self.retain_frames),PATStateMachine(candidate.lock)
            self.last_known=TrackingState(); self.loss_time=0.; self.trajectory=deque(maxlen=300); self.frame_number=0
            self.state_entered_s=0.; self.pat_events=deque(maxlen=100)
            if candidate.lock.start_locked:
                self.pat_events.append({"timestamp_s":0.,"event":"LOCKED","reason":"demo starts with coarse alignment established"})
            self.state_history=deque(maxlen=candidate.temporal.history_frames)
            self.measurement_quality_history=deque(maxlen=candidate.temporal.history_frames)
            self.pending_predictions=deque()
            self.latest_frame=np.zeros((candidate.camera.height_px,candidate.camera.width_px,3),dtype=np.uint8)
            self.latest_raw_frame=self.latest_frame.copy(); self.last_measurement=None; self.last_classical_measurement=None
            self.latest_telemetry={}; self.config_change_events=[] if change_event is None else [change_event]
            if change_event is not None:
                self.pat_events.append({"timestamp_s":0.,"event":"CONFIG_CHANGED","reason":f"{change_event['path']}: {change_event['before']} -> {change_event['after']}"})

    @property
    def actuator(self): return self.gimbal.actuator

    def config_snapshot(self):
        with self.lock: return self.config.model_dump(mode='json')

    def apply_config_change(self,path,value):
        """Validate and restart from a timestamped user-controlled variable change."""
        with self.lock:
            data=self.config.model_dump(mode='json'); target=data; parts=path.split('.')
            for part in parts[:-1]: target=target[part]
            if parts[-1] not in target: raise ValueError(f"Unknown configuration path '{path}'")
            previous=target[parts[-1]]; target[parts[-1]]=value
            event={"timestamp_s":self.frame_number/self.config.fps,"path":path,"before":previous,"after":value,
                   "policy":"validated_restart_with_same_master_seed"}
            self.reset(validate_config(data),event)
            return event

    def step(self,publish=True,annotate=True):
        with self.lock: return self._step(publish,annotate)

    def _world_state(self,state):
        if state.frame=='world': return state.position_m,state.velocity_m_s
        p,v=state.position_m,state.velocity_m_s
        return np.array([p[1],p[2],self.config.orbit.nominal_camera_range_m+p[0]]),np.array([v[1],v[2],v[0]])

    def _failure(self,scene,measurement,truth,error,incorrect):
        c=self.config; e=self.stages[Stage.DISTURBANCE].current
        out=truth is None or not (0<=truth[0]<c.camera.width_px and 0<=truth[1]<c.camera.height_px)
        vibration=np.linalg.norm(e.get('vibration_rad',[0,0]))
        if incorrect: return FailureCause.FALSE_TARGET.value
        if scene.beacon_dropout or scene.frame_dropout: return FailureCause.DROPOUT.value
        if out: return FailureCause.OUT_OF_FOV.value
        if scene.beacon_intensity<c.camera.threshold: return FailureCause.BEACON_TOO_WEAK.value
        if self.gimbal.position_saturated: return FailureCause.GIMBAL_POSITION_SATURATION.value
        if self.gimbal.rate_saturated and error is not None and error>c.lock.loss_error_threshold_px: return FailureCause.GIMBAL_RATE_SATURATION.value
        if vibration>.0015 and not measurement.valid: return FailureCause.EXCESSIVE_VIBRATION.value
        if error is not None and error>c.lock.loss_error_threshold_px: return FailureCause.EXCESSIVE_POINTING_ERROR.value
        if not measurement.valid: return FailureCause.MEASUREMENT_FAILURE.value
        return None

    def _step(self,publish,annotate):
        started=time.perf_counter(); c=self.config; dt=1/c.fps; physics_dt=dt*c.time_scale
        timestamp=(self.frame_number+1)*dt; simulation_time=(self.frame_number+1)*physics_dt
        nominal=self.stages[Stage.MOTION].update(simulation_time,physics_dt)
        truth_state=self.stages[Stage.DISTURBANCE].manoeuvre(nominal,timestamp)
        reported=self.stages[Stage.DISTURBANCE].reported_navigation(truth_state)
        world,world_velocity=self._world_state(truth_state); reported_world,_=self._world_state(reported)
        range_squared=float(world@world)
        los_rate=float(np.linalg.norm(np.cross(world,world_velocity))/range_squared) if range_squared>0 else 0.
        self.trajectory.append(world.tolist())
        capture=self.gimbal.reported_state(); physical_gimbal=self.gimbal.physical_state()
        optical_axis=self.stages[Stage.DISTURBANCE].camera_orientation(physical_gimbal,timestamp,dt)
        nominal_projection=self.camera.project(world,optical_axis)
        scene=self.stages[Stage.DISTURBANCE].scene(nominal_projection,timestamp)
        frame=self.stages[Stage.DISTURBANCE].image(self.camera.render(scene),timestamp,scene)
        raw_frame=frame.copy()
        classical_measurement=self.stages[Stage.VISION].measure(raw_frame,timestamp)
        measurement=self.stages[Stage.CORRECTION].correct(raw_frame,classical_measurement,
                                                          {"timestamp":timestamp,"scenario":c.scenario.value})
        self.last_classical_measurement=classical_measurement
        self.last_measurement=measurement
        estimated=self.stages[Stage.ESTIMATOR].update(measurement,dt)
        self.state_history.append(estimated)
        self.measurement_quality_history.append({**measurement.quality,
                                                 "measurement_valid":measurement.valid,
                                                 "image_snr_estimate":measurement.image_snr_estimate})
        horizon=c.prediction_horizon_s
        if c.predictor!='none' and horizon<=0: horizon=c.latency.total_effective_latency_s
        prediction_request=PredictionInput(estimated,tuple(self.state_history),tuple(self.measurement_quality_history),
                                           tuple(state.timestamp for state in self.state_history),horizon,
                                           {"gimbal_pan":capture.pan,"gimbal_tilt":capture.tilt,
                                            "focal_length_px":c.camera.focal_length_px})
        predicted=self.stages[Stage.PREDICTOR].predict(prediction_request)
        centre=(c.camera.width_px/2,c.camera.height_px/2)
        controller_request=ControllerInput(
            current_state_estimate=estimated,
            predicted_state=predicted,
            target_pixel=centre,
            current_gimbal_state=capture,
            actuator_limits=ActuatorLimits(
                max_rate_rad_s=c.pid.max_rate_rad_s,
                max_acceleration_rad_s2=c.disturbance.gimbal.max_acceleration_rad_s2,
                pan_position_limit_rad=c.disturbance.gimbal.position_limit_rad[0],
                tilt_position_limit_rad=c.disturbance.gimbal.position_limit_rad[1]),
            focal_length_px=c.camera.focal_length_px,
            dt=dt,
            timestamp=timestamp,
            prediction_uncertainty=predicted.covariance,
            estimator_covariance=estimated.covariance,
            context={"pat_state":self.state_machine.state.value})
        controller_started=time.perf_counter()
        command=self.stages[Stage.CONTROLLER].compute(controller_request)
        controller_wall_latency_ms=(time.perf_counter()-controller_started)*1000
        controller_latency_ms=float(getattr(command,'diagnostics',{}).get('solver_latency_ms',0.0))
        controller_deadline_ms=c.controller_lab.control_deadline_s*1000
        estimated_error=None if not estimated.valid else float(np.linalg.norm(np.array(estimated.pixel)-centre))
        previous=self.state_machine.state
        confidence=1.0 if measurement.confidence is None else measurement.confidence
        innovation_consistent=estimated.nis is None or estimated.nis<=c.acquisition.innovation_gate_nis
        confirmation_gate_required=previous in (PATState.SEARCH,PATState.ACQUIRE,PATState.REACQUIRE)
        confirmation_valid=(measurement.valid and confidence>=c.acquisition.minimum_confirmation_confidence and
                            (innovation_consistent or not confirmation_gate_required))
        state=self.state_machine.update(confirmation_valid,estimated_error)
        if state!=previous:
            reason="candidate confirmed" if confirmation_valid else (measurement.failure_reason.value if measurement.failure_reason else "measurement unavailable or outside gate")
            self.pat_events.append({"timestamp_s":timestamp,"event":state.value,"from":previous.value,"reason":reason})
            self.state_entered_s=timestamp
        if confirmation_valid: self.last_known,self.loss_time=estimated,0.
        else: self.loss_time+=dt
        search=None
        if state in (PATState.SEARCH,PATState.LOST,PATState.REACQUIRE) and not confirmation_valid:
            search_request=SearchInput(
                self.last_known,predicted,
                predicted.covariance if predicted.covariance is not None else estimated.covariance,
                capture,controller_request.actuator_limits,centre,c.camera.focal_length_px,
                self.loss_time,dt,timestamp,state.value)
            search=self.stages[Stage.REACQUISITION].search(search_request)
            if search is not None: command=search
        if previous==PATState.REACQUIRE and state==PATState.TRACK:
            self.stages[Stage.CONTROLLER].reset()
        applied=self.gimbal.step(command,dt)
        truth=None if scene.beacon_pixel is None else list(scene.beacon_pixel)
        localization=None if truth is None or not measurement.valid else float(np.linalg.norm(np.array(measurement.pixel)-truth))
        classical_localization=None if truth is None or not classical_measurement.valid else float(np.linalg.norm(np.array(classical_measurement.pixel)-truth))
        true_error=None if truth is None else float(np.linalg.norm(np.array(truth)-centre))
        distractor_distances=[] if not classical_measurement.valid else [np.linalg.norm(np.array(classical_measurement.pixel)-np.array(d[:2])) for d in scene.distractors]
        target_distance=math.inf if classical_localization is None else classical_localization
        incorrect=bool(distractor_distances and min(distractor_distances)<target_distance)
        false_lock=incorrect and state==PATState.LOCKED
        estimator_error=None if truth is None or not estimated.valid else float(np.linalg.norm(np.array(estimated.pixel)-truth))
        control_error_rad=(None if not estimated.valid else
                           float(np.linalg.norm((np.asarray(estimated.pixel)-np.asarray(centre))/c.camera.focal_length_px)))
        actuator_rate_error_rad_s=float(np.linalg.norm(
            np.asarray([command.pan,command.tilt])-np.asarray([self.gimbal.pan_rate_rad_s,self.gimbal.tilt_rate_rad_s])))
        lead_prediction_error=None
        while self.pending_predictions and self.pending_predictions[0][0] <= timestamp + dt/2:
            _, pending_pixel = self.pending_predictions.popleft()
            if truth is not None: lead_prediction_error=float(np.linalg.norm(np.asarray(pending_pixel)-np.asarray(truth)))
        if c.predictor!='none' and predicted.valid and horizon>0:
            self.pending_predictions.append((timestamp+horizon,predicted.pixel))
        prediction_error=lead_prediction_error
        nees=None
        if truth is not None and estimated.valid and estimated.image_covariance is not None:
            difference=np.asarray(estimated.pixel)-np.asarray(truth)
            try: nees=float(difference@np.linalg.solve(estimated.image_covariance,difference))
            except np.linalg.LinAlgError: nees=None
        failure=self._failure(scene,measurement,truth,estimated_error,incorrect)
        noise=max(c.camera.noise_sigma,1e-6); background=self.stages[Stage.DISTURBANCE].current.get('background_brightness',0)
        snr=max(0,scene.beacon_intensity-background)/noise
        latency=(time.perf_counter()-started)*1000
        pixel=lambda p: None if p is None else [float(x) for x in p]
        current=self.stages[Stage.DISTURBANCE].current
        los_pan=math.atan2(float(world[0]),float(world[2]))
        los_tilt=math.atan2(float(world[1]),math.hypot(float(world[0]),float(world[2])))
        link=self.optical_link.evaluate(los_pan,los_tilt,optical_axis.pan,optical_axis.tilt,
                                        float(np.linalg.norm(world)),float(current.get('atmospheric_factor',1.0)))
        link_data=link.as_dict()
        record=dict(frame=self.frame_number,dt_s=dt,timestamp=timestamp,simulation_time_s=simulation_time,
                    ground_truth_pixel=truth,opencv_pixel=pixel(classical_measurement.pixel),corrected_pixel=pixel(measurement.pixel),estimated_pixel=pixel(estimated.pixel),
                    predicted_pixel=pixel(predicted.pixel) if c.predictor!='none' else None,camera_centre_pixel=list(centre),
                    tracking_error_px=estimated_error,true_tracking_error_px=true_error,detection_error_px=localization,
                    classical_detection_error_px=classical_localization,estimator_error_px=estimator_error,
                    prediction_error_px=prediction_error,lead_prediction_error_px=lead_prediction_error,
                    control_error_rad=control_error_rad,actuator_rate_error_rad_s=actuator_rate_error_rad_s,
                    camera_pan_rad=self.gimbal.pan_rad,camera_tilt_rad=self.gimbal.tilt_rad,
                    capture_pan_rad=capture.pan,capture_tilt_rad=capture.tilt,
                    true_gimbal_axis_rad=[physical_gimbal.pan,physical_gimbal.tilt],true_optical_axis_rad=[optical_axis.pan,optical_axis.tilt],
                    camera_pan_rate_rad_s=self.gimbal.pan_rate_rad_s,camera_tilt_rate_rad_s=self.gimbal.tilt_rate_rad_s,
                    measurement_valid=measurement.valid,previous_lock_state=previous.value,lock_state=state.value,locked=state==PATState.LOCKED,
                    confirmation_valid=confirmation_valid,innovation_consistent=innovation_consistent,
                    controller_output=[command.pan,command.tilt],applied_controller_output=[applied.pan,applied.tilt],
                    gimbal_rate_saturated=self.gimbal.rate_saturated,gimbal_position_saturated=self.gimbal.position_saturated,
                    gimbal_acceleration_saturated=self.gimbal.acceleration_saturated,
                    beacon_intensity=scene.beacon_intensity,snr_proxy=snr,image_snr_estimate=measurement.image_snr_estimate,
                    measurement_confidence=measurement.confidence,measurement_failure_reason=None if measurement.failure_reason is None else measurement.failure_reason.value,
                    measurement_quality=measurement.quality,measurement_covariance=None if measurement.covariance is None else measurement.covariance.tolist(),
                    vision_algorithm=measurement.algorithm_name,vision_latency_ms=measurement.quality.get('processing_latency_ms'),
                    correction_algorithm=measurement.correction_algorithm,correction_vector_px=pixel(measurement.correction),
                    correction_model_id=measurement.correction_model_id,correction_model_version=measurement.correction_model_version,
                    correction_latency_ms=measurement.correction_latency_ms,correction_fallback=measurement.correction_fallback,
                    correction_failure_reason=None if measurement.correction_failure_reason is None else measurement.correction_failure_reason.value,
                    estimator_nis=estimated.nis,estimator_nees_position=nees,
                    estimator_covariance_trace=None if estimated.covariance is None else float(np.trace(estimated.covariance)),
                    estimator_prediction_only=estimated.prediction_only,estimator_measurement_used=estimated.measurement_used,
                    estimator_innovation=None if estimated.innovation is None else list(estimated.innovation),
                    estimator_innovation_covariance=None if estimated.innovation_covariance is None else estimated.innovation_covariance.tolist(),
                    estimator_image_covariance=None if estimated.image_covariance is None else estimated.image_covariance.tolist(),
                    estimator_uncertainty_ellipse=estimated.uncertainty_ellipse,
                    estimator_r_scale=estimated.r_scale,current_R=None if estimated.current_r is None else estimated.current_r.tolist(),
                    current_Q=None if estimated.current_q is None else estimated.current_q.tolist(),
                    estimator_latency_ms=estimated.update_latency_ms,numerical_failure_events=list(estimated.numerical_events),
                    predictor_name=predicted.predictor_name,predictor_version=predicted.predictor_version,
                    predictor_model_id=predicted.model_id,prediction_horizon_s=horizon,
                    prediction_uncertainty=None if predicted.covariance is None else predicted.covariance.tolist(),
                    prediction_uncertainty_weight=predicted.uncertainty_weight,
                    predictor_latency_ms=predicted.inference_latency_ms,predictor_fallback=predicted.fallback_used,
                    predictor_failure_reason=predicted.failure_reason,
                    beacon_dropout=scene.beacon_dropout,frame_dropout=scene.frame_dropout,
                    distractor_pixels=[list(d) for d in scene.distractors],incorrect_target_selection=incorrect,false_lock=false_lock,
                    controller_name=c.controller,controller_version=getattr(self.stages[Stage.CONTROLLER],'version','unknown'),
                    controller_latency_ms=controller_latency_ms,
                    controller_deadline_ms=controller_deadline_ms,
                    controller_deadline_missed=controller_wall_latency_ms>controller_deadline_ms,
                    controller_fallback=getattr(command,'fallback_used',False),
                    controller_diagnostics=getattr(command,'diagnostics',{}),
                    controller_command_saturated=getattr(command,'saturated',False),
                    search_active=search is not None,search_strategy=c.reacquisition,
                    search_mode=None if search is None else search.phase,
                    search_fallback=False if search is None else search.fallback_used,
                    search_diagnostics={} if search is None else search.diagnostics,
                    controller_feedback=None if not hasattr(self.stages[Stage.CONTROLLER],'last_feedback') else [self.stages[Stage.CONTROLLER].last_feedback.pan,self.stages[Stage.CONTROLLER].last_feedback.tilt],
                    controller_feedforward=None if not hasattr(self.stages[Stage.CONTROLLER],'last_feedforward') else [self.stages[Stage.CONTROLLER].last_feedforward.pan,self.stages[Stage.CONTROLLER].last_feedforward.tilt],
                    feedforward_uncertainty_weight=getattr(self.stages[Stage.CONTROLLER],'last_uncertainty_weight',None),
                    optical_link=link_data,optical_link_enabled=link.enabled,
                    optical_model_version=link.model_version,optical_range_m=link.range_m,
                    pointing_error_x_rad=link.pointing_error_x_rad,pointing_error_y_rad=link.pointing_error_y_rad,
                    pointing_error_rad=link.pointing_error_rad,pointing_factor=link.pointing_factor,
                    pointing_loss_db=link.pointing_loss_db,geometric_capture_factor=link.geometric_capture_factor,
                    atmospheric_transmission=link.atmospheric_transmission,optical_efficiency_factor=link.optical_efficiency_factor,
                    received_power_w=link.received_power_w,received_power_dbm=link.received_power_dbm,
                    link_margin_db=link.link_margin_db,link_available=link.link_available,link_state=link.link_state,
                    communication_snr_db=link.communication_snr_db,communication_snr_status=link.communication_snr_status,
                    link_failure_cause=link.failure_cause,pat_link_combination=(
                        'PAT_REACQUIRING' if state in (PATState.LOST,PATState.REACQUIRE,PATState.SEARCH) else
                        f"PAT_{'LOCKED' if state==PATState.LOCKED else 'NOT_LOCKED'}_LINK_{'AVAILABLE' if link.link_available else 'UNAVAILABLE'}"),
                    failure_cause=failure,processing_latency_ms=latency,
                    error_budget=dict(navigation_error_m=current.get('navigation_error_m'),attitude_error_rad=current.get('attitude_error_rad'),
                                      vibration_rad=current.get('vibration_rad'),boresight_error_rad=current.get('boresight_error_rad'),
                                      beam_wander_rad=current.get('beam_wander_rad'),gimbal_static_bias_rad=list(c.disturbance.gimbal.static_bias_rad),
                                      classical_measurement_error_px=classical_localization,measurement_error_px=localization,
                                      correction_effect_px=None if localization is None or classical_localization is None else localization-classical_localization,
                                      estimator_error_px=estimator_error,prediction_error_px=prediction_error,
                                      control_error_rad=control_error_rad,actuator_rate_error_rad_s=actuator_rate_error_rad_s,
                                      final_tracking_residual_px=true_error,
                                      final_pointing_error_rad=link.pointing_error_rad,
                                      pointing_loss_db=link.pointing_loss_db,
                                      received_power_dbm=link.received_power_dbm,
                                      link_margin_db=link.link_margin_db))
        self.metrics.update(record)
        fov=[math.degrees(2*math.atan(size/(2*c.camera.focal_length_px))) for size in (c.camera.width_px,c.camera.height_px)]
        telemetry=dict(record,target_world_m=world.tolist(),true_relative_position_m=truth_state.position_m.tolist(),
                       true_relative_velocity_m_s=truth_state.velocity_m_s.tolist(),reported_relative_position_m=reported.position_m.tolist(),
                       reported_target_world_m=reported_world.tolist(),target_hill_position_m=truth_state.position_m.tolist(),
                       target_hill_velocity_m_s=truth_state.velocity_m_s.tolist(),kalman_pixel=pixel(estimated.pixel) if c.estimator=='kalman' else None,
                       los_rate_rad_s=los_rate,
                       fov_deg=fov,camera_size_px=[c.camera.width_px,c.camera.height_px],trajectory_m=list(self.trajectory),metrics=self.metrics.summary(),estimator=c.estimator,predictor=c.predictor,
                       estimator_state=None if estimated.state_vector is None else estimated.state_vector.tolist(),
                       estimator_representation=estimated.representation,
                       scenario=c.scenario.value,scenario_preset=c.scenario_preset,motion_model=c.motion_model,
                       cw_case=c.cw_case,
                       cw_bounded_residual_m_s=(bounded_motion_residual(np.asarray(c.orbit.initial_hill_position_m),
                                                  np.asarray(c.orbit.initial_hill_velocity_m_s),
                                                  math.sqrt(c.orbit.earth_mu_m3_s2/(c.orbit.earth_radius_m+c.orbit.altitude_m)**3))
                                                  if c.motion_model.startswith('cw') else None),
                       time_scale=c.time_scale,seed=c.seed,run_index=c.run_index,config_change_events=list(self.config_change_events),
                       pat_events=list(self.pat_events),time_in_pat_state_s=max(0.,timestamp-self.state_entered_s),
                       error_waterfall=__import__('src.experiments.evidence',fromlist=['waterfall_from_record']).waterfall_from_record(record),
                       resolved_error_config={path:self._config_value(path) for path in __import__('src.experiments.evidence',fromlist=['relevant_variables']).relevant_variables(c.scenario.value)},
                       subsystem_seeds=self.streams.seeds,point_ahead_angle_rad=list(c.disturbance.point_ahead.angle_rad),
                       point_ahead_estimate_rad=list(c.disturbance.point_ahead.estimate_rad),point_ahead_error_rad=list(c.point_ahead_error_rad))
        if annotate: self.camera.annotate(frame,measurement,estimated)
        self.frame_number+=1
        if publish: self.latest_frame,self.latest_raw_frame,self.latest_telemetry=frame,raw_frame,telemetry
        return frame,telemetry

    def _config_value(self,path):
        value=self.config.model_dump(mode='json')
        for part in path.split('.'): value=value[part]
        return value

    def run(self):
        while not self.stop_event.is_set():
            started=time.perf_counter(); self.step()
            self.stop_event.wait(max(0,1/self.config.fps-(time.perf_counter()-started)))
    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear(); self.thread=threading.Thread(target=self.run,name='simulation-engine',daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
    def snapshot(self): return self.jpeg_snapshot(),self.telemetry_snapshot()
    def telemetry_snapshot(self):
        with self.lock: return dict(self.latest_telemetry)
    def metrics_snapshot(self):
        with self.lock: return dict(summary=self.metrics.summary(),frames=list(self.metrics.frames))
    def jpeg_snapshot(self):
        with self.lock: frame=self.latest_frame
        encoded,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
        if not encoded: raise RuntimeError('Could not encode camera frame')
        return jpeg.tobytes()
    def raw_jpeg_snapshot(self):
        with self.lock: frame=self.latest_raw_frame.copy()
        encoded,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
        if not encoded: raise RuntimeError('Could not encode raw camera frame')
        return jpeg.tobytes()
    def raw_snapshot(self):
        with self.lock:
            return self.latest_raw_frame.copy(),dict(self.latest_telemetry),self.last_measurement
    def metadata(self):
        return dict(simulation_schema_version=self.config.simulation_schema_version,physics_model_version=self.config.physics_model_version,
                    scenario_version=self.config.scenario_version,algorithm_versions=algorithm_versions(self.config),subsystem_seeds=self.streams.seeds,
                    optical_model=self.optical_link.metadata())
    def algorithm_metadata(self):
        return algorithm_metadata(self.config,self.stages)
