"""PAT state machine, failure evidence and run-level statistics."""
from collections import Counter, deque
from enum import Enum
import math
import numpy as np


class PATState(str, Enum):
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    LOCKED = "LOCKED"
    LOST = "LOST"
    REACQUIRE = "REACQUIRE"


class FailureCause(str, Enum):
    BEACON_TOO_WEAK="BEACON_TOO_WEAK"; OUT_OF_FOV="OUT_OF_FOV"; DROPOUT="DROPOUT"
    FALSE_TARGET="FALSE_TARGET"; EXCESSIVE_VIBRATION="EXCESSIVE_VIBRATION"
    GIMBAL_RATE_SATURATION="GIMBAL_RATE_SATURATION"; GIMBAL_POSITION_SATURATION="GIMBAL_POSITION_SATURATION"
    MEASUREMENT_FAILURE="MEASUREMENT_FAILURE"; EXCESSIVE_POINTING_ERROR="EXCESSIVE_POINTING_ERROR"; UNKNOWN="UNKNOWN"


class PATStateMachine:
    def __init__(self, config):
        self.c=config
        self.state=PATState.LOCKED if config.start_locked else PATState.SEARCH
        self.previous=self.state
        self.stable=self.lock_streak=self.missing=self.reacquire_stable=0

    def update(self, valid, error):
        self.previous=self.state
        excessive=error is not None and error>self.c.loss_error_threshold_px
        if self.state==PATState.SEARCH:
            if valid: self.state,self.stable=PATState.ACQUIRE,1
        elif self.state==PATState.ACQUIRE:
            if not valid: self.state,self.stable=PATState.SEARCH,0
            else:
                self.stable+=1
                if self.stable>=self.c.acquisition_required_frames: self.state,self.lock_streak=PATState.TRACK,0
        elif self.state==PATState.TRACK:
            if not valid or excessive: self.state,self.missing=PATState.LOST,int(not valid)
            else:
                self.lock_streak=self.lock_streak+1 if error is not None and error<self.c.lock_error_threshold_px else 0
                if self.lock_streak>=self.c.lock_required_frames: self.state=PATState.LOCKED
        elif self.state==PATState.LOCKED:
            self.missing=self.missing+1 if not valid else 0
            if excessive or self.missing>=self.c.max_missing_frames:
                self.state=PATState.LOST
        elif self.state==PATState.LOST:
            self.state,self.reacquire_stable=PATState.REACQUIRE,1 if valid else 0
        elif self.state==PATState.REACQUIRE:
            self.reacquire_stable=self.reacquire_stable+1 if valid else 0
            if self.reacquire_stable>=self.c.reacquisition_required_frames:
                self.state,self.lock_streak=PATState.TRACK,0
        return self.state


def stats(values):
    if not values: return {k:None for k in ('count','mean','median','std','min','max','p05','p95')}
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(np.mean(a)),median=float(np.median(a)),std=float(np.std(a)),
                min=float(np.min(a)),max=float(np.max(a)),p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95)))


class Metrics:
    def __init__(self, retain_frames=1200):
        self.frames=deque(maxlen=retain_frames); self.count=0
        self.tracking=[]; self.localization=[]; self.classical_localization=[]; self.estimation_errors=[]; self.prediction_errors=[]
        self.control_errors=[]; self.actuator_errors=[]; self.latencies=[]
        self.vision_latencies=[]; self.correction_latencies=[]; self.estimator_latencies=[]; self.predictor_latencies=[]
        self.controller_latencies=[]
        self.lead_errors=[]
        self.intensities=[]; self.snr=[]; self.image_snr=[]; self.nis=[]; self.nees=[]; self.covariance_trace=[]
        self.dropouts=self.false_detections=self.false_locked=self.locked=self.rate_sat=self.position_sat=0
        self.acceleration_sat=self.command_sat=self.controller_deadline_misses=self.controller_fallbacks=0
        self.search_frames=self.search_fallbacks=0; self.search_modes=Counter(); self.search_effort=0.
        self.pointing_errors=[]; self.pointing_losses=[]; self.received_power_dbm=[]; self.link_margins=[]
        self.link_samples=self.link_available_count=self.link_interruptions=0
        self.current_outage=self.longest_outage=0; self.pat_link_combinations=Counter(); self.link_failures=Counter()
        self.first_detection=self.first_lock=None; self.lock_losses=0; self.longest_lock=self.current_lock=0
        self.reacquire_events=0; self.reacquire_success=0; self.reacquire_started=None; self.reacquire_times=[]
        self.failures=Counter(); self.control_effort=0.; self.control_smoothness=0.; self.max_rate=0.
        self.previous_command=None
        self.numerical_failures=Counter(); self.measurement_rejections=0
        self.correction_fallbacks=self.predictor_fallbacks=0
        self.last_dt=0.

    def update(self,r):
        self.frames.append(r); self.count+=1; self.last_dt=r.get('dt_s',self.last_dt)
        for name,target in [('true_tracking_error_px',self.tracking),('detection_error_px',self.localization),
                            ('classical_detection_error_px',self.classical_localization),
                            ('processing_latency_ms',self.latencies),('vision_latency_ms',self.vision_latencies),
                            ('correction_latency_ms',self.correction_latencies),('estimator_latency_ms',self.estimator_latencies),
                            ('predictor_latency_ms',self.predictor_latencies),('controller_latency_ms',self.controller_latencies),
                            ('lead_prediction_error_px',self.lead_errors),
                            ('estimator_error_px',self.estimation_errors),('prediction_error_px',self.prediction_errors),
                            ('control_error_rad',self.control_errors),('actuator_rate_error_rad_s',self.actuator_errors),
                            ('beacon_intensity',self.intensities),
                            ('snr_proxy',self.snr),('image_snr_estimate',self.image_snr),
                            ('estimator_nis',self.nis),('estimator_nees_position',self.nees),
                            ('estimator_covariance_trace',self.covariance_trace)]:
            value=r.get(name)
            if value is not None and math.isfinite(value): target.append(value)
        valid=r['measurement_valid']; self.dropouts+=int(not valid)
        self.false_detections+=int(r.get('incorrect_target_selection',False)); self.false_locked+=int(r.get('false_lock',False))
        timestamp=r.get('timestamp',self.count-1)
        if valid and self.first_detection is None: self.first_detection=timestamp
        locked=r.get('lock_state')==PATState.LOCKED.value or (r.get('lock_state') is None and r.get('locked',False)); self.locked+=int(locked)
        self.current_lock=self.current_lock+1 if locked else 0; self.longest_lock=max(self.longest_lock,self.current_lock)
        previous,current=r.get('previous_lock_state'),r.get('lock_state')
        if previous==PATState.LOCKED.value and current==PATState.LOST.value: self.lock_losses+=1
        if self.first_lock is None and locked: self.first_lock=timestamp
        if previous==PATState.LOST.value and current==PATState.REACQUIRE.value:
            self.reacquire_events+=1; self.reacquire_started=timestamp
        if previous==PATState.REACQUIRE.value and current==PATState.TRACK.value and self.reacquire_started is not None:
            self.reacquire_success+=1; self.reacquire_times.append(timestamp-self.reacquire_started); self.reacquire_started=None
        self.rate_sat+=int(r.get('gimbal_rate_saturated',False)); self.position_sat+=int(r.get('gimbal_position_saturated',False))
        self.acceleration_sat+=int(r.get('gimbal_acceleration_saturated',False))
        self.command_sat+=int(r.get('controller_command_saturated',False))
        self.controller_deadline_misses+=int(r.get('controller_deadline_missed',False))
        self.controller_fallbacks+=int(r.get('controller_fallback',False))
        command=np.asarray(r.get('controller_output',[0,0]),dtype=float)
        self.control_effort+=float(command@command)*self.last_dt
        if self.previous_command is not None:
            delta=command-self.previous_command; self.control_smoothness+=float(delta@delta)
        self.previous_command=command
        self.max_rate=max(self.max_rate,abs(r.get('camera_pan_rate_rad_s',0)),abs(r.get('camera_tilt_rate_rad_s',0)))
        cause=r.get('failure_cause')
        if cause: self.failures[cause]+=1
        for event in r.get('numerical_failure_events',[]): self.numerical_failures[event]+=1
        self.measurement_rejections+=int(valid and r.get('estimator_prediction_only',False) and not r.get('estimator_measurement_used',False))
        self.correction_fallbacks+=int(r.get('correction_fallback',False))
        self.predictor_fallbacks+=int(r.get('predictor_fallback',False))
        if r.get('search_active',False):
            self.search_frames+=1; self.search_fallbacks+=int(r.get('search_fallback',False))
            if r.get('search_mode'): self.search_modes[r['search_mode']]+=1
            self.search_effort+=float(command@command)*self.last_dt
        if r.get('optical_link_enabled',False):
            self.link_samples+=1; available=bool(r.get('link_available',False))
            self.link_available_count+=int(available)
            if available:
                if self.current_outage: self.link_interruptions+=1
                self.current_outage=0
            else:
                self.current_outage+=1; self.longest_outage=max(self.longest_outage,self.current_outage)
            for key,target in (("pointing_error_rad",self.pointing_errors),("pointing_loss_db",self.pointing_losses),
                               ("received_power_dbm",self.received_power_dbm),("link_margin_db",self.link_margins)):
                value=r.get(key)
                if value is not None and math.isfinite(value): target.append(value)
            if r.get('pat_link_combination'): self.pat_link_combinations[r['pat_link_combination']]+=1
            if r.get('link_failure_cause'): self.link_failures[r['link_failure_cause']]+=1

    def summary(self):
        t,l,lat=stats(self.tracking),stats(self.localization),stats(self.latencies)
        vision_latency,correction_latency=stats(self.vision_latencies),stats(self.correction_latencies)
        estimator_latency,predictor_latency=stats(self.estimator_latencies),stats(self.predictor_latencies)
        controller_latency=stats(self.controller_latencies)
        pointing_error,pointing_loss=stats(self.pointing_errors),stats(self.pointing_losses)
        received_power,link_margin=stats(self.received_power_dbm),stats(self.link_margins)
        stage_stats={
            'measurement':stats(self.classical_localization),'cnn_or_corrected':stats(self.localization),
            'estimation':stats(self.estimation_errors),'prediction_at_horizon':stats(self.prediction_errors),
            'control_rad':stats(self.control_errors),'actuator_rate_rad_s':stats(self.actuator_errors),
            'pointing_rad':stats(self.pointing_errors)}
        for values,key in ((self.classical_localization,'measurement'),(self.localization,'cnn_or_corrected'),
                           (self.estimation_errors,'estimation'),(self.prediction_errors,'prediction_at_horizon'),
                           (self.control_errors,'control_rad'),(self.actuator_errors,'actuator_rate_rad_s'),
                           (self.pointing_errors,'pointing_rad')):
            stage_stats[key]['rmse']=math.sqrt(np.mean(np.square(values))) if values else None
        return dict(frame_count=self.count,valid_error_samples=t['count'],rmse_tracking_error_px=math.sqrt(np.mean(np.square(self.tracking))) if self.tracking else None,
                    mean_tracking_error_px=t['mean'],median_tracking_error_px=t['median'],max_tracking_error_px=t['max'],p95_tracking_error_px=t['p95'],
                    cv_localization_rmse_px=math.sqrt(np.mean(np.square(self.localization))) if self.localization else None,
                    corrected_localization_rmse_px=math.sqrt(np.mean(np.square(self.localization))) if self.localization else None,
                    classical_localization_rmse_px=math.sqrt(np.mean(np.square(self.classical_localization))) if self.classical_localization else None,
                    estimator_rmse_px=math.sqrt(np.mean(np.square(self.estimation_errors))) if self.estimation_errors else None,
                    prediction_rmse_px=math.sqrt(np.mean(np.square(self.prediction_errors))) if self.prediction_errors else None,
                    control_rmse_rad=math.sqrt(np.mean(np.square(self.control_errors))) if self.control_errors else None,
                    actuator_rate_rmse_rad_s=math.sqrt(np.mean(np.square(self.actuator_errors))) if self.actuator_errors else None,
                    lead_prediction_rmse_px=math.sqrt(np.mean(np.square(self.lead_errors))) if self.lead_errors else None,
                    missed_detections=self.dropouts,measurement_dropout_count=self.dropouts,
                    measurement_availability_percent=100*(self.count-self.dropouts)/self.count if self.count else 0,
                    false_detections=self.false_detections,false_lock_probability=self.false_locked/self.count if self.count else 0,
                    time_to_first_detection_s=self.first_detection,time_to_first_lock_s=self.first_lock,
                    acquisition_time_s=None if self.first_lock is None or self.first_detection is None else self.first_lock-self.first_detection,
                    locked_percentage=100*self.locked/self.count if self.count else 0,lock_losses=self.lock_losses,
                    longest_lock_duration_s=self.longest_lock*self.last_dt,
                    reacquisition_events=self.reacquire_events,reacquisition_success_percent=100*self.reacquire_success/self.reacquire_events if self.reacquire_events else None,
                    mean_reacquisition_time_s=float(np.mean(self.reacquire_times)) if self.reacquire_times else None,
                    p95_reacquisition_time_s=float(np.percentile(self.reacquire_times,95)) if self.reacquire_times else None,
                    mean_beacon_intensity=stats(self.intensities)['mean'],minimum_beacon_intensity=stats(self.intensities)['min'],
                    mean_snr_proxy=stats(self.snr)['mean'],mean_image_snr_estimate=stats(self.image_snr)['mean'],
                    mean_nis=stats(self.nis)['mean'],p95_nis=stats(self.nis)['p95'],mean_nees_position=stats(self.nees)['mean'],
                    p95_nees_position=stats(self.nees)['p95'],mean_covariance_trace=stats(self.covariance_trace)['mean'],
                    measurement_rejection_count=self.measurement_rejections,numerical_failure_counts=dict(self.numerical_failures),
                    numerical_recovery_count=sum(self.numerical_failures.values()),
                    rate_saturation_percent=100*self.rate_sat/self.count if self.count else 0,
                    position_saturation_percent=100*self.position_sat/self.count if self.count else 0,
                    acceleration_saturation_percent=100*self.acceleration_sat/self.count if self.count else 0,
                    controller_command_saturation_percent=100*self.command_sat/self.count if self.count else 0,
                    saturation_duration_s=(self.rate_sat+self.position_sat+self.acceleration_sat)*self.last_dt,
                    max_angular_rate_rad_s=self.max_rate,
                    control_effort_integral=self.control_effort,control_smoothness_integral=self.control_smoothness,
                    mean_control_effort=self.control_effort/max(self.count*self.last_dt,1e-12) if self.count else 0,
                    effective_fps=1000/lat['mean'] if lat['mean'] else None,
                    mean_processing_latency_ms=lat['mean'],p95_processing_latency_ms=lat['p95'],
                    mean_vision_latency_ms=vision_latency['mean'],p95_vision_latency_ms=vision_latency['p95'],
                    mean_correction_latency_ms=correction_latency['mean'],p95_correction_latency_ms=correction_latency['p95'],
                    mean_estimator_latency_ms=estimator_latency['mean'],p95_estimator_latency_ms=estimator_latency['p95'],
                    mean_predictor_latency_ms=predictor_latency['mean'],p95_predictor_latency_ms=predictor_latency['p95'],
                    mean_controller_latency_ms=controller_latency['mean'],p95_controller_latency_ms=controller_latency['p95'],
                    max_controller_latency_ms=controller_latency['max'],controller_deadline_miss_count=self.controller_deadline_misses,
                    controller_fallback_count=self.controller_fallbacks,
                    correction_fallback_count=self.correction_fallbacks,predictor_fallback_count=self.predictor_fallbacks,
                    search_active_percent=100*self.search_frames/self.count if self.count else 0,
                    search_duration_s=self.search_frames*self.last_dt,search_effort_integral=self.search_effort,
                    search_fallback_count=self.search_fallbacks,search_mode_counts=dict(self.search_modes),
                    optical_link_enabled=self.link_samples>0,optical_link_sample_count=self.link_samples,
                    pointing_rmse_rad=math.sqrt(np.mean(np.square(self.pointing_errors))) if self.pointing_errors else None,
                    mean_pointing_error_rad=pointing_error['mean'],p95_pointing_error_rad=pointing_error['p95'],
                    mean_pointing_loss_db=pointing_loss['mean'],median_pointing_loss_db=pointing_loss['median'],
                    p95_pointing_loss_db=pointing_loss['p95'],mean_received_power_dbm=received_power['mean'],
                    p05_received_power_dbm=received_power['p05'],mean_link_margin_db=link_margin['mean'],
                    link_availability_percent=100*self.link_available_count/self.link_samples if self.link_samples else None,
                    link_outage_percent=100*(self.link_samples-self.link_available_count)/self.link_samples if self.link_samples else None,
                    longest_link_outage_s=self.longest_outage*self.last_dt,link_interruption_count=self.link_interruptions,
                    pat_link_combination_counts=dict(self.pat_link_combinations),link_failure_counts=dict(self.link_failures),
                    error_stage_statistics=stage_stats,
                    failure_counts=dict(self.failures),
                    dominant_failure_cause=self.failures.most_common(1)[0][0] if self.failures else None)
