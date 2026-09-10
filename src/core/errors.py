"""Unified, scenario-configured physical and sensor degradation system."""
from dataclasses import dataclass
import math
from collections import deque
import cv2
import numpy as np
from src.core.contracts import GimbalState, PhysicalState


@dataclass(frozen=True)
class OpticalScene:
    beacon_pixel: tuple[float, float] | None
    beacon_intensity: float
    beacon_radius_px: float
    ellipse_ratio: float
    beacon_dropout: bool
    frame_dropout: bool
    distractors: tuple[tuple[float, float, float, float], ...]
    beam_wander_rad: tuple[float, float]
    atmospheric_factor: float


class PhysicalErrorSystem:
    """One owner for errors; independent RNG streams prevent cross-subsystem drift."""
    def __init__(self, config, streams):
        self.c, self.streams = config, streams
        self.current = {}
        self._attitude_history = deque()
        self._vibration = self._prepare_vibration()
        dc, cam = config.disturbance.distractors, config.camera
        rng = streams['distractor']
        self._distractors = tuple((rng.uniform(0,cam.width_px), rng.uniform(0,cam.height_px),
                                   rng.uniform(0,2*math.pi)) for _ in range(dc.count))
        sensor = config.disturbance.sensor
        self._hot = tuple(zip(streams['camera_noise'].integers(0,cam.width_px,sensor.hot_pixel_count),
                              streams['camera_noise'].integers(0,cam.height_px,sensor.hot_pixel_count)))

    def _prepare_vibration(self):
        rng = self.streams['vibration']
        result = []
        for axis in (self.c.disturbance.vibration.pan, self.c.disturbance.vibration.tilt):
            result.append(tuple((x.amplitude_rad,x.frequency_hz,
                                 rng.uniform(0,2*math.pi) if x.phase_rad is None else x.phase_rad) for x in axis))
        return tuple(result)

    def manoeuvre(self, state, t):
        m = self.c.disturbance.manoeuvre
        if not m.enabled or t < m.start_s:
            offset, velocity = np.zeros(3), np.zeros(3)
        else:
            tau = t-m.start_s
            active = min(tau,m.duration_s)
            impulse, accel = np.array(m.velocity_impulse_m_s), np.array(m.acceleration_m_s2)
            offset = impulse*tau + .5*accel*active**2 + accel*active*max(0,tau-active)
            velocity = impulse + accel*active
        self.current['manoeuvre_position_m'] = offset.tolist()
        return PhysicalState(state.position_m+offset,state.velocity_m_s+velocity,state.frame)

    def reported_navigation(self, state):
        n = self.c.disturbance.navigation
        if not n.enabled:
            p, v = state.position_m.copy(), state.velocity_m_s.copy()
        else:
            rng = self.streams['navigation']
            p = state.position_m + np.array(n.position_bias_m) + rng.normal(0,n.position_noise_sigma_m,3)
            v = state.velocity_m_s*(1+n.model_mismatch_fraction) + rng.normal(0,n.velocity_noise_sigma_m_s,3)
        self.current['navigation_error_m'] = (p-state.position_m).tolist()
        return PhysicalState(p,v,state.frame)

    def camera_orientation(self, gimbal, t, dt):
        d, arng, vrng = self.c.disturbance, self.streams['attitude'], self.streams['vibration']
        a = d.attitude
        attitude = np.zeros(2)
        if a.enabled:
            attitude = np.array(a.bias_rad)+np.array(a.drift_rate_rad_s)*t
            attitude += arng.normal(0,a.white_noise_sigma_rad,2)
            attitude += arng.normal(0,a.angular_rate_noise_sigma_rad_s,2)*dt
        vibration = np.zeros(2)
        if d.vibration.enabled:
            for index, components in enumerate(self._vibration):
                vibration[index] = sum(amp*math.sin(2*math.pi*freq*t+phase) for amp,freq,phase in components)
            vibration += vrng.normal(0,d.vibration.stochastic_sigma_rad,2)
        bore = np.array([d.boresight.pan_offset_rad,d.boresight.tilt_offset_rad]) if d.boresight.enabled else np.zeros(2)
        total = np.array([gimbal.pan,gimbal.tilt])+attitude+vibration+bore
        self._attitude_history.append((t,total.copy()))
        while len(self._attitude_history)>2 and self._attitude_history[1][0] <= t-a.reference_latency_s:
            self._attitude_history.popleft()
        reported_axis=self._attitude_history[0][1]
        reference_valid=not (a.enabled and arng.random()<a.reference_dropout_probability)
        self.current.update(attitude_error_rad=attitude.tolist(),vibration_rad=vibration.tolist(),
                            boresight_error_rad=bore.tolist(),effective_camera_axis_rad=total.tolist(),
                            reported_attitude_axis_rad=reported_axis.tolist() if reference_valid else None,
                            attitude_reference_valid=reference_valid)
        return GimbalState(*total)

    def _airmass(self):
        a = self.c.disturbance.atmosphere
        if not a.enabled or not a.elevation_scaling:
            return 1.0
        return min(4.0,1/max(0.15,math.sin(math.radians(self.c.platform.elevation_deg))))

    @staticmethod
    def _window_factor(windows,t):
        factor = 1.0
        for window in windows:
            if window.start_s <= t < window.end_s:
                factor *= window.factor
        return factor

    def scene(self, projected, t):
        d, cam = self.c.disturbance, self.c.camera
        atmosphere, beacon = d.atmosphere, d.beacon
        air = self._airmass()
        atmospheric = (atmosphere.attenuation*atmosphere.cloud_attenuation)**air if atmosphere.enabled else 1.0
        wander_sigma = atmosphere.beam_wander_sigma_rad*math.sqrt(air)*(1+atmosphere.turbulence_strength) if atmosphere.enabled else 0
        wander = self.streams['environment'].normal(0,wander_sigma,2)
        u,v,z = projected
        pixel = None if not np.isfinite([u,v]).all() else (u+cam.focal_length_px*wander[0],v+cam.focal_length_px*wander[1])
        variation = 1+beacon.brightness_variation_fraction*math.sin(2*math.pi*beacon.flicker_frequency_hz*t)
        intensity = beacon.nominal_intensity*variation*beacon.attenuation*atmospheric*self._window_factor(beacon.attenuation_windows,t)
        scheduled_drop = self._window_factor(beacon.hard_dropout_windows,t) == 0
        random_drop = self.streams['dropout'].random() < beacon.random_dropout_probability
        frame_drop = self.streams['dropout'].random() < d.sensor.frame_dropout_probability
        dropped = scheduled_drop or random_drop
        radius = beacon.spot_radius_px or cam.beacon_radius_px
        radius *= 1+beacon.radius_variation_fraction*math.sin(2*math.pi*max(.25,beacon.flicker_frequency_hz)*t)
        distractors = []
        dc = d.distractors
        if dc.enabled and dc.appear_s <= t < dc.disappear_s:
            for x,y,angle in self._distractors:
                if dc.moving:
                    x = (x+math.cos(angle)*dc.speed_px_s*t)%cam.width_px
                    y = (y+math.sin(angle)*dc.speed_px_s*t)%cam.height_px
                distractors.append((x,y,dc.intensity,dc.radius_px))
        self.current.update(beam_wander_rad=wander.tolist(),atmospheric_factor=atmospheric,
                            beacon_intensity=float(max(0,intensity)),beacon_dropout=dropped,frame_dropout=frame_drop,
                            point_ahead_angle_rad=list(d.point_ahead.angle_rad),point_ahead_estimate_rad=list(d.point_ahead.estimate_rad),
                            point_ahead_error_rad=list(self.c.point_ahead_error_rad))
        return OpticalScene(pixel,float(max(0,intensity)),float(radius),beacon.elliptical_ratio,dropped,frame_drop,
                            tuple(distractors),(float(wander[0]),float(wander[1])),float(atmospheric))

    def background(self, frame, t):
        d, h, w = self.c.disturbance, frame.shape[0], frame.shape[1]
        level = d.sensor.background_brightness + (d.atmosphere.environment_brightness*self._airmass() if d.atmosphere.enabled else 0)
        gradient = np.linspace(0,d.sensor.background_gradient,w,dtype=np.float32)[None,:,None]
        image = frame.astype(np.float32)+level+gradient
        if d.glare.enabled:
            yy,xx = np.ogrid[:h,:w]
            cx,cy = d.glare.centre_fraction[0]*w,d.glare.centre_fraction[1]*h
            radius = max(1,d.glare.radius_fraction*min(w,h))
            amplitude = d.glare.intensity*(1+d.glare.variation_fraction*math.sin(1.7*t))
            image += amplitude*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*radius**2))[...,None]
        return np.clip(image,0,255).astype(np.uint8)

    def image(self, frame, t, scene=None):
        d = self.c.disturbance
        frame = self.background(frame,t)
        sigma = d.sensor.defocus_sigma_px + (1.2*d.atmosphere.turbulence_strength if d.atmosphere.enabled else 0)
        if d.blur:
            frame = cv2.GaussianBlur(frame,(d.sensor.blur_kernel_px,d.sensor.blur_kernel_px),sigma)
        image = frame.astype(np.float32)*d.sensor.exposure_gain
        rng = self.streams['camera_noise']
        if d.sensor.shot_noise_scale:
            image += rng.normal(0,np.sqrt(np.maximum(image,0))*d.sensor.shot_noise_scale,image.shape)
        if d.gaussian_noise:
            image += rng.normal(0,self.c.camera.noise_sigma,image.shape)
        levels = d.sensor.quantization_levels
        image = np.round(np.clip(image,0,d.sensor.saturation_level)*(levels-1)/d.sensor.saturation_level)*d.sensor.saturation_level/(levels-1)
        result = np.clip(image,0,d.sensor.saturation_level).astype(np.uint8)
        for x,y in self._hot:
            result[y,x] = d.sensor.saturation_level
        if scene is not None and scene.frame_dropout:
            result.fill(0)
        self.current['background_brightness'] = float(level if (level := d.sensor.background_brightness+(d.atmosphere.environment_brightness if d.atmosphere.enabled else 0)) else 0)
        return result
