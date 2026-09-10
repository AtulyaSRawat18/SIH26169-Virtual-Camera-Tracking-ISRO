"""Configurable plant around the preserved rate-limited pan/tilt actuator."""
from collections import deque
import numpy as np
from src.control.pid import PanTiltActuator
from src.core.contracts import Command, GimbalState


class GimbalPlant:
    def __init__(self, pid, error, fps):
        self.actuator = PanTiltActuator(pid.actuator_response_time_s, pid.max_rate_rad_s)
        self.error, self.max_rate, self.dt = error, pid.max_rate_rad_s, 1/fps
        self.delay_frames = round(error.command_latency_s*fps)
        self.queue = deque([Command()]*(self.delay_frames+1), maxlen=self.delay_frames+1)
        self.previous_applied = Command()
        self.rate_saturated = self.position_saturated = self.acceleration_saturated = False

    @property
    def pan_rad(self): return self.actuator.pan_rad
    @property
    def tilt_rad(self): return self.actuator.tilt_rad
    @property
    def pan_rate_rad_s(self): return self.actuator.pan_rate_rad_s
    @property
    def tilt_rate_rad_s(self): return self.actuator.tilt_rate_rad_s

    def reported_state(self):
        return GimbalState(self.pan_rad, self.tilt_rad, self.pan_rate_rad_s, self.tilt_rate_rad_s)

    def physical_state(self):
        return GimbalState(self.pan_rad+self.error.static_bias_rad[0], self.tilt_rad+self.error.static_bias_rad[1],
                           self.pan_rate_rad_s, self.tilt_rate_rad_s)

    def initialize_attitude(self, pan_rad, tilt_rad):
        """Start a presentation run from an already established coarse alignment."""
        pan_limit,tilt_limit=self.error.position_limit_rad
        self.actuator.pan_rad=float(np.clip(pan_rad-self.error.static_bias_rad[0],-pan_limit,pan_limit))
        self.actuator.tilt_rad=float(np.clip(tilt_rad-self.error.static_bias_rad[1],-tilt_limit,tilt_limit))
        self.actuator.pan_rate_rad_s=self.actuator.tilt_rate_rad_s=0.

    def step(self, command, dt):
        raw = np.array([command.pan, command.tilt], dtype=float)
        self.rate_saturated = bool(np.any(np.abs(raw) >= self.max_rate-1e-12))
        raw[np.abs(raw) < self.error.deadband_rad_s] = 0
        q = self.error.command_quantization_rad_s
        if q:
            raw = np.round(raw/q)*q
        if self.error.max_acceleration_rad_s2:
            old = np.array([self.previous_applied.pan, self.previous_applied.tilt])
            limited = np.clip(raw, old-self.error.max_acceleration_rad_s2*dt,
                              old+self.error.max_acceleration_rad_s2*dt)
            self.acceleration_saturated = bool(np.any(np.abs(limited-raw) > 1e-12))
            raw = limited
        else:
            self.acceleration_saturated = False
        self.queue.append(Command(*raw))
        delayed = self.queue[0]
        self.previous_applied = delayed
        self.actuator.step(delayed.pan, delayed.tilt, dt)
        pan_limit, tilt_limit = self.error.position_limit_rad
        before = (self.pan_rad, self.tilt_rad)
        self.actuator.pan_rad = float(np.clip(self.pan_rad, -pan_limit, pan_limit))
        self.actuator.tilt_rad = float(np.clip(self.tilt_rad, -tilt_limit, tilt_limit))
        at_pan=abs(self.pan_rad)>=pan_limit-1e-10 and self.pan_rad*self.pan_rate_rad_s>0
        at_tilt=abs(self.tilt_rad)>=min(tilt_limit,1.2)-1e-10 and self.tilt_rad*self.tilt_rate_rad_s>0
        self.position_saturated = before != (self.pan_rad, self.tilt_rad) or at_pan or at_tilt
        return delayed
