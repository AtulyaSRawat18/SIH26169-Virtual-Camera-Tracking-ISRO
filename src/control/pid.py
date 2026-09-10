"""Bounded PID and first-order pan/tilt actuator dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDAxis:
    kp: float
    ki: float
    kd: float
    output_limit: float
    integral_limit: float
    integral: float = 0.0
    previous_error: float = 0.0
    initialized: bool = False
    derivative_filter_tau_s: float = 0.04
    filtered_derivative: float = 0.0
    saturated: bool = False
    terms: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False
        self.filtered_derivative = 0.0
        self.saturated = False
        self.terms = (0.0, 0.0, 0.0)

    def step(self, error: float, dt_s: float, derivative_override: float | None = None) -> float:
        raw_derivative = (0.0 if not self.initialized else (error - self.previous_error) / dt_s) if derivative_override is None else derivative_override
        alpha = 1.0 if self.derivative_filter_tau_s <= 0 else dt_s / (self.derivative_filter_tau_s + dt_s)
        self.filtered_derivative += alpha * (raw_derivative - self.filtered_derivative)
        candidate = float(np.clip(self.integral + error * dt_s, -self.integral_limit, self.integral_limit))
        raw = self.kp * error + self.ki * candidate + self.kd * self.filtered_derivative
        self.saturated = abs(raw) > self.output_limit
        if not self.saturated or raw * error <= 0:
            self.integral = candidate
        self.previous_error = error
        self.initialized = True
        self.terms = (self.kp * error, self.ki * self.integral, self.kd * self.filtered_derivative)
        output = sum(self.terms)
        return float(np.clip(output, -self.output_limit, self.output_limit))


class PanTiltActuator:
    def __init__(self, response_time_s: float, max_rate_rad_s: float) -> None:
        self.response_time_s = response_time_s
        self.max_rate_rad_s = max_rate_rad_s
        self.pan_rad = 0.0
        self.tilt_rad = 0.0
        self.pan_rate_rad_s = 0.0
        self.tilt_rate_rad_s = 0.0

    def reset(self) -> None:
        self.pan_rad = self.tilt_rad = 0.0
        self.pan_rate_rad_s = self.tilt_rate_rad_s = 0.0

    def step(self, pan_command: float, tilt_command: float, dt_s: float) -> None:
        alpha = 1.0 - np.exp(-dt_s / max(self.response_time_s, 1e-6))
        pan_command = float(np.clip(pan_command, -self.max_rate_rad_s, self.max_rate_rad_s))
        tilt_command = float(np.clip(tilt_command, -self.max_rate_rad_s, self.max_rate_rad_s))
        self.pan_rate_rad_s += alpha * (pan_command - self.pan_rate_rad_s)
        self.tilt_rate_rad_s += alpha * (tilt_command - self.tilt_rate_rad_s)
        self.pan_rad += self.pan_rate_rad_s * dt_s
        self.tilt_rad = float(np.clip(self.tilt_rad + self.tilt_rate_rad_s * dt_s, -1.2, 1.2))

