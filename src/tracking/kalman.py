"""Constant-velocity Kalman filter for image-plane tracking."""

from __future__ import annotations

import numpy as np


class ImageKalmanFilter:
    def __init__(self, process_noise: float = 8.0, measurement_noise: float = 4.0) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.state = np.zeros(4, dtype=float)  # u, v, du/dt, dv/dt
        self.covariance = np.eye(4, dtype=float) * 100.0
        self.initialized = False

    def reset(self) -> None:
        self.state.fill(0.0)
        self.covariance = np.eye(4, dtype=float) * 100.0
        self.initialized = False

    def step(self, measurement: tuple[float, float] | None, dt_s: float) -> np.ndarray | None:
        if not self.initialized:
            if measurement is None:
                return None
            self.state[:2] = measurement
            self.initialized = True
            return self.state.copy()

        transition = np.array(
            [[1, 0, dt_s, 0], [0, 1, 0, dt_s], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        dt2 = dt_s * dt_s
        process = self.process_noise * np.array(
            [[dt2, 0, dt_s, 0], [0, dt2, 0, dt_s], [dt_s, 0, 1, 0], [0, dt_s, 0, 1]],
            dtype=float,
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

        if measurement is not None:
            observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            residual = np.asarray(measurement, dtype=float) - observation @ self.state
            residual_covariance = observation @ self.covariance @ observation.T + np.eye(2) * self.measurement_noise
            gain = self.covariance @ observation.T @ np.linalg.inv(residual_covariance)
            self.state = self.state + gain @ residual
            self.covariance = (np.eye(4) - gain @ observation) @ self.covariance
        return self.state.copy()

