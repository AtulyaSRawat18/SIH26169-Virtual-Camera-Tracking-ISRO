"""Clohessy-Wiltshire relative orbital dynamics in the Hill frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClohessyWiltshire:
    """Propagate [radial, along-track, cross-track] position and velocity."""

    mean_motion_rad_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z, x_dot, y_dot, z_dot = state
        n = self.mean_motion_rad_s
        return np.array(
            [
                x_dot,
                y_dot,
                z_dot,
                3.0 * n * n * x + 2.0 * n * y_dot,
                -2.0 * n * x_dot,
                -(n * n) * z,
            ],
            dtype=float,
        )

    def step(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the linearized equations with one RK4 step."""
        state = np.concatenate((self.position_m, self.velocity_m_s))
        k1 = self.derivative(state)
        k2 = self.derivative(state + 0.5 * dt_s * k1)
        k3 = self.derivative(state + 0.5 * dt_s * k2)
        k4 = self.derivative(state + dt_s * k3)
        state = state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.position_m = state[:3]
        self.velocity_m_s = state[3:]
        return self.position_m.copy(), self.velocity_m_s.copy()

