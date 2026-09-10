"""Clohessy-Wiltshire relative orbital dynamics in the Hill frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def propagate_cw_analytical(position_m: np.ndarray, velocity_m_s: np.ndarray,
                            time_s: float, mean_motion_rad_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact state-transition solution of the linear CW equations."""
    x0,y0,z0=np.asarray(position_m,dtype=float); vx0,vy0,vz0=np.asarray(velocity_m_s,dtype=float)
    n=mean_motion_rad_s
    if n <= 0: raise ValueError("CW mean motion must be positive")
    nt=n*time_s; s=np.sin(nt); c=np.cos(nt)
    position=np.array([(4-3*c)*x0+s*vx0/n+2*(1-c)*vy0/n,
                       6*(s-nt)*x0+y0-2*(1-c)*vx0/n+(4*s-3*nt)*vy0/n,
                       c*z0+s*vz0/n],dtype=float)
    velocity=np.array([3*n*s*x0+c*vx0+2*s*vy0,
                       6*n*(c-1)*x0-2*s*vx0+(4*c-3)*vy0,
                       -n*s*z0+c*vz0],dtype=float)
    return position,velocity


CW_CASES={
    "bounded_ellipse": lambda n: ((10.,0.,0.),(0.,-20*n,0.)),
    "in_plane_ellipse": lambda n: ((15.,30.,0.),(.04,-30*n,0.)),
    "along_track_drift": lambda n: ((10.,0.,0.),(0.,-12*n,0.)),
    "cross_track_oscillation": lambda n: ((0.,0.,25.),(0.,0.,.05)),
    "formation_3d": lambda n: ((12.,20.,15.),(.03,-24*n,.04)),
}


def cw_case_initial_state(name: str, mean_motion_rad_s: float):
    if name not in CW_CASES: raise ValueError(f"Unknown CW case '{name}'")
    position,velocity=CW_CASES[name](mean_motion_rad_s)
    return np.asarray(position,dtype=float),np.asarray(velocity,dtype=float)


def bounded_motion_residual(position_m: np.ndarray, velocity_m_s: np.ndarray,
                            mean_motion_rad_s: float) -> float:
    """Residual of the no-secular-drift condition vy0 = -2 n x0."""
    return float(np.asarray(velocity_m_s)[1]+2*mean_motion_rad_s*np.asarray(position_m)[0])


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


@dataclass
class AnalyticalClohessyWiltshire:
    """Closed-form CW propagation from an immutable initial condition."""
    mean_motion_rad_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray

    def __post_init__(self):
        self.initial_position_m=np.asarray(self.position_m,dtype=float).copy()
        self.initial_velocity_m_s=np.asarray(self.velocity_m_s,dtype=float).copy()
        self.time_s=0.0

    def step(self,dt_s: float):
        self.time_s+=dt_s
        self.position_m,self.velocity_m_s=propagate_cw_analytical(
            self.initial_position_m,self.initial_velocity_m_s,self.time_s,self.mean_motion_rad_s)
        return self.position_m.copy(),self.velocity_m_s.copy()

