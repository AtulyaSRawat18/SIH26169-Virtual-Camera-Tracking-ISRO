"""Single source of truth for image/line-of-sight coordinate transforms."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int
    height_px: int
    version: str = "pinhole-1.0.0"

    @classmethod
    def from_config(cls, config):
        camera = config.camera
        return cls(camera.focal_length_px, camera.focal_length_px,
                   camera.width_px / 2, camera.height_px / 2,
                   camera.width_px, camera.height_px,
                   config.estimation.camera_calibration_version)

    def as_dict(self):
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                    width_px=self.width_px, height_px=self.height_px, version=self.version)


def pixel_to_angles(pixel: tuple[float, float] | np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    u, v = np.asarray(pixel, dtype=float)
    # Image y and theta_y are both positive downward in the estimator convention.
    return np.array([math.atan((u - calibration.cx) / calibration.fx),
                     math.atan((v - calibration.cy) / calibration.fy)], dtype=float)


def angles_to_pixel(angles: tuple[float, float] | np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    theta_x, theta_y = np.asarray(angles, dtype=float)
    if abs(theta_x) >= math.pi / 2 - 1e-6 or abs(theta_y) >= math.pi / 2 - 1e-6:
        raise ValueError("ANGLE_OUTSIDE_MODEL_DOMAIN")
    return np.array([calibration.cx + calibration.fx * math.tan(theta_x),
                     calibration.cy + calibration.fy * math.tan(theta_y)], dtype=float)


def angular_measurement_jacobian(state: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    theta_x, theta_y = float(state[0]), float(state[1])
    if abs(theta_x) >= math.pi / 2 - 1e-6 or abs(theta_y) >= math.pi / 2 - 1e-6:
        raise ValueError("ANGLE_OUTSIDE_MODEL_DOMAIN")
    return np.array([[calibration.fx / math.cos(theta_x) ** 2, 0, 0, 0],
                     [0, calibration.fy / math.cos(theta_y) ** 2, 0, 0]], dtype=float)


def uncertainty_ellipse(covariance: np.ndarray | None, confidence: float = 0.95):
    if covariance is None or covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return None
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if np.any(values < 0):
        return None
    chi_square = 5.991 if confidence == 0.95 else 1.0
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return dict(major_axis_px=float(2 * math.sqrt(chi_square * values[0])),
                minor_axis_px=float(2 * math.sqrt(chi_square * values[1])),
                orientation_deg=float(math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))),
                confidence_level=confidence)
