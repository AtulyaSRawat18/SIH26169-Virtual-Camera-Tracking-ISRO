"""Deterministic ROI and scalar features shared by CNN training and inference."""
from __future__ import annotations

import cv2
import numpy as np


AUX_FEATURE_SCHEMA = (
    "classical_x_in_roi", "classical_y_in_roi", "spot_radius_px",
    "sigma_x_px", "sigma_y_px", "peak_intensity", "background_std",
    "image_snr", "ellipticity", "saturation_fraction", "candidate_ambiguity",
    "border_clipped",
)


def fixed_roi(frame: np.ndarray, pixel: tuple[float, float], size: int) -> tuple[np.ndarray, tuple[int, int], bool]:
    """Return a fixed production ROI centred on the classical estimate, never truth."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("Frame must be HxW or HxWx3")
    gray = np.asarray(gray, dtype=np.float32)
    if not np.isfinite(gray).all() or size < 2:
        raise ValueError("Invalid frame/ROI")
    cx, cy = float(pixel[0]), float(pixel[1])
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    x1, y1 = x0 + size, y0 + size
    h, w = gray.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    clipped = sx0 != x0 or sy0 != y0 or sx1 != x1 or sy1 != y1
    fill = float(np.median(gray)) if gray.size else 0.0
    roi = np.full((size, size), fill, dtype=np.float32)
    if sx1 > sx0 and sy1 > sy0:
        roi[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    return np.clip(roi / 255.0, 0.0, 1.0), (x0, y0), clipped


def auxiliary_features(measurement, origin: tuple[int, int], roi_size: int) -> np.ndarray:
    q = measurement.quality or {}
    x, y = measurement.pixel
    values = (
        (x - origin[0]) / roi_size,
        (y - origin[1]) / roi_size,
        measurement.spot_radius_px or 0.0,
        measurement.spot_sigma_x_px or 0.0,
        measurement.spot_sigma_y_px or 0.0,
        (measurement.intensity_peak or 0.0) / 255.0,
        (measurement.background_std or 0.0) / 64.0,
        np.log1p(max(0.0, measurement.image_snr_estimate or 0.0)),
        q.get("ellipticity") or 1.0,
        q.get("saturation_fraction") or 0.0,
        q.get("candidate_ambiguity") or 0.0,
        float(bool(q.get("border_clipped", False))),
    )
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=20.0, neginf=-20.0)


def fit_normalizer(features: np.ndarray) -> dict[str, list[float]]:
    mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    std = np.asarray(features, dtype=np.float64).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def normalize_features(features: np.ndarray, normalizer: dict) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    return (np.asarray(features, dtype=np.float32) - mean) / std
