"""Explicit, shared preprocessing for classical localizers."""
from __future__ import annotations

import cv2
import numpy as np


def grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        gray = cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Frame must be HxW or HxWx3")
    return np.asarray(gray, dtype=np.float32)


def background_statistics(gray: np.ndarray) -> tuple[float, float]:
    """Robust image-background estimate using median and MAD."""
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return 0.0, 0.0
    mean = float(np.median(finite))
    mad = float(np.median(np.abs(finite - mean)))
    std = 1.4826 * mad
    if std < 1e-6:
        lower = finite[finite <= np.percentile(finite, 75)]
        std = float(np.std(lower)) if lower.size else 0.0
    return mean, std


def preprocess(frame: np.ndarray, config) -> tuple[np.ndarray, float, float]:
    gray = grayscale(frame)
    size = int(config.vision.denoise_kernel_px)
    if size > 1:
        size = size if size % 2 else size + 1
        gray = cv2.GaussianBlur(gray, (size, size), 0)
    background_mean, background_std = background_statistics(gray)
    return gray, background_mean, background_std


def threshold_value(gray: np.ndarray, background_mean: float, background_std: float, config) -> float:
    mode = config.vision.threshold_strategy
    if mode == "fixed":
        return float(config.vision.fixed_threshold)
    if mode == "relative_peak":
        peak = float(np.nanmax(gray)) if gray.size else 0.0
        return background_mean + config.vision.relative_peak_fraction * max(0.0, peak - background_mean)
    if mode == "background_sigma":
        # The one-DN floor prevents an all-true mask for a perfectly black image.
        return background_mean + max(1.0, config.vision.background_sigma_k * background_std)
    raise ValueError(f"Unknown threshold strategy '{mode}'")


def binary_mask(gray: np.ndarray, threshold: float, config) -> np.ndarray:
    mask = np.asarray(gray > threshold, dtype=np.uint8) * 255
    size = int(config.vision.morphology_kernel_px)
    if size > 1:
        kernel = np.ones((size, size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
