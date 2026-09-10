"""Bright-region detection, deliberately separate from centre localization."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.contracts import SpotCandidate


def generate_candidates(gray: np.ndarray, mask: np.ndarray, config) -> tuple[list[SpotCandidate], np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[SpotCandidate] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < config.vision.min_component_area or area > config.vision.max_component_area:
            continue
        values = gray[labels == label]
        candidates.append(SpotCandidate((x, y, width, height), area, float(values.max()),
                                        float(values.sum()), tuple(float(v) for v in centroids[label]), label))
    return candidates, labels


def select_candidate(candidates: list[SpotCandidate], image_shape: tuple[int, ...], config,
                     predicted_pixel: tuple[float, float] | None = None,
                     previous_radius: float | None = None) -> SpotCandidate | None:
    if not candidates:
        return None
    policy = config.vision.candidate_selection
    if policy == "brightest":
        return max(candidates, key=lambda c: (c.integrated_intensity, c.peak_intensity))
    h, w = image_shape[:2]
    reference = np.asarray(predicted_pixel if predicted_pixel is not None else (w / 2, h / 2), dtype=float)
    max_distance = max(float(np.hypot(w, h)), 1.0)
    max_intensity = max(c.integrated_intensity for c in candidates) or 1.0

    def score(candidate: SpotCandidate) -> float:
        distance = float(np.linalg.norm(np.asarray(candidate.approximate_centroid) - reference)) / max_distance
        brightness = candidate.integrated_intensity / max_intensity
        radius = float(np.sqrt(candidate.area / np.pi))
        size_error = 0.0 if previous_radius is None else min(abs(radius - previous_radius) / max(previous_radius, 1.0), 1.0)
        if policy == "nearest_previous":
            return distance - 0.05 * brightness
        return 0.55 * distance + 0.20 * size_error - 0.25 * brightness

    return min(candidates, key=score)


def expanded_roi(candidate: SpotCandidate, shape: tuple[int, ...], padding: int) -> tuple[int, int, int, int]:
    x, y, width, height = candidate.bounding_box
    h, w = shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(w, x + width + padding), min(h, y + height + padding)
    return x0, y0, x1, y1
