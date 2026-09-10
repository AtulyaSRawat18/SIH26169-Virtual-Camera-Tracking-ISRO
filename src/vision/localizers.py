"""Classical spot localizers sharing one truthful measurement contract."""
from __future__ import annotations

import math
from time import perf_counter

import cv2
import numpy as np

from src.core.contracts import Measurement, MeasurementFailureReason
from src.simulation.stationary_demo import detect_beacon
from src.vision.candidates import expanded_roi, generate_candidates, select_candidate
from src.vision.preprocessing import binary_mask, preprocess, threshold_value


class ClassicalLocalizer:
    version = "classical-localizer-1.0.0"
    algorithm_name = "classical"

    def __init__(self, config, streams=None):
        self.config = config
        self.previous_pixel: tuple[float, float] | None = None
        self.previous_radius: float | None = None
        self.last_debug: dict = {}

    def _invalid(self, timestamp: float, reason: MeasurementFailureReason,
                 background_mean: float | None = None, background_std: float | None = None,
                 candidate_count: int = 0) -> Measurement:
        self.last_debug.update(failure_reason=reason.value, candidate_count=candidate_count)
        return Measurement(None, timestamp, algorithm_name=self.algorithm_name,
                           background_mean=background_mean, background_std=background_std,
                           quality={"candidate_count": candidate_count}, failure_reason=reason)

    def _prepare(self, frame: np.ndarray, timestamp: float):
        if frame.size == 0 or not np.isfinite(frame).all():
            return None, self._invalid(timestamp, MeasurementFailureReason.INVALID_NUMERICS)
        gray, background_mean, background_std = preprocess(frame, self.config)
        threshold = threshold_value(gray, background_mean, background_std, self.config)
        mask = binary_mask(gray, threshold, self.config)
        candidates, labels = generate_candidates(gray, mask, self.config)
        selected = select_candidate(candidates, gray.shape, self.config, self.previous_pixel, self.previous_radius)
        self.last_debug = {"gray": gray, "mask": mask, "labels": labels, "candidates": candidates,
                           "selected": selected, "threshold": threshold}
        if selected is None:
            peak = float(gray.max()) if gray.size else 0.0
            reason = MeasurementFailureReason.LOW_SIGNAL if peak <= threshold else MeasurementFailureReason.NO_CANDIDATE
            return None, self._invalid(timestamp, reason, background_mean, background_std, len(candidates))
        x0, y0, x1, y1 = expanded_roi(selected, gray.shape, self.config.vision.roi_padding_px)
        roi = gray[y0:y1, x0:x1]
        selected_mask = labels[y0:y1, x0:x1] == selected.label
        self.last_debug.update(roi=roi, roi_bounds=(x0, y0, x1, y1), selected_mask=selected_mask)
        return (gray, background_mean, background_std, candidates, selected, roi, selected_mask, x0, y0), None

    def _measurement(self, pixel: tuple[float, float], timestamp: float, prepared,
                     weights: np.ndarray, fit_error: float | None = None,
                     covariance: np.ndarray | None = None, iterations: int | None = None) -> Measurement:
        gray, background_mean, background_std, candidates, selected, roi, selected_mask, x0, y0 = prepared
        yy, xx = np.indices(roi.shape, dtype=float)
        usable = np.asarray(weights, dtype=float)
        total = float(usable.sum())
        if total > 1e-12:
            mx = float((xx * usable).sum() / total)
            my = float((yy * usable).sum() / total)
            sigma_x = math.sqrt(max(0.0, float((((xx - mx) ** 2) * usable).sum() / total)))
            sigma_y = math.sqrt(max(0.0, float((((yy - my) ** 2) * usable).sum() / total)))
        else:
            sigma_x = sigma_y = None
        candidate_values = roi[selected_mask]
        peak = float(candidate_values.max()) if candidate_values.size else float(roi.max())
        intensity_sum = float(np.maximum(candidate_values - background_mean, 0).sum()) if candidate_values.size else 0.0
        image_snr = max(0.0, peak - background_mean) / max(background_std, 1.0)
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        gradient_energy = float(np.mean(gx * gx + gy * gy))
        area = selected.area
        radius = math.sqrt(area / math.pi)
        ellipticity = None if not sigma_x or not sigma_y else max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-9)
        saturated_fraction = float(np.mean(candidate_values >= 254)) if candidate_values.size else 0.0
        bx, by, bw, bh = selected.bounding_box
        border_clipped = bx == 0 or by == 0 or bx + bw >= gray.shape[1] or by + bh >= gray.shape[0]
        ambiguity = 1.0 / len(candidates)
        snr_score = image_snr / (image_snr + 5.0)
        shape_score = 1.0 if ellipticity is None else math.exp(-0.35 * max(0.0, ellipticity - 1.0))
        fit_score = 1.0 if fit_error is None else 1.0 / (1.0 + max(0.0, fit_error) / 20.0)
        confidence = float(np.clip(snr_score * shape_score * fit_score * math.sqrt(ambiguity)
                                   * (0.5 if border_clipped else 1.0), 0, 1))
        quality = dict(candidate_count=len(candidates), candidate_area_px=area, equivalent_radius_px=radius,
                       ellipticity=ellipticity, gradient_energy=gradient_energy,
                       saturation_fraction=saturated_fraction, border_clipped=border_clipped,
                       candidate_ambiguity=1.0 - ambiguity, threshold=float(self.last_debug["threshold"]),
                       fit_iterations=iterations, processing_latency_ms=(perf_counter() - self._started) * 1000)
        self.previous_pixel, self.previous_radius = pixel, radius
        return Measurement(tuple(float(v) for v in pixel), timestamp, confidence=confidence,
                           spot_radius_px=radius, spot_sigma_x_px=sigma_x, spot_sigma_y_px=sigma_y,
                           intensity_peak=peak, intensity_sum=intensity_sum,
                           background_mean=background_mean, background_std=background_std,
                           image_snr_estimate=image_snr, fit_error=fit_error,
                           algorithm_name=self.algorithm_name, covariance=covariance,
                           quality=quality)


class LegacyCentroid(ClassicalLocalizer):
    """Unmodified Prompt 1 contour-centroid detector, retained as reference."""
    version = "opencv-centroid-1.0.0"
    algorithm_name = "legacy_centroid"

    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement:
        self._started = perf_counter()
        pixel = detect_beacon(frame, int(self.config.camera.threshold))
        if pixel is None:
            return self._invalid(timestamp, MeasurementFailureReason.NO_CANDIDATE)
        prepared, invalid = self._prepare(frame, timestamp)
        if invalid:
            # The legacy implementation is authoritative for whether it detected.
            return Measurement(tuple(float(v) for v in pixel), timestamp, algorithm_name=self.algorithm_name)
        weights = np.asarray(self.last_debug["selected_mask"], dtype=float)
        return self._measurement(tuple(float(v) for v in pixel), timestamp, prepared, weights)


class BinaryCentroid(ClassicalLocalizer):
    version = "binary-centroid-1.0.0"
    algorithm_name = "binary_centroid"

    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement:
        self._started = perf_counter()
        prepared, invalid = self._prepare(frame, timestamp)
        if invalid:
            return invalid
        *_, selected, roi, selected_mask, x0, y0 = prepared
        yy, xx = np.nonzero(selected_mask)
        if xx.size == 0:
            return self._invalid(timestamp, MeasurementFailureReason.ZERO_WEIGHT, candidate_count=len(prepared[3]))
        pixel = (float(x0 + xx.mean()), float(y0 + yy.mean()))
        return self._measurement(pixel, timestamp, prepared, selected_mask.astype(float))


class IntensityWeightedCentroid(ClassicalLocalizer):
    version = "intensity-centroid-1.0.0"
    algorithm_name = "weighted_centroid"

    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement:
        self._started = perf_counter()
        prepared, invalid = self._prepare(frame, timestamp)
        if invalid:
            return invalid
        _, background, _, _, _, roi, selected_mask, x0, y0 = prepared
        weights = np.maximum(roi - background, 0) * selected_mask
        total = float(weights.sum())
        if total <= 1e-12:
            return self._invalid(timestamp, MeasurementFailureReason.ZERO_WEIGHT, background, prepared[2], len(prepared[3]))
        yy, xx = np.indices(roi.shape, dtype=float)
        pixel = (x0 + float((xx * weights).sum() / total), y0 + float((yy * weights).sum() / total))
        return self._measurement(pixel, timestamp, prepared, weights)


class GradientCentroid(ClassicalLocalizer):
    version = "gradient-centroid-1.0.0"
    algorithm_name = "gradient_centroid"

    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement:
        self._started = perf_counter()
        prepared, invalid = self._prepare(frame, timestamp)
        if invalid:
            return invalid
        _, background, _, _, _, roi, selected_mask, x0, y0 = prepared
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        cutoff = self.config.vision.gradient_threshold_fraction * float(gradient.max())
        # Exact rule: w=(max(I-B,0)+1)*G for G above a relative threshold.
        weights = (np.maximum(roi - background, 0) + 1.0) * gradient * (gradient >= cutoff) * selected_mask
        total = float(weights.sum())
        if total <= 1e-12:
            return self._invalid(timestamp, MeasurementFailureReason.ZERO_WEIGHT, background, prepared[2], len(prepared[3]))
        yy, xx = np.indices(roi.shape, dtype=float)
        pixel = (x0 + float((xx * weights).sum() / total), y0 + float((yy * weights).sum() / total))
        self.last_debug["gradient"] = gradient
        return self._measurement(pixel, timestamp, prepared, weights)


class GaussianFit(ClassicalLocalizer):
    """Bounded unrotated 2-D Gaussian fit using damped Gauss-Newton."""
    version = "gaussian-fit-1.0.0"
    algorithm_name = "gaussian_fit"

    @staticmethod
    def _model_and_jacobian(params, xx, yy):
        background, amplitude, x0, y0, log_sx, log_sy = params
        sx, sy = np.exp(log_sx), np.exp(log_sy)
        dx, dy = xx - x0, yy - y0
        exponent = -0.5 * ((dx / sx) ** 2 + (dy / sy) ** 2)
        exp_term = np.exp(np.clip(exponent, -80, 0))
        signal = amplitude * exp_term
        model = background + signal
        jacobian = np.column_stack((np.ones(xx.size), exp_term,
                                    signal * dx / (sx * sx), signal * dy / (sy * sy),
                                    signal * (dx / sx) ** 2, signal * (dy / sy) ** 2))
        return model, jacobian

    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement:
        self._started = perf_counter()
        prepared, invalid = self._prepare(frame, timestamp)
        if invalid:
            return invalid
        _, background, _, _, _, roi, selected_mask, x_offset, y_offset = prepared
        values = roi.astype(float)
        positive = np.maximum(values - background, 0) * selected_mask
        total = float(positive.sum())
        if total <= 1e-9 or roi.shape[0] < 3 or roi.shape[1] < 3:
            return self._invalid(timestamp, MeasurementFailureReason.LOW_SIGNAL, background, prepared[2], len(prepared[3]))
        yy2, xx2 = np.indices(roi.shape, dtype=float)
        initial_x = float((xx2 * positive).sum() / total)
        initial_y = float((yy2 * positive).sum() / total)
        sx = math.sqrt(max(0.5, float((((xx2 - initial_x) ** 2) * positive).sum() / total)))
        sy = math.sqrt(max(0.5, float((((yy2 - initial_y) ** 2) * positive).sum() / total)))
        params = np.array([background, max(1.0, values.max() - background), initial_x, initial_y,
                           math.log(sx), math.log(sy)], dtype=float)
        xx, yy, observed = xx2.ravel(), yy2.ravel(), values.ravel()
        lower = np.array([0, 0, -0.5, -0.5, math.log(self.config.vision.gaussian_min_sigma_px)] * 1 +
                         [math.log(self.config.vision.gaussian_min_sigma_px)], dtype=float)
        upper = np.array([255, 255, roi.shape[1] - 0.5, roi.shape[0] - 0.5,
                          math.log(min(self.config.vision.gaussian_max_sigma_px, max(1.0, roi.shape[1]))),
                          math.log(min(self.config.vision.gaussian_max_sigma_px, max(1.0, roi.shape[0])))], dtype=float)
        damping, previous_cost, iterations = 1e-3, math.inf, 0
        try:
            for iterations in range(1, self.config.vision.gaussian_max_iterations + 1):
                model, jacobian = self._model_and_jacobian(params, xx, yy)
                residual = observed - model
                system = jacobian.T @ jacobian + damping * np.eye(6)
                step = np.linalg.solve(system, jacobian.T @ residual)
                candidate = np.clip(params + step, lower, upper)
                new_model, _ = self._model_and_jacobian(candidate, xx, yy)
                cost = float(np.mean((observed - new_model) ** 2))
                if cost <= previous_cost:
                    params, previous_cost, damping = candidate, cost, max(1e-9, damping / 3)
                    if np.linalg.norm(step) < 1e-5:
                        break
                else:
                    damping *= 10
            model, jacobian = self._model_and_jacobian(params, xx, yy)
            residual = observed - model
            fit_error = float(np.sqrt(np.mean(residual ** 2)))
            if not np.isfinite(params).all() or params[1] <= 0:
                raise FloatingPointError("invalid fit")
            covariance = None
            try:
                parameter_covariance = np.linalg.solve(jacobian.T @ jacobian,
                                                       np.eye(6)) * max(float(np.var(residual)), 1e-9)
                covariance = parameter_covariance[np.ix_([2, 3], [2, 3])]
                if not np.isfinite(covariance).all() or np.any(np.linalg.eigvalsh(covariance) <= 0):
                    covariance = None
            except np.linalg.LinAlgError:
                covariance = None
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            return self._invalid(timestamp, MeasurementFailureReason.FIT_FAILED, background, prepared[2], len(prepared[3]))
        pixel = (x_offset + float(params[2]), y_offset + float(params[3]))
        fitted_weights = np.maximum(model.reshape(roi.shape) - params[0], 0)
        self.last_debug["gaussian_parameters"] = params
        return self._measurement(pixel, timestamp, prepared, fitted_weights, fit_error, covariance, iterations)


VISION_LOCALIZERS = {
    "centroid": LegacyCentroid,
    "binary_centroid": BinaryCentroid,
    "weighted_centroid": IntensityWeightedCentroid,
    "gradient_centroid": GradientCentroid,
    "gaussian_fit": GaussianFit,
}
