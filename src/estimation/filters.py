"""NONE, KF-CV, nonlinear angular EKF/UKF, and quality-adaptive KF-R."""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np

from src.core.contracts import Measurement, TrackingState
from src.estimation.geometry import (CameraCalibration, angles_to_pixel,
                                     angular_measurement_jacobian, pixel_to_angles,
                                     uncertainty_ellipse)


NUMERICAL_FAILURES = ("CHOLESKY_RECOVERY", "NON_POSITIVE_COVARIANCE", "SINGULAR_INNOVATION",
                      "INVALID_STATE", "INVALID_MEASUREMENT", "ANGLE_OUTSIDE_MODEL_DOMAIN", "RESET_REQUIRED")


def _transition(dt: float) -> np.ndarray:
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _cv_process_covariance(dt: float, spectral_density: float) -> np.ndarray:
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    return spectral_density * np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
                                        [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]], dtype=float)


def _symmetrize_positive(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, tuple[str, ...]]:
    covariance = (covariance + covariance.T) / 2
    events: list[str] = []
    values = np.linalg.eigvalsh(covariance)
    if not np.isfinite(values).all():
        return np.eye(covariance.shape[0]) * max(floor, 1.0), ("INVALID_STATE", "RESET_REQUIRED")
    if values.min() < floor:
        covariance += np.eye(covariance.shape[0]) * (floor - values.min())
        events.append("NON_POSITIVE_COVARIANCE")
    return covariance, tuple(events)


class BaseEstimator:
    version = "1.0.0"
    estimator_name = "none"
    representation = "image_cv"

    def __init__(self, config, streams=None):
        self.config = config
        self.settings = config.estimation
        self.calibration = CameraCalibration.from_config(config)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.missing_frames = 0
        self.numerical_recovery_count = 0
        self.measurement_rejection_count = 0

    def metadata(self):
        return dict(estimator_name=self.estimator_name, estimator_version=self.version,
                    state_model=self.representation, process_model="constant_velocity",
                    measurement_model="pixel_linear" if self.representation == "image_cv" else "pixel_from_tan_los_angles",
                    Q_config={"pixel": self.settings.process_noise_px_s2,
                              "angular": self.settings.angular_process_noise_rad_s2},
                    R_config={"pixel_variance": self.settings.measurement_noise_px2},
                    adaptive_R_config={"min_scale": self.settings.adaptive_r_min_scale,
                                       "max_scale": self.settings.adaptive_r_max_scale} if self.estimator_name == "akf_r" else None,
                    adaptive_Q_config=None,
                    gating_config={"enabled": self.settings.gate_enabled,
                                   "nis_threshold": self.settings.gate_nis_threshold},
                    initial_covariance={"position": self.settings.initial_position_variance,
                                        "velocity": self.settings.initial_velocity_variance},
                    UKF_alpha=self.settings.ukf_alpha if self.estimator_name == "ukf_angular" else None,
                    UKF_beta=self.settings.ukf_beta if self.estimator_name == "ukf_angular" else None,
                    UKF_kappa=self.settings.ukf_kappa if self.estimator_name == "ukf_angular" else None,
                    camera_calibration_version=self.calibration.version)


class NoneEstimator(BaseEstimator):
    version = "passthrough-2.0.0"
    estimator_name = "none"

    def update(self, measurement: Measurement | None, dt: float) -> TrackingState:
        started = perf_counter()
        timestamp = 0.0 if measurement is None else measurement.timestamp
        if measurement is None or not measurement.valid:
            return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                 prediction_only=False, update_latency_ms=(perf_counter() - started) * 1000)
        return TrackingState(x=float(measurement.pixel[0]), y=float(measurement.pixel[1]), timestamp=timestamp,
                             valid=True, confidence=measurement.confidence, state_vector=np.asarray(measurement.pixel),
                             predicted_measurement=measurement.pixel, measurement_used=True,
                             estimator_name=self.estimator_name,
                             image_covariance=measurement.covariance,
                             uncertainty_ellipse=uncertainty_ellipse(measurement.covariance),
                             update_latency_ms=(perf_counter() - started) * 1000)


class PixelCVKF(BaseEstimator):
    version = "linear-kf-cv-2.0.0"
    estimator_name = "kf_cv"
    representation = "image_cv"
    adaptive = False

    def _initialize(self, measurement: Measurement):
        self.x = np.array([measurement.pixel[0], measurement.pixel[1], 0, 0], dtype=float)
        self.P = np.diag([self.settings.initial_position_variance] * 2 +
                         [self.settings.initial_velocity_variance] * 2).astype(float)

    def measurement_covariance(self, measurement: Measurement) -> tuple[np.ndarray, float]:
        return np.eye(2) * self.settings.measurement_noise_px2, 1.0

    def update(self, measurement: Measurement | None, dt: float) -> TrackingState:
        started = perf_counter()
        timestamp = 0.0 if measurement is None else measurement.timestamp
        valid_measurement = measurement is not None and measurement.valid
        if self.x is None:
            if not valid_measurement:
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                     update_latency_ms=(perf_counter() - started) * 1000)
            self._initialize(measurement)
            covariance = self.P[:2, :2].copy()
            return self._result(timestamp, measurement, True, False, measurement.pixel, None, None,
                                np.eye(2) * self.settings.measurement_noise_px2, 1.0, (), started, covariance)

        if not math.isfinite(dt) or dt <= 0:
            dt = 1 / max(self.config.fps, 1)
        transition = _transition(dt)
        process_covariance = _cv_process_covariance(dt, self.settings.process_noise_px_s2)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + process_covariance
        self.P, covariance_events = _symmetrize_positive(self.P, self.settings.covariance_floor)
        predicted = tuple(float(v) for v in self.x[:2])
        if not valid_measurement:
            self.missing_frames += 1
            events = list(covariance_events)
            if self.missing_frames > self.settings.max_prediction_only_frames:
                events.append("RESET_REQUIRED")
                self.x = self.P = None
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                     prediction_only=True, numerical_events=tuple(events),
                                     update_latency_ms=(perf_counter() - started) * 1000)
            return self._result(timestamp, measurement, False, True, predicted, None, None,
                                np.eye(2) * self.settings.measurement_noise_px2, 1.0,
                                tuple(events), started, self.P[:2, :2].copy(), process_covariance)
        self.missing_frames = 0
        z = np.asarray(measurement.pixel, dtype=float)
        if not np.isfinite(z).all():
            return self._result(timestamp, measurement, False, True, predicted, None, None,
                                np.eye(2) * self.settings.measurement_noise_px2, 1.0,
                                covariance_events + ("INVALID_MEASUREMENT",), started, self.P[:2, :2].copy(), process_covariance)
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        innovation = z - observation @ self.x
        measurement_covariance, r_scale = self.measurement_covariance(measurement)
        innovation_covariance = observation @ self.P @ observation.T + measurement_covariance
        try:
            nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        except np.linalg.LinAlgError:
            return self._result(timestamp, measurement, False, True, predicted, innovation, None,
                                measurement_covariance, r_scale, covariance_events + ("SINGULAR_INNOVATION",),
                                started, self.P[:2, :2].copy(), process_covariance)
        if self.settings.gate_enabled and nis > self.settings.gate_nis_threshold:
            self.measurement_rejection_count += 1
            return self._result(timestamp, measurement, False, True, predicted, innovation,
                                innovation_covariance, measurement_covariance, r_scale, covariance_events,
                                started, self.P[:2, :2].copy(), process_covariance, nis)
        gain = np.linalg.solve(innovation_covariance, observation @ self.P).T
        self.x = self.x + gain @ innovation
        identity = np.eye(4)
        residual_operator = identity - gain @ observation
        self.P = residual_operator @ self.P @ residual_operator.T + gain @ measurement_covariance @ gain.T
        self.P, update_events = _symmetrize_positive(self.P, self.settings.covariance_floor)
        return self._result(timestamp, measurement, True, False, predicted, innovation, innovation_covariance,
                            measurement_covariance, r_scale, covariance_events + update_events,
                            started, self.P[:2, :2].copy(), process_covariance, nis)

    def _result(self, timestamp, measurement, used, prediction_only, predicted, innovation,
                innovation_covariance, current_r, r_scale, events, started, image_covariance,
                current_q=None, nis=None):
        if events:
            self.numerical_recovery_count += len(events)
        return TrackingState(x=float(self.x[0]), y=float(self.x[1]), vx=float(self.x[2]), vy=float(self.x[3]),
                             timestamp=timestamp, valid=True, covariance=self.P.copy(),
                             confidence=None if measurement is None else measurement.confidence,
                             representation=self.representation, state_vector=self.x.copy(),
                             predicted_measurement=tuple(float(v) for v in predicted),
                             innovation=None if innovation is None else tuple(float(v) for v in innovation),
                             innovation_covariance=None if innovation_covariance is None else innovation_covariance.copy(),
                             measurement_used=used, prediction_only=prediction_only,
                             estimator_name=self.estimator_name, image_covariance=image_covariance,
                             uncertainty_ellipse=uncertainty_ellipse(image_covariance), nis=nis,
                             current_r=current_r.copy(), current_q=None if current_q is None else current_q.copy(),
                             r_scale=r_scale, numerical_events=tuple(events),
                             update_latency_ms=(perf_counter() - started) * 1000)


class AdaptivePixelKF(PixelCVKF):
    version = "adaptive-kf-r-1.0.0"
    estimator_name = "akf_r"
    adaptive = True

    @staticmethod
    def quality_score(measurement: Measurement) -> float:
        confidence = 0.5 if measurement.confidence is None else float(np.clip(measurement.confidence, 0, 1))
        snr = 0.0 if measurement.image_snr_estimate is None else max(0.0, measurement.image_snr_estimate)
        snr_score = snr / (snr + 5.0)
        fit_score = 1.0 if measurement.fit_error is None else 1.0 / (1.0 + max(0.0, measurement.fit_error) / 20.0)
        quality = measurement.quality
        unclipped = 0.0 if quality.get("border_clipped", False) else 1.0
        unsaturated = 1.0 - float(np.clip(quality.get("saturation_fraction", 0.0), 0, 1))
        unambiguous = 1.0 - float(np.clip(quality.get("candidate_ambiguity", 0.0), 0, 1))
        return float(np.clip(0.35 * confidence + 0.30 * snr_score + 0.15 * fit_score +
                             0.10 * unclipped + 0.05 * unsaturated + 0.05 * unambiguous, 0.05, 1.0))

    def measurement_covariance(self, measurement: Measurement) -> tuple[np.ndarray, float]:
        score = self.quality_score(measurement)
        scale = float(np.clip(1.0 / (score * score), self.settings.adaptive_r_min_scale,
                              self.settings.adaptive_r_max_scale))
        floor = np.eye(2) * self.settings.measurement_noise_px2 * self.settings.adaptive_r_min_scale
        if measurement.covariance is not None and np.shape(measurement.covariance) == (2, 2):
            covariance = np.asarray(measurement.covariance, dtype=float)
            covariance, _ = _symmetrize_positive(covariance, self.settings.covariance_floor)
            values, vectors = np.linalg.eigh(covariance)
            values = np.clip(values, self.settings.measurement_noise_px2 * self.settings.adaptive_r_min_scale,
                             self.settings.measurement_noise_px2 * self.settings.adaptive_r_max_scale)
            return floor + vectors @ np.diag(values) @ vectors.T, scale
        return np.eye(2) * self.settings.measurement_noise_px2 * scale, scale


class AngularEstimator(BaseEstimator):
    representation = "los_angular_cv"

    def _initialize(self, measurement: Measurement):
        angle = pixel_to_angles(measurement.pixel, self.calibration)
        scale_x = self.calibration.fx / math.cos(angle[0]) ** 2
        scale_y = self.calibration.fy / math.cos(angle[1]) ** 2
        self.x = np.array([angle[0], angle[1], 0, 0], dtype=float)
        self.P = np.diag([self.settings.initial_position_variance / scale_x ** 2,
                          self.settings.initial_position_variance / scale_y ** 2,
                          self.settings.initial_velocity_variance / scale_x ** 2,
                          self.settings.initial_velocity_variance / scale_y ** 2])

    def _render_result(self, timestamp, measurement, used, prediction_only, predicted_pixel,
                       innovation, innovation_covariance, current_r, events, started, current_q, nis):
        try:
            pixel = angles_to_pixel(self.x[:2], self.calibration)
            jacobian = angular_measurement_jacobian(self.x, self.calibration)
            image_covariance = jacobian[:, :2] @ self.P[:2, :2] @ jacobian[:, :2].T
            velocity = jacobian[:, :2] @ self.x[2:4]
        except ValueError:
            events = tuple(events) + ("ANGLE_OUTSIDE_MODEL_DOMAIN",)
            return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                 numerical_events=events, update_latency_ms=(perf_counter() - started) * 1000)
        if events:
            self.numerical_recovery_count += len(events)
        return TrackingState(x=float(pixel[0]), y=float(pixel[1]), vx=float(velocity[0]), vy=float(velocity[1]),
                             timestamp=timestamp, valid=True, covariance=self.P.copy(),
                             confidence=None if measurement is None else measurement.confidence,
                             representation=self.representation, state_vector=self.x.copy(),
                             predicted_measurement=tuple(float(v) for v in predicted_pixel),
                             innovation=None if innovation is None else tuple(float(v) for v in innovation),
                             innovation_covariance=None if innovation_covariance is None else innovation_covariance.copy(),
                             measurement_used=used, prediction_only=prediction_only,
                             estimator_name=self.estimator_name,
                             angular_position_rad=tuple(float(v) for v in self.x[:2]),
                             angular_velocity_rad_s=tuple(float(v) for v in self.x[2:4]),
                             image_covariance=image_covariance, uncertainty_ellipse=uncertainty_ellipse(image_covariance),
                             nis=nis, current_r=current_r.copy(), current_q=current_q.copy(),
                             r_scale=1.0, numerical_events=tuple(events),
                             update_latency_ms=(perf_counter() - started) * 1000)


class AngularEKF(AngularEstimator):
    version = "angular-ekf-1.0.0"
    estimator_name = "ekf_angular"

    def update(self, measurement: Measurement | None, dt: float) -> TrackingState:
        started = perf_counter(); timestamp = 0.0 if measurement is None else measurement.timestamp
        valid_measurement = measurement is not None and measurement.valid
        if self.x is None:
            if not valid_measurement:
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                     update_latency_ms=(perf_counter() - started) * 1000)
            self._initialize(measurement)
            predicted = tuple(float(v) for v in measurement.pixel)
            q = _cv_process_covariance(max(dt, 1e-6), self.settings.angular_process_noise_rad_s2)
            return self._render_result(timestamp, measurement, True, False, predicted, None, None,
                                       np.eye(2) * self.settings.measurement_noise_px2, (), started, q, None)
        dt = dt if math.isfinite(dt) and dt > 0 else 1 / self.config.fps
        transition = _transition(dt); q = _cv_process_covariance(dt, self.settings.angular_process_noise_rad_s2)
        self.x = transition @ self.x; self.x[:2] = (self.x[:2] + math.pi) % (2 * math.pi) - math.pi
        self.P = transition @ self.P @ transition.T + q
        self.P, events = _symmetrize_positive(self.P, self.settings.covariance_floor)
        try:
            predicted = angles_to_pixel(self.x[:2], self.calibration)
            jacobian = angular_measurement_jacobian(self.x, self.calibration)
        except ValueError:
            return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                 numerical_events=events + ("ANGLE_OUTSIDE_MODEL_DOMAIN",),
                                 update_latency_ms=(perf_counter() - started) * 1000)
        r = np.eye(2) * self.settings.measurement_noise_px2
        if not valid_measurement:
            self.missing_frames += 1
            if self.missing_frames > self.settings.max_prediction_only_frames:
                self.x = self.P = None
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name, prediction_only=True,
                                     numerical_events=events + ("RESET_REQUIRED",),
                                     update_latency_ms=(perf_counter() - started) * 1000)
            return self._render_result(timestamp, measurement, False, True, predicted, None, None, r, events, started, q, None)
        self.missing_frames = 0
        innovation = np.asarray(measurement.pixel) - predicted
        s = jacobian @ self.P @ jacobian.T + r
        try:
            nis = float(innovation @ np.linalg.solve(s, innovation))
            if self.settings.gate_enabled and nis > self.settings.gate_nis_threshold:
                self.measurement_rejection_count += 1
                return self._render_result(timestamp, measurement, False, True, predicted, innovation, s, r, events, started, q, nis)
            gain = np.linalg.solve(s, jacobian @ self.P).T
        except np.linalg.LinAlgError:
            return self._render_result(timestamp, measurement, False, True, predicted, innovation, None, r,
                                       events + ("SINGULAR_INNOVATION",), started, q, None)
        self.x += gain @ innovation
        identity = np.eye(4); residual = identity - gain @ jacobian
        self.P = residual @ self.P @ residual.T + gain @ r @ gain.T
        self.P, update_events = _symmetrize_positive(self.P, self.settings.covariance_floor)
        return self._render_result(timestamp, measurement, True, False, predicted, innovation, s, r,
                                   events + update_events, started, q, nis)


def sigma_points(mean: np.ndarray, covariance: np.ndarray, alpha: float, beta: float, kappa: float,
                 floor: float = 1e-12):
    n = len(mean); lam = alpha * alpha * (n + kappa) - n
    events: list[str] = []
    scaled = (n + lam) * covariance
    jitter = 0.0
    for attempt in range(7):
        try:
            root = np.linalg.cholesky(scaled + np.eye(n) * jitter)
            if attempt:
                events.append("CHOLESKY_RECOVERY")
            break
        except np.linalg.LinAlgError:
            jitter = floor if jitter == 0 else jitter * 10
    else:
        raise np.linalg.LinAlgError("UKF Cholesky decomposition failed")
    points = [mean]
    for index in range(n):
        points.extend((mean + root[:, index], mean - root[:, index]))
    wm = np.full(2 * n + 1, 1 / (2 * (n + lam))); wc = wm.copy()
    wm[0] = lam / (n + lam); wc[0] = wm[0] + (1 - alpha * alpha + beta)
    return np.asarray(points), wm, wc, tuple(events)


def weighted_sigma_statistics(points: np.ndarray, mean_weights: np.ndarray, covariance_weights: np.ndarray,
                              additive_covariance: np.ndarray):
    mean = np.sum(points * mean_weights[:, None], axis=0)
    delta = points - mean
    covariance = sum(covariance_weights[i] * np.outer(delta[i], delta[i]) for i in range(len(points)))
    return mean, covariance + additive_covariance


class AngularUKF(AngularEstimator):
    version = "angular-ukf-1.0.0"
    estimator_name = "ukf_angular"

    def update(self, measurement: Measurement | None, dt: float) -> TrackingState:
        started = perf_counter(); timestamp = 0.0 if measurement is None else measurement.timestamp
        valid_measurement = measurement is not None and measurement.valid
        dt = dt if math.isfinite(dt) and dt > 0 else 1 / self.config.fps
        q = _cv_process_covariance(dt, self.settings.angular_process_noise_rad_s2)
        r = np.eye(2) * self.settings.measurement_noise_px2
        if self.x is None:
            if not valid_measurement:
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                     update_latency_ms=(perf_counter() - started) * 1000)
            self._initialize(measurement)
            return self._render_result(timestamp, measurement, True, False, measurement.pixel, None, None, r, (), started, q, None)
        try:
            points, wm, wc, events = sigma_points(self.x, self.P, self.settings.ukf_alpha,
                                                  self.settings.ukf_beta, self.settings.ukf_kappa,
                                                  self.settings.covariance_floor)
            transition = _transition(dt)
            propagated = points @ transition.T
            predicted_state, predicted_covariance = weighted_sigma_statistics(propagated, wm, wc, q)
            predicted_covariance, positive_events = _symmetrize_positive(predicted_covariance, self.settings.covariance_floor)
            projected = np.asarray([angles_to_pixel(point[:2], self.calibration) for point in propagated])
            predicted_measurement = np.sum(projected * wm[:, None], axis=0)
            dz = projected - predicted_measurement; dx = propagated - predicted_state
            s = sum(wc[i] * np.outer(dz[i], dz[i]) for i in range(len(points))) + r
            cross = sum(wc[i] * np.outer(dx[i], dz[i]) for i in range(len(points)))
        except np.linalg.LinAlgError:
            self.numerical_recovery_count += 1
            return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                 numerical_events=("CHOLESKY_RECOVERY", "RESET_REQUIRED"),
                                 update_latency_ms=(perf_counter() - started) * 1000)
        except ValueError:
            return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name,
                                 numerical_events=("ANGLE_OUTSIDE_MODEL_DOMAIN",),
                                 update_latency_ms=(perf_counter() - started) * 1000)
        self.x, self.P = predicted_state, predicted_covariance
        events = events + positive_events
        if not valid_measurement:
            self.missing_frames += 1
            if self.missing_frames > self.settings.max_prediction_only_frames:
                self.x = self.P = None
                return TrackingState(timestamp=timestamp, estimator_name=self.estimator_name, prediction_only=True,
                                     numerical_events=events + ("RESET_REQUIRED",),
                                     update_latency_ms=(perf_counter() - started) * 1000)
            return self._render_result(timestamp, measurement, False, True, predicted_measurement, None, None,
                                       r, events, started, q, None)
        self.missing_frames = 0
        innovation = np.asarray(measurement.pixel) - predicted_measurement
        try:
            nis = float(innovation @ np.linalg.solve(s, innovation))
            if self.settings.gate_enabled and nis > self.settings.gate_nis_threshold:
                self.measurement_rejection_count += 1
                return self._render_result(timestamp, measurement, False, True, predicted_measurement,
                                           innovation, s, r, events, started, q, nis)
            gain = np.linalg.solve(s, cross.T).T
        except np.linalg.LinAlgError:
            return self._render_result(timestamp, measurement, False, True, predicted_measurement, innovation,
                                       None, r, events + ("SINGULAR_INNOVATION",), started, q, None)
        self.x = self.x + gain @ innovation
        self.P = self.P - gain @ s @ gain.T
        self.P, update_events = _symmetrize_positive(self.P, self.settings.covariance_floor)
        return self._render_result(timestamp, measurement, True, False, predicted_measurement,
                                   innovation, s, r, events + update_events, started, q, nis)
