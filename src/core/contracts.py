"""Shared stage contracts. Simulator truth never crosses an algorithm boundary."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
import numpy as np


@dataclass(frozen=True)
class PhysicalState:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    frame: str = "world"


class MeasurementFailureReason(str, Enum):
    NO_CANDIDATE = "NO_CANDIDATE"
    LOW_SIGNAL = "LOW_SIGNAL"
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    ZERO_WEIGHT = "ZERO_WEIGHT"
    FIT_FAILED = "FIT_FAILED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    SATURATED = "SATURATED"
    INVALID_NUMERICS = "INVALID_NUMERICS"
    UNKNOWN = "UNKNOWN"


class CorrectionFailureReason(str, Enum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    EXCESSIVE_CORRECTION = "EXCESSIVE_CORRECTION"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    INVALID_ROI = "INVALID_ROI"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    OOD_WARNING = "OOD_WARNING"


@dataclass(frozen=True)
class SpotCandidate:
    """Detection result supplied to localizers; contains no simulator identity."""
    bounding_box: tuple[int, int, int, int]
    area: int
    peak_intensity: float
    integrated_intensity: float
    approximate_centroid: tuple[float, float]
    label: int | None = None


@dataclass(frozen=True)
class Measurement:
    """Algorithm-independent, subpixel optical spot measurement.

    ``pixel`` remains first for compatibility with the Prompt 1/2 call sites.
    Unknown quantities are ``None`` rather than invented.
    """
    pixel: tuple[float, float] | None
    timestamp: float
    confidence: float | None = None
    spot_radius_px: float | None = None
    spot_sigma_x_px: float | None = None
    spot_sigma_y_px: float | None = None
    intensity_peak: float | None = None
    intensity_sum: float | None = None
    background_mean: float | None = None
    background_std: float | None = None
    image_snr_estimate: float | None = None
    fit_error: float | None = None
    algorithm_name: str = "unknown"
    covariance: np.ndarray | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    failure_reason: MeasurementFailureReason | None = None
    classical_pixel: tuple[float, float] | None = None
    correction: tuple[float, float] | None = None
    correction_algorithm: str = "none"
    correction_model_id: str | None = None
    correction_model_version: str | None = None
    correction_latency_ms: float | None = None
    correction_fallback: bool = False
    correction_failure_reason: CorrectionFailureReason | None = None

    @property
    def valid(self):
        return self.pixel is not None

    @property
    def x_px(self):
        return None if self.pixel is None else self.pixel[0]

    @property
    def y_px(self):
        return None if self.pixel is None else self.pixel[1]

    def as_dict(self):
        return dict(pixel=None if self.pixel is None else list(self.pixel), timestamp=self.timestamp,
                    valid=self.valid, confidence=self.confidence, spot_radius_px=self.spot_radius_px,
                    spot_sigma_x_px=self.spot_sigma_x_px, spot_sigma_y_px=self.spot_sigma_y_px,
                    intensity_peak=self.intensity_peak, intensity_sum=self.intensity_sum,
                    background_mean=self.background_mean, background_std=self.background_std,
                    image_snr_estimate=self.image_snr_estimate, fit_error=self.fit_error,
                    algorithm_name=self.algorithm_name,
                    covariance=None if self.covariance is None else self.covariance.tolist(),
                    quality=self.quality,
                    failure_reason=None if self.failure_reason is None else self.failure_reason.value,
                    classical_pixel=None if self.classical_pixel is None else list(self.classical_pixel),
                    correction=None if self.correction is None else list(self.correction),
                    correction_algorithm=self.correction_algorithm,
                    correction_model_id=self.correction_model_id,
                    correction_model_version=self.correction_model_version,
                    correction_latency_ms=self.correction_latency_ms,
                    correction_fallback=self.correction_fallback,
                    correction_failure_reason=None if self.correction_failure_reason is None else self.correction_failure_reason.value)

    @classmethod
    def from_dict(cls, value):
        reason=value.get("failure_reason")
        correction_reason=value.get("correction_failure_reason")
        return cls(None if value.get("pixel") is None else tuple(value["pixel"]), value["timestamp"],
                   confidence=value.get("confidence"), spot_radius_px=value.get("spot_radius_px"),
                   spot_sigma_x_px=value.get("spot_sigma_x_px"), spot_sigma_y_px=value.get("spot_sigma_y_px"),
                   intensity_peak=value.get("intensity_peak"), intensity_sum=value.get("intensity_sum"),
                   background_mean=value.get("background_mean"), background_std=value.get("background_std"),
                   image_snr_estimate=value.get("image_snr_estimate"), fit_error=value.get("fit_error"),
                   algorithm_name=value.get("algorithm_name", "unknown"),
                   covariance=None if value.get("covariance") is None else np.asarray(value["covariance"], dtype=float),
                   quality=dict(value.get("quality") or {}),
                   failure_reason=None if reason is None else MeasurementFailureReason(reason),
                   classical_pixel=None if value.get("classical_pixel") is None else tuple(value["classical_pixel"]),
                   correction=None if value.get("correction") is None else tuple(value["correction"]),
                   correction_algorithm=value.get("correction_algorithm", "none"),
                   correction_model_id=value.get("correction_model_id"),
                   correction_model_version=value.get("correction_model_version"),
                   correction_latency_ms=value.get("correction_latency_ms"),
                   correction_fallback=bool(value.get("correction_fallback", False)),
                   correction_failure_reason=None if correction_reason is None else CorrectionFailureReason(correction_reason))


@dataclass(frozen=True)
class TrackingState:
    x: float = 0
    y: float = 0
    vx: float = 0
    vy: float = 0
    timestamp: float = 0
    valid: bool = False
    covariance: np.ndarray | None = None
    confidence: float | None = None
    acceleration: tuple[float, float] | None = None
    representation: str = "image_cv"
    state_vector: np.ndarray | None = None
    predicted_measurement: tuple[float, float] | None = None
    innovation: tuple[float, float] | None = None
    innovation_covariance: np.ndarray | None = None
    measurement_used: bool = False
    prediction_only: bool = False
    estimator_name: str = "none"
    angular_position_rad: tuple[float, float] | None = None
    angular_velocity_rad_s: tuple[float, float] | None = None
    image_covariance: np.ndarray | None = None
    uncertainty_ellipse: dict[str, float] | None = None
    nis: float | None = None
    current_r: np.ndarray | None = None
    current_q: np.ndarray | None = None
    r_scale: float | None = None
    numerical_events: tuple[str, ...] = ()
    update_latency_ms: float | None = None

    @property
    def pixel(self):
        return (self.x, self.y) if self.valid else None


@dataclass(frozen=True)
class PredictionInput:
    """Runtime predictor input. Ground truth is deliberately absent."""
    current_state: TrackingState
    state_history: tuple[TrackingState, ...]
    measurement_quality_history: tuple[dict[str, Any], ...]
    timestamp_history: tuple[float, ...]
    prediction_horizon_s: float
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictedState:
    valid: bool
    horizon_s: float
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float | None = None
    ay: float | None = None
    covariance: np.ndarray | None = None
    predictor_name: str = "none"
    predictor_version: str = "1.0.0"
    model_id: str | None = None
    inference_latency_ms: float | None = None
    fallback_used: bool = False
    failure_reason: str | None = None
    uncertainty_weight: float = 1.0

    @property
    def pixel(self):
        return (self.x, self.y) if self.valid else None


@dataclass(frozen=True)
class GimbalState:
    pan: float
    tilt: float
    pan_rate: float = 0.0
    tilt_rate: float = 0.0


@dataclass(frozen=True)
class Command:
    pan: float = 0
    tilt: float = 0


@dataclass(frozen=True)
class ActuatorLimits:
    max_rate_rad_s: float
    max_acceleration_rad_s2: float | None
    pan_position_limit_rad: float
    tilt_position_limit_rad: float


@dataclass(frozen=True)
class ControllerInput:
    """Operational controller input. Simulation truth is deliberately absent."""
    current_state_estimate: TrackingState
    predicted_state: PredictedState
    target_pixel: tuple[float, float]
    current_gimbal_state: GimbalState
    actuator_limits: ActuatorLimits
    focal_length_px: float
    dt: float
    timestamp: float
    prediction_uncertainty: np.ndarray | None = None
    estimator_covariance: np.ndarray | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlCommand:
    pan_command: float = 0.0
    tilt_command: float = 0.0
    command_type: str = "angular_rate_rad_s"
    controller_name: str = "none"
    controller_version: str = "1.0.0"
    saturated: bool = False
    fallback_used: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def pan(self) -> float:
        return self.pan_command

    @property
    def tilt(self) -> float:
        return self.tilt_command


@dataclass(frozen=True)
class SearchInput:
    """Truth-free acquisition/reacquisition input."""
    last_known_state: TrackingState
    predicted_state: PredictedState
    uncertainty: np.ndarray | None
    current_gimbal_state: GimbalState
    actuator_limits: ActuatorLimits
    image_centre: tuple[float, float]
    focal_length_px: float
    elapsed_search_s: float
    dt: float
    timestamp: float
    pat_state: str


@dataclass(frozen=True)
class SearchCommand:
    pan_command: float = 0.0
    tilt_command: float = 0.0
    strategy_name: str = "hold"
    phase: str = "HOLD"
    fallback_used: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def pan(self) -> float:
        return self.pan_command

    @property
    def tilt(self) -> float:
        return self.tilt_command


class MotionModel(Protocol):
    def update(self, timestamp: float, dt: float) -> PhysicalState: ...


class DisturbanceModel(Protocol):
    def attitude(self, state: GimbalState, timestamp: float) -> GimbalState: ...
    def image(self, frame: np.ndarray, timestamp: float) -> np.ndarray: ...


class OpticalLinkModel(Protocol):
    """Reserved link-budget boundary; no BER implementation exists."""
    def evaluate(self, range_m: float, pointing_error_rad: float, beam_divergence_rad: float,
                 atmospheric_loss: float, transmit_power_w: float, receiver_aperture_m: float): ...


class FinePointingActuator(Protocol):
    """Reserved residual-angle actuator boundary for a future steering mirror."""
    def update(self, command: Command, dt: float) -> GimbalState: ...


class OpticalAlignmentModel(Protocol):
    """Reserved transmit/receive boresight and mirror-alignment boundary."""
    def effective_axis(self, coarse_axis: GimbalState, timestamp: float) -> GimbalState: ...


class PointAheadModel(Protocol):
    """Reserved transverse-velocity/light-time calculation boundary."""
    def estimate(self, relative_position: PhysicalState, timestamp: float) -> tuple[float, float]: ...


class VisionTracker(Protocol):
    def measure(self, frame: np.ndarray, timestamp: float) -> Measurement: ...


class MeasurementCorrector(Protocol):
    def correct(self, frame: np.ndarray, measurement: Measurement, context: dict[str, Any] | None = None) -> Measurement: ...


class StateEstimator(Protocol):
    def update(self, measurement: Measurement, dt: float) -> TrackingState: ...


class Predictor(Protocol):
    def predict(self, request: PredictionInput) -> PredictedState: ...


class Controller(Protocol):
    def compute(self, request: ControllerInput) -> ControlCommand: ...
    def reset(self) -> None: ...


class ReacquisitionStrategy(Protocol):
    def search(self, request: SearchInput) -> SearchCommand | None: ...
    def reset(self) -> None: ...
