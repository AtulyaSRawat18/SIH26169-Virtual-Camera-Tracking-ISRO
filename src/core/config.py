"""Validated, immutable runtime experiment configuration."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator

SIMULATION_SCHEMA_VERSION = "6.0.0"
PHYSICS_MODEL_VERSION = "cw-rk4-1.0.0"
DEFAULT_MASTER_SEED = 42


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Scenario(str, Enum):
    SATELLITE_SATELLITE = "satellite_satellite"
    GROUND_SATELLITE = "ground_satellite"
    UAV_GROUND = "uav_ground"
    UAV_UAV = "uav_uav"
    UAV_SATELLITE = "uav_satellite"


class Orbit(Settings):
    earth_radius_m: float = Field(gt=0)
    altitude_m: float = Field(ge=0)
    earth_mu_m3_s2: float = Field(gt=0)
    initial_hill_position_m: tuple[float, float, float]
    initial_hill_velocity_m_s: tuple[float, float, float]
    nominal_camera_range_m: float = Field(gt=0)


class KinematicMotion(Settings):
    initial_position_m: tuple[float, float, float] = (40, 15, 600)
    velocity_m_s: tuple[float, float, float] = (8, 0, -0.5)
    angular_rate_rad_s: float = Field(default=0.01, ge=0)
    lateral_amplitude_m: float = Field(default=10, ge=0)
    initial_position_sigma_m: float = Field(default=0, ge=0)
    initial_velocity_sigma_m_s: float = Field(default=0, ge=0)


class Platform(Settings):
    observer_type: str = "satellite"
    target_type: str = "satellite"
    altitude_m: float = Field(default=400000, ge=0)
    target_altitude_m: float = Field(default=400000, ge=0)
    speed_mps: float = Field(default=0, ge=0)
    angular_rate_rad_s: float = Field(default=0, ge=0)
    elevation_deg: float = Field(default=90, ge=0, le=90)
    wind_speed_mps: float = Field(default=0, ge=0)


class Camera(Settings):
    width_px: int = Field(ge=32, le=1920)
    height_px: int = Field(ge=32, le=1080)
    focal_length_px: float = Field(gt=0)
    beacon_radius_px: int = Field(gt=0, le=100)
    noise_sigma: float = Field(ge=0, le=255)
    threshold: int = Field(ge=0, le=255)


class Kalman(Settings):
    process_noise: float = Field(ge=0)
    measurement_noise: float = Field(gt=0)


class Estimation(Settings):
    """Common estimator tuning; units are explicit in the state-estimation docs."""
    process_noise_px_s2: float = Field(default=8.0, ge=0)
    angular_process_noise_rad_s2: float = Field(default=0.001, ge=0)
    measurement_noise_px2: float = Field(default=4.0, gt=0)
    initial_position_variance: float = Field(default=25.0, gt=0)
    initial_velocity_variance: float = Field(default=100.0, gt=0)
    gate_enabled: bool = False
    gate_nis_threshold: float = Field(default=9.210, gt=0)
    max_prediction_only_frames: int = Field(default=300, ge=1)
    covariance_floor: float = Field(default=1e-9, gt=0)
    ukf_alpha: float = Field(default=0.3, gt=0, le=1)
    ukf_beta: float = Field(default=2.0, ge=0)
    ukf_kappa: float = Field(default=0.0)
    adaptive_r_min_scale: float = Field(default=0.5, gt=0)
    adaptive_r_max_scale: float = Field(default=25.0, gt=0)
    camera_calibration_version: str = "pinhole-v1"

    @model_validator(mode="after")
    def valid_scales(self):
        if self.adaptive_r_max_scale < self.adaptive_r_min_scale:
            raise ValueError("adaptive R maximum scale must be >= minimum scale")
        return self


class PID(Settings):
    kp: float = Field(ge=0)
    ki: float = Field(ge=0)
    kd: float = Field(ge=0)
    max_rate_rad_s: float = Field(gt=0)
    integral_limit: float = Field(ge=0)
    actuator_response_time_s: float = Field(gt=0)
    derivative_filter_tau_s: float = Field(default=0.04, ge=0)
    derivative_source: str = "estimated_rate"

    @model_validator(mode="after")
    def valid_derivative_source(self):
        if self.derivative_source not in {"estimated_rate", "finite_difference"}:
            raise ValueError("PID derivative_source must be estimated_rate or finite_difference")
        return self


class NavigationError(Settings):
    enabled: bool = False
    initial_position_sigma_m: float = Field(default=0, ge=0)
    initial_velocity_sigma_m_s: float = Field(default=0, ge=0)
    position_bias_m: tuple[float, float, float] = (0, 0, 0)
    position_noise_sigma_m: float = Field(default=0, ge=0)
    velocity_noise_sigma_m_s: float = Field(default=0, ge=0)
    model_mismatch_fraction: float = Field(default=0, ge=0)


class AttitudeError(Settings):
    enabled: bool = False
    bias_rad: tuple[float, float] = (0, 0)
    white_noise_sigma_rad: float = Field(default=0, ge=0)
    angular_rate_noise_sigma_rad_s: float = Field(default=0, ge=0)
    drift_rate_rad_s: tuple[float, float] = (0, 0)
    reference_latency_s: float = Field(default=0, ge=0)
    reference_dropout_probability: float = Field(default=0, ge=0, le=1)


class VibrationComponent(Settings):
    amplitude_rad: float = Field(ge=0)
    frequency_hz: float = Field(ge=0)
    phase_rad: float | None = None


class VibrationError(Settings):
    enabled: bool = False
    level: str = "NONE"
    pan: tuple[VibrationComponent, ...] = ()
    tilt: tuple[VibrationComponent, ...] = ()
    stochastic_sigma_rad: float = Field(default=0, ge=0)


class GimbalError(Settings):
    static_bias_rad: tuple[float, float] = (0, 0)
    position_limit_rad: tuple[float, float] = (3.14159, 1.2)
    command_latency_s: float = Field(default=0, ge=0)
    deadband_rad_s: float = Field(default=0, ge=0)
    command_quantization_rad_s: float = Field(default=0, ge=0)
    max_acceleration_rad_s2: float | None = Field(default=None, gt=0)


class BoresightError(Settings):
    enabled: bool = False
    pan_offset_rad: float = 0
    tilt_offset_rad: float = 0


class AttenuationWindow(Settings):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    factor: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s <= self.start_s:
            raise ValueError("Attenuation window end must follow start")
        return self


class BeaconError(Settings):
    nominal_intensity: float = Field(default=255, ge=0, le=255)
    brightness_variation_fraction: float = Field(default=0, ge=0, le=1)
    flicker_frequency_hz: float = Field(default=0, ge=0)
    attenuation: float = Field(default=1, ge=0, le=1)
    attenuation_windows: tuple[AttenuationWindow, ...] = ()
    hard_dropout_windows: tuple[AttenuationWindow, ...] = ()
    random_dropout_probability: float = Field(default=0, ge=0, le=1)
    spot_radius_px: float | None = Field(default=None, gt=0)
    radius_variation_fraction: float = Field(default=0, ge=0, le=0.9)
    elliptical_ratio: float = Field(default=1, gt=0, le=5)
    saturation_level: int = Field(default=255, ge=1, le=255)


class SensorError(Settings):
    frame_dropout_probability: float = Field(default=0, ge=0, le=1)
    exposure_gain: float = Field(default=1, gt=0)
    background_brightness: float = Field(default=0, ge=0, le=255)
    background_gradient: float = Field(default=0, ge=-255, le=255)
    shot_noise_scale: float = Field(default=0, ge=0)
    quantization_levels: int = Field(default=256, ge=2, le=256)
    saturation_level: int = Field(default=255, ge=1, le=255)
    hot_pixel_count: int = Field(default=0, ge=0, le=1000)
    blur_kernel_px: int = Field(default=7, ge=1, le=31)
    defocus_sigma_px: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def odd_kernel(self):
        if self.blur_kernel_px % 2 == 0:
            raise ValueError("Blur kernel must be odd")
        return self


class DistractorError(Settings):
    enabled: bool = False
    count: int = Field(default=0, ge=0, le=20)
    moving: bool = False
    intensity: float = Field(default=230, ge=0, le=255)
    radius_px: int = Field(default=7, ge=1, le=50)
    speed_px_s: float = Field(default=15, ge=0)
    appear_s: float = Field(default=0, ge=0)
    disappear_s: float = Field(default=1e9, gt=0)


class GlareError(Settings):
    enabled: bool = False
    intensity: float = Field(default=0, ge=0, le=255)
    centre_fraction: tuple[float, float] = (0.75, 0.25)
    radius_fraction: float = Field(default=0.2, gt=0, le=1)
    variation_fraction: float = Field(default=0, ge=0, le=1)


class AtmosphereError(Settings):
    enabled: bool = False
    attenuation: float = Field(default=1, ge=0, le=1)
    cloud_attenuation: float = Field(default=1, ge=0, le=1)
    beam_wander_sigma_rad: float = Field(default=0, ge=0)
    turbulence_strength: float = Field(default=0, ge=0, le=1)
    environment_brightness: float = Field(default=0, ge=0, le=255)
    elevation_scaling: bool = False
    visibility_km: float = Field(default=100, gt=0)


class ManoeuvreError(Settings):
    enabled: bool = False
    start_s: float = Field(default=0, ge=0)
    velocity_impulse_m_s: tuple[float, float, float] = (0, 0, 0)
    acceleration_m_s2: tuple[float, float, float] = (0, 0, 0)
    duration_s: float = Field(default=0, ge=0)


class PointAhead(Settings):
    angle_rad: tuple[float, float] = (0, 0)
    estimate_rad: tuple[float, float] = (0, 0)


class Disturbance(Settings):
    algorithm: str = "physical_error_system"
    preset_level: str = "NOMINAL"
    gaussian_noise: bool = True
    blur: bool = True
    navigation: NavigationError = NavigationError()
    attitude: AttitudeError = AttitudeError()
    vibration: VibrationError = VibrationError()
    gimbal: GimbalError = GimbalError()
    boresight: BoresightError = BoresightError()
    beacon: BeaconError = BeaconError()
    sensor: SensorError = SensorError()
    distractors: DistractorError = DistractorError()
    glare: GlareError = GlareError()
    atmosphere: AtmosphereError = AtmosphereError()
    manoeuvre: ManoeuvreError = ManoeuvreError()
    point_ahead: PointAhead = PointAhead()


class Vision(Settings):
    algorithm: str = "centroid"
    correction: str = "none"
    cnn_correction: bool = False
    threshold_strategy: str = "fixed"
    fixed_threshold: float = Field(default=120, ge=0, le=255)
    relative_peak_fraction: float = Field(default=0.55, gt=0, le=1)
    background_sigma_k: float = Field(default=4.0, ge=0)
    min_component_area: int = Field(default=3, ge=1)
    max_component_area: int = Field(default=10000, ge=1)
    candidate_selection: str = "prediction_intensity_size"
    roi_padding_px: int = Field(default=5, ge=0, le=128)
    denoise_kernel_px: int = Field(default=0, ge=0, le=15)
    morphology_kernel_px: int = Field(default=0, ge=0, le=15)
    gradient_threshold_fraction: float = Field(default=0.1, ge=0, le=1)
    gaussian_max_iterations: int = Field(default=15, ge=1, le=100)
    gaussian_min_sigma_px: float = Field(default=0.4, gt=0)
    gaussian_max_sigma_px: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def available(self):
        if self.correction not in {"none", "cnn_residual"}:
            raise ValueError("Unknown vision correction")
        if self.threshold_strategy not in {"fixed","relative_peak","background_sigma"}:
            raise ValueError("Unknown vision threshold strategy")
        if self.candidate_selection not in {"prediction_intensity_size","brightest","nearest_previous"}:
            raise ValueError("Unknown candidate-selection policy")
        if self.max_component_area < self.min_component_area:
            raise ValueError("max_component_area must be >= min_component_area")
        if self.gaussian_max_sigma_px <= self.gaussian_min_sigma_px:
            raise ValueError("Gaussian maximum sigma must exceed minimum sigma")
        return self


class CNNCorrection(Settings):
    enabled: bool = False
    model_id: str = "cnn-tiny-residual-v4"
    model_path: str = "models/cnn/cnn-tiny-residual-v4-best.pt"
    roi_size_px: int = Field(default=32, ge=16, le=128)
    use_aux_features: bool = True
    uncertainty_mode: str = "learned"
    max_correction_px: float = Field(default=8.0, gt=0)
    sigma_min_px: float = Field(default=0.15, gt=0)
    sigma_max_px: float = Field(default=20.0, gt=0)
    max_total_latency_ms: float = Field(default=50.0, gt=0)
    ood_z_threshold: float = Field(default=8.0, gt=0)
    fallback_to_classical: bool = True

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.sigma_max_px <= self.sigma_min_px:
            raise ValueError("CNN sigma_max_px must exceed sigma_min_px")
        if self.uncertainty_mode not in {"learned", "heuristic", "fixed"}:
            raise ValueError("Unknown CNN uncertainty mode")
        return self


class TemporalPrediction(Settings):
    history_frames: int = Field(default=20, ge=2, le=240)
    model_id: str = "gru-small-v1"
    model_path: str = "models/temporal/gru-small-v1-best.pt"
    gru_model_path: str = "models/temporal/gru-small-v1-best.pt"
    lstm_model_path: str = "models/temporal/lstm-small-v1-best.pt"
    gru_residual_model_path: str = "models/temporal/gru-residual-cv-v1-best.pt"
    fallback: str = "cv"
    max_displacement_px: float = Field(default=200.0, gt=0)
    max_sigma_px: float = Field(default=100.0, gt=0)
    input_feature_schema: tuple[str, ...] = (
        "x", "y", "vx", "vy", "log_cov_x", "log_cov_y",
        "confidence", "snr", "measurement_valid", "prediction_only", "dt"
    )

    @model_validator(mode="after")
    def valid_fallback(self):
        if self.fallback not in {"cv", "none"}:
            raise ValueError("Temporal fallback must be cv or none")
        return self


class SystemLatency(Settings):
    camera_s: float = Field(default=0.005, ge=0)
    vision_s: float = Field(default=0.002, ge=0)
    cnn_s: float = Field(default=0.0, ge=0)
    estimator_s: float = Field(default=0.001, ge=0)
    predictor_s: float = Field(default=0.0, ge=0)
    controller_s: float = Field(default=0.0002, ge=0)
    command_s: float = Field(default=0.002, ge=0)
    actuator_s: float = Field(default=0.03, ge=0)

    @property
    def total_effective_latency_s(self) -> float:
        return float(sum((self.camera_s, self.vision_s, self.cnn_s, self.estimator_s,
                         self.predictor_s, self.controller_s, self.command_s, self.actuator_s)))


class FeedForwardPID(Settings):
    gain: float = Field(default=0.8, ge=0, le=2)
    uncertainty_scale_px2: float = Field(default=25.0, gt=0)
    minimum_weight: float = Field(default=0.0, ge=0, le=1)


class GainScheduledPID(Settings):
    low_rate_rad_s: float = Field(default=0.002, ge=0)
    high_rate_rad_s: float = Field(default=0.02, gt=0)
    low_gain_scale: tuple[float, float, float] = (0.75, 0.7, 0.8)
    medium_gain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    high_gain_scale: tuple[float, float, float] = (1.3, 0.65, 1.25)
    uncertainty_scale_px2: float = Field(default=100.0, gt=0)
    smoothing: float = Field(default=0.2, gt=0, le=1)

    @model_validator(mode="after")
    def ordered_rates(self):
        if self.high_rate_rad_s <= self.low_rate_rad_s:
            raise ValueError("GS-PID high rate must exceed low rate")
        return self


class LQRConfig(Settings):
    tracking_error_weight: float = Field(default=30.0, gt=0)
    angular_rate_weight: float = Field(default=1.0, gt=0)
    control_effort_weight: float = Field(default=0.8, gt=0)
    riccati_iterations: int = Field(default=200, ge=20, le=2000)


class MPCConfig(Settings):
    prediction_horizon_steps: int = Field(default=10, ge=2, le=40)
    control_horizon_steps: int = Field(default=5, ge=1, le=40)
    tracking_error_weight: float = Field(default=35.0, gt=0)
    angular_rate_weight: float = Field(default=1.0, ge=0)
    control_effort_weight: float = Field(default=1.0, gt=0)
    command_change_weight: float = Field(default=0.5, ge=0)
    solver_timeout_ms: float = Field(default=8.0, gt=0)
    fallback_controller: str = "ff_pid"

    @model_validator(mode="after")
    def valid_horizon(self):
        if self.control_horizon_steps > self.prediction_horizon_steps:
            raise ValueError("MPC control horizon cannot exceed prediction horizon")
        if self.fallback_controller not in {"pid", "ff_pid"}:
            raise ValueError("MPC fallback must be pid or ff_pid")
        return self


class ControllerLabConfig(Settings):
    control_deadline_s: float = Field(default=1 / 30, gt=0)


class AcquisitionConfig(Settings):
    hold_duration_s: float = Field(default=0.15, ge=0)
    last_known_duration_s: float = Field(default=0.5, ge=0)
    scan_rate_rad_s: float = Field(default=0.08, gt=0)
    raster_period_s: float = Field(default=2.0, gt=0)
    raster_rows: int = Field(default=7, ge=2, le=100)
    spiral_radial_rate_rad_s: float = Field(default=0.012, gt=0)
    spiral_angular_rate_rad_s: float = Field(default=4.0, gt=0)
    covariance_sigma_scale: float = Field(default=2.0, gt=0)
    max_prediction_age_s: float = Field(default=0.5, gt=0)
    minimum_confirmation_confidence: float = Field(default=0.05, ge=0, le=1)
    innovation_gate_nis: float = Field(default=25.0, gt=0)
    search_command_limit_rad_s: float = Field(default=0.12, gt=0)


class OpticalLinkConfig(Settings):
    enabled: bool = False
    preset: str = "NOMINAL"
    model_version: str = "gaussian-far-field-1.0.0"
    wavelength_nm: float = Field(default=1550.0, gt=0)
    transmit_power_w: float = Field(default=1.0, gt=0)
    beam_divergence_urad: float = Field(default=25.0, gt=0)
    receiver_aperture_m: float = Field(default=0.15, gt=0)
    transmitter_efficiency: float = Field(default=0.75, gt=0, le=1)
    receiver_efficiency: float = Field(default=0.70, gt=0, le=1)
    range_mode: str = "AUTO_FROM_SCENARIO"
    manual_range_m: float = Field(default=1000.0, gt=0)
    atmosphere_mode: str = "FROM_SCENARIO"
    manual_atmospheric_transmission: float = Field(default=1.0, gt=0, le=1)
    receiver_sensitivity_dbm: float = -45.0
    good_margin_db: float = Field(default=6.0, gt=0)
    marginal_margin_db: float = Field(default=-3.0, lt=0)
    communication_noise_power_w: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_modes(self):
        if self.range_mode not in {"AUTO_FROM_SCENARIO","MANUAL_OVERRIDE"}:
            raise ValueError("optical range_mode must be AUTO_FROM_SCENARIO or MANUAL_OVERRIDE")
        if self.atmosphere_mode not in {"FROM_SCENARIO","MANUAL_OVERRIDE"}:
            raise ValueError("optical atmosphere_mode must be FROM_SCENARIO or MANUAL_OVERRIDE")
        if self.good_margin_db <= self.marginal_margin_db:
            raise ValueError("good link margin must exceed marginal margin")
        return self


class LockConfig(Settings):
    start_locked: bool = False
    lock_error_threshold_px: float = Field(default=20, gt=0)
    lock_required_frames: int = Field(default=3, ge=1)
    acquisition_required_frames: int = Field(default=2, ge=1)
    loss_error_threshold_px: float = Field(default=80, gt=0)
    max_missing_frames: int = Field(default=10, ge=1)
    reacquisition_required_frames: int = Field(default=2, ge=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_prompt1_names(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for old, new in {"handoff_error_threshold_px":"lock_error_threshold_px", "consecutive_frames":"lock_required_frames", "missing_frames_before_lost":"max_missing_frames"}.items():
                if old in data and new not in data:
                    data[new] = data.pop(old)
        return data

    @model_validator(mode="after")
    def thresholds(self):
        if self.loss_error_threshold_px < self.lock_error_threshold_px:
            raise ValueError("Loss threshold must be >= lock threshold")
        return self


class ExperimentConfig(Settings):
    simulation_schema_version: str = SIMULATION_SCHEMA_VERSION
    physics_model_version: str = PHYSICS_MODEL_VERSION
    scenario_version: str = "1.0.0"
    scenario: Scenario = Scenario.SATELLITE_SATELLITE
    scenario_preset: str = "SAT_SAT_NOMINAL"
    pipeline_preset: str = "DEFAULT_STABLE"
    pipeline_order: tuple[str, ...] = ("vision", "correction", "estimator", "predictor", "controller", "reacquisition")
    motion_model: str = "cw"
    cw_case: str = "custom"
    platform: Platform = Platform()
    kinematic: KinematicMotion = KinematicMotion()
    disturbance: Disturbance = Disturbance()
    vision: Vision = Vision()
    cnn: CNNCorrection = CNNCorrection()
    estimator: str = "kalman"
    predictor: str = "none"
    controller: str = "pid"
    reacquisition: str = "basic"
    prediction_horizon_s: float = Field(default=0, ge=0)
    temporal: TemporalPrediction = TemporalPrediction()
    latency: SystemLatency = SystemLatency()
    ff_pid: FeedForwardPID = FeedForwardPID()
    gain_scheduled_pid: GainScheduledPID = GainScheduledPID()
    lqr: LQRConfig = LQRConfig()
    mpc: MPCConfig = MPCConfig()
    controller_lab: ControllerLabConfig = ControllerLabConfig()
    acquisition: AcquisitionConfig = AcquisitionConfig()
    optical_link: OpticalLinkConfig = OpticalLinkConfig()
    lock: LockConfig = LockConfig()
    seed: int = Field(default=DEFAULT_MASTER_SEED, ge=0, le=4294967295)
    run_index: int = Field(default=0, ge=0)
    fps: float = Field(gt=0, le=120)
    time_scale: float = Field(gt=0, le=120)
    duration_s: float = Field(default=10, gt=0, le=10000)
    orbit: Orbit
    camera: Camera
    kalman: Kalman
    estimation: Estimation = Estimation()
    pid: PID

    @model_validator(mode="after")
    def pipeline_is_ordered(self):
        expected=("vision","correction","estimator","predictor","controller","reacquisition")
        if self.pipeline_order != expected:
            raise ValueError(f"Illegal pipeline ordering; required {expected}")
        if self.vision.cnn_correction != (self.vision.correction == "cnn_residual"):
            raise ValueError("CNN compatibility flag must match vision.correction")
        if self.vision.correction == "cnn_residual" and not self.cnn.enabled:
            raise ValueError("CNN correction requires cnn.enabled=true")
        if self.motion_model.startswith("cw") and self.scenario != Scenario.SATELLITE_SATELLITE:
            raise ValueError("Clohessy-Wiltshire motion is valid only for satellite_satellite scenarios")
        if self.cw_case != "custom" and self.motion_model != "cw_analytical":
            raise ValueError("Named CW cases require the cw_analytical motion model")
        return self

    def __getitem__(self, key):
        return self.model_dump(mode="json")[key]

    @property
    def point_ahead_error_rad(self):
        a, e = self.disturbance.point_ahead.angle_rad, self.disturbance.point_ahead.estimate_rad
        return (e[0]-a[0], e[1]-a[1])
