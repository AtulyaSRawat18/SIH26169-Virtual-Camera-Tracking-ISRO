"""One registry controls validation, factories and UI capabilities."""
from enum import Enum
from src.core.config import ExperimentConfig
from src.core.stages import AnalyticalCWMotion, CWMotion, KinematicMotion, DisturbanceStage
from src.vision.localizers import (LegacyCentroid, BinaryCentroid, IntensityWeightedCentroid,
                                   GradientCentroid, GaussianFit)
from src.estimation.filters import NoneEstimator, PixelCVKF, AngularEKF, AngularUKF, AdaptivePixelKF
from src.ml.correctors import CNNResidualCorrector, NoCorrection
from src.prediction.predictors import (ConstantAccelerationPredictor, ConstantVelocityPredictor,
                                       GRUPredictor, GRUResidualCVPredictor, LSTMPredictor, NoPrediction)
from src.control.controllers import (FeedForwardPIDController, GainScheduledPIDController,
                                     LQRController, MPCController, PIDController)
from src.control.reacquisition import (CovarianceSearch, HoldSearch, HybridSearch, LastKnownSearch,
                                       NoSearch, PredictedPointSearch, PredictiveCovarianceSearch,
                                       RasterSearch, SpiralSearch)


class Stage(str, Enum):
    MOTION = "motion_model"
    DISTURBANCE = "disturbance"
    VISION = "vision"
    CORRECTION = "correction"
    ESTIMATOR = "estimator"
    PREDICTOR = "predictor"
    CONTROLLER = "controller"
    REACQUISITION = "reacquisition"


REGISTRY = {
    Stage.MOTION: {"cw": CWMotion, "cw_analytical": AnalyticalCWMotion, "kinematic": KinematicMotion},
    Stage.DISTURBANCE: {"physical_error_system": DisturbanceStage},
    Stage.VISION: {"centroid": LegacyCentroid, "legacy_centroid": LegacyCentroid,
                   "binary_centroid": BinaryCentroid, "weighted_centroid": IntensityWeightedCentroid,
                   "gradient_centroid": GradientCentroid, "gaussian_fit": GaussianFit},
    Stage.ESTIMATOR: {"kalman": PixelCVKF, "kf_cv": PixelCVKF, "none": NoneEstimator,
                      "ekf_angular": AngularEKF, "ukf_angular": AngularUKF, "akf_r": AdaptivePixelKF},
    Stage.CORRECTION: {"none": NoCorrection, "cnn_residual": CNNResidualCorrector},
    Stage.PREDICTOR: {"none": NoPrediction, "cv": ConstantVelocityPredictor,
                      "ca": ConstantAccelerationPredictor,
                      "gru": GRUPredictor, "lstm": LSTMPredictor,
                      "gru_residual_cv": GRUResidualCVPredictor},
    Stage.CONTROLLER: {"pid": PIDController, "ff_pid": FeedForwardPIDController,
                       "gain_scheduled_pid": GainScheduledPIDController,
                       "lqr": LQRController, "mpc": MPCController},
    Stage.REACQUISITION: {"basic": HybridSearch, "hybrid": HybridSearch, "hold": HoldSearch,
                          "last_known": LastKnownSearch, "raster": RasterSearch,
                          "spiral": SpiralSearch, "predicted_point": PredictedPointSearch,
                          "covariance_search": CovarianceSearch,
                          "predictive_covariance": PredictiveCovarianceSearch,
                          "none": NoSearch},
}
CORRECTIONS = REGISTRY[Stage.CORRECTION]


def selections(config):
    result = {}
    for stage in Stage:
        if stage in (Stage.VISION, Stage.DISTURBANCE):
            result[stage] = getattr(config, stage.value).algorithm
        elif stage is Stage.CORRECTION:
            result[stage] = config.vision.correction
        else:
            result[stage] = getattr(config, stage.value)
    return result


def validate_config(value):
    config = value if isinstance(value, ExperimentConfig) else ExperimentConfig.model_validate(value)
    for stage, name in selections(config).items():
        if name not in REGISTRY[stage]:
            raise ValueError(f"Unavailable {stage.value} algorithm '{name}'. Registered: {', '.join(REGISTRY[stage])}")
    if config.vision.correction == "cnn_residual" and not config.cnn.enabled:
        raise ValueError("cnn_residual correction requires cnn.enabled=true")
    return config


def build_stages(config, rng):
    return {stage: REGISTRY[stage][name](config, rng) for stage, name in selections(config).items()}


def capabilities():
    result={stage.value: list(algorithms) for stage, algorithms in REGISTRY.items()}
    return result


def algorithm_versions(config):
    built=selections(config)
    return {stage.value:getattr(REGISTRY[stage][name],"version","1.0.0") for stage,name in built.items()}


def algorithm_metadata(config, stages=None):
    versions = algorithm_versions(config)
    vision = dict(algorithm=config.vision.algorithm, version=versions["vision"],
                  threshold_strategy=config.vision.threshold_strategy,
                  preprocessing_config={"denoise_kernel_px": config.vision.denoise_kernel_px,
                                        "morphology_kernel_px": config.vision.morphology_kernel_px},
                  candidate_selection_method=config.vision.candidate_selection,
                  benchmark_mode="FULL_FRAME")
    estimator = None
    correction = None
    predictor = None
    controller = None
    if stages is not None:
        implementation = stages[Stage.ESTIMATOR]
        if hasattr(implementation, "metadata"):
            estimator = implementation.metadata()
        for stage, target in ((Stage.CORRECTION, "correction"), (Stage.PREDICTOR, "predictor"),
                              (Stage.CONTROLLER, "controller")):
            implementation = stages[stage]
            value = implementation.metadata() if hasattr(implementation, "metadata") else {
                "name": selections(config)[stage], "version": versions[stage.value]}
            if target == "correction": correction = value
            elif target == "predictor": predictor = value
            else: controller = value
    return {"vision": vision, "correction": correction, "estimator": estimator,
            "predictor": predictor, "controller": controller}
