"""Same-image and closed-loop experiments for learned measurement correction."""
from __future__ import annotations

import base64
from pathlib import Path
import math

import cv2
import numpy as np

from src.core.randomness import RandomStreams
from src.core.registry import REGISTRY, Stage, validate_config
from src.experiments.monte_carlo import compare_paired
from src.ml.correctors import CNNResidualCorrector
from src.ml.dataset import generate_spot_dataset
from src.ml.training import CNNTrainingConfig, train_spot_model


def _encoded(frame):
    ok, image = cv2.imencode(".png", frame)
    return None if not ok else "data:image/png;base64," + base64.b64encode(image).decode()


def compare_cnn_same_frame(config, frame: np.ndarray, truth_pixel=None, model_path=None) -> dict:
    base_data = validate_config(config).model_dump(mode="json")
    base_data["vision"].update(correction="none", cnn_correction=False)
    base_data["cnn"]["enabled"] = False
    base_data["pipeline_preset"] = "CUSTOM"
    base = validate_config(base_data)
    classical = REGISTRY[Stage.VISION][base.vision.algorithm](base, RandomStreams(base.seed, base.run_index)).measure(frame.copy(), 0.0)
    cnn_data = base.model_dump(mode="json")
    cnn_data["vision"].update(correction="cnn_residual", cnn_correction=True)
    cnn_data["cnn"]["enabled"] = True
    if model_path is not None:
        cnn_data["cnn"]["model_path"] = str(model_path)
    corrected = CNNResidualCorrector(validate_config(cnn_data)).correct(frame.copy(), classical)
    overlay = frame.copy()
    if classical.valid:
        cv2.drawMarker(overlay, tuple(int(round(v)) for v in classical.pixel), (0, 190, 255), cv2.MARKER_CROSS, 13, 2)
    if corrected.valid:
        cv2.drawMarker(overlay, tuple(int(round(v)) for v in corrected.pixel), (255, 80, 80), cv2.MARKER_TILTED_CROSS, 13, 2)
    if truth_pixel is not None:
        cv2.circle(overlay, tuple(int(round(v)) for v in truth_pixel), 5, (80, 255, 80), 1)
    classical_error = None if truth_pixel is None or not classical.valid else float(np.linalg.norm(np.asarray(classical.pixel) - truth_pixel))
    corrected_error = None if truth_pixel is None or not corrected.valid else float(np.linalg.norm(np.asarray(corrected.pixel) - truth_pixel))
    return dict(
        kind="CNN_SAME_FRAME_COMPARISON", evaluation_truth_used_only_for_metrics=truth_pixel is not None,
        frame_checksum=__import__("hashlib").sha256(frame.tobytes()).hexdigest(),
        classical=classical.as_dict(), corrected=corrected.as_dict(),
        classical_error_px=classical_error, corrected_error_px=corrected_error,
        improvement_px=None if classical_error is None or corrected_error is None else classical_error - corrected_error,
        overlay_image=_encoded(overlay),
    )


def build_and_train_cnn(trajectories_per_scenario=4, frames_per_trajectory=24,
                        epochs=18, use_aux_features=True) -> dict:
    dataset = generate_spot_dataset(trajectories_per_scenario=trajectories_per_scenario,
                                    frames_per_trajectory=frames_per_trajectory)
    training = CNNTrainingConfig(epochs=epochs, use_aux_features=use_aux_features)
    return {"dataset": dataset, "training": train_spot_model(dataset["directory"], training)}


def closed_loop_cnn_comparison(config, runs=3, duration_s=3.0, base_seed=42, store=None) -> dict:
    classical_data = validate_config(config).model_dump(mode="json")
    classical_data.update(pipeline_preset="CUSTOM", estimator="akf_r", predictor="none", controller="pid")
    classical_data["vision"].update(algorithm="gradient_centroid", correction="none", cnn_correction=False)
    classical_data["cnn"]["enabled"] = False
    cnn_data = __import__("copy").deepcopy(classical_data)
    cnn_data["vision"].update(correction="cnn_residual", cnn_correction=True)
    cnn_data["cnn"]["enabled"] = True
    return compare_paired(validate_config(classical_data), validate_config(cnn_data), runs=runs,
                          base_seed=base_seed, duration_s=duration_s, summary_only=True, store=store)
