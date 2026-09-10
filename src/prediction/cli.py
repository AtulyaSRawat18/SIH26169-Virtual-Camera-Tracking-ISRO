"""Command-line workflow for the temporal prediction laboratory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prediction.dataset import generate_temporal_dataset
from src.prediction.training import TemporalTrainingConfig, train_temporal_model


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate a trajectory-safe temporal dataset")
    generate.add_argument("--trajectories", type=int, default=3)
    generate.add_argument("--frames", type=int, default=70)
    generate.add_argument("--history", type=int, default=12)
    generate.add_argument("--fps", type=float, default=30.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--width", type=int, default=640)
    generate.add_argument("--height", type=int, default=480)
    generate.add_argument("--focal-length", type=float, default=520.0)
    generate.add_argument("--beacon-radius", type=int, default=8)
    generate.add_argument("--noise-sigma", type=float, default=5.0)

    train = subcommands.add_parser("train", help="Train one comparable temporal model")
    train.add_argument("dataset", type=Path)
    train.add_argument("--kind", choices=("gru", "lstm", "gru_residual_cv"), required=True)
    train.add_argument("--model-id", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=6)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "generate":
        output = generate_temporal_dataset(
            trajectories_per_scenario=arguments.trajectories,
            frames_per_trajectory=arguments.frames,
            history_frames=arguments.history,
            horizons_s=(0.02, 0.05, 0.1, 0.2),
            fps=arguments.fps,
            seed=arguments.seed,
            camera_width_px=arguments.width,
            camera_height_px=arguments.height,
            focal_length_px=arguments.focal_length,
            beacon_radius_px=arguments.beacon_radius,
            camera_noise_sigma=arguments.noise_sigma,
        )
    else:
        output = train_temporal_model(
            arguments.dataset,
            TemporalTrainingConfig(
                model_id=arguments.model_id,
                kind=arguments.kind,
                epochs=arguments.epochs,
                early_stopping_patience=arguments.patience,
            ),
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
