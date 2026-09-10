"""Command-line workflow for the CNN spot-correction laboratory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.dataset import generate_spot_dataset
from src.ml.training import CNNTrainingConfig, train_spot_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate a trajectory-safe ROI dataset")
    generate.add_argument("--trajectories", type=int, default=4)
    generate.add_argument("--frames", type=int, default=32)
    generate.add_argument("--seed", type=int, default=42)
    train = subcommands.add_parser("train", help="Train the residual CNN")
    train.add_argument("dataset", type=Path)
    train.add_argument("--model-id", default="cnn-tiny-residual-v4")
    train.add_argument("--model-version", default="4.0.0")
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--roi-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "generate":
        output = generate_spot_dataset(
            trajectories_per_scenario=arguments.trajectories,
            frames_per_trajectory=arguments.frames,
            seed=arguments.seed,
        )
    else:
        output = train_spot_model(
            arguments.dataset,
            CNNTrainingConfig(
                model_id=arguments.model_id,
                model_version=arguments.model_version,
                use_aux_features=not arguments.roi_only,
                epochs=arguments.epochs,
                early_stopping_patience=arguments.patience,
            ),
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
