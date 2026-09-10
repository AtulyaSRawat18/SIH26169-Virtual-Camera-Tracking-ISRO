# Temporal dataset

Dataset version `temporal-state-1.3.0` is generated from actual AKF-R outputs, not
ground-truth state inputs. Each sample contains 12 timestamped rows of `(x, y, vx, vy)`,
log covariance, confidence, SNR, measurement-valid/prediction-only flags, and `dt`.
Labels are future ground-truth pixels used only for evaluation/training.

Ground truth is linearly interpolated to the exact requested physical horizon. Thus a
50 ms label remains 50 ms even when the simulation runs at 30 FPS. The production
dataset matches the live 640×480, 520 px focal-length camera and contains 4,704 samples from 21 trajectories.
Train, validation, and test each contain one whole trajectory from every ordinary
scenario family; aggressive UAV motion is isolated as OOD.

```powershell
python -m src.prediction.cli generate --trajectories 3 --frames 70 --history 12 --fps 30 --seed 42 --width 640 --height 480 --focal-length 520
python -m src.prediction.cli train data/ml/temporal/<dataset-id> --kind gru --model-id gru-small-v1 --epochs 30 --patience 6
```

The dataset manifest records upstream estimator/correction versions, camera scale,
seeds, feature schema, horizons, scenario distributions, and the trajectory split.
