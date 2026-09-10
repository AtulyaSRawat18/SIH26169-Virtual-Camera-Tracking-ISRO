# CNN dataset and reproducibility

Dataset version `spot-residual-1.2.0` stores ROI tensors, residual labels, auxiliary
features, trajectory IDs, scenario names, seeds, split names, simulator versions, and
an immutable generation definition. Splits are made by whole trajectory and stratified
by scenario; a frame from one trajectory cannot leak into another split.

The current evidence dataset contains 768 samples from 24 trajectories. Counts are
320 train, 160 validation, 160 held-out test, and 128 OOD test. The OOD set uses unseen
aggressive UAV manoeuvre/vibration. Association failures are excluded from regression
and recorded as hard negatives rather than assigned a false residual target.

Reproduce it with:

```powershell
python -m src.ml.cli generate --trajectories 4 --frames 32 --seed 42
python -m src.ml.cli train data/ml/spot/<dataset-id> --model-id cnn-tiny-residual-v4 --model-version 4.0.0 --epochs 30 --patience 6
```

Training statistics are fitted on the training split only. Early stopping and
uncertainty calibration use validation only; the held-out and OOD splits remain final
evaluation sets.
