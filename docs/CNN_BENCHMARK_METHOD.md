# CNN benchmark method

The primary test is paired and uses the same frames, seeds, classical localizer, and
physical configuration for both paths. Metrics include RMSE, median, p95, maximum
error, false-correction rate, uncertainty coverage/NLL, inference latency, estimator
consistency, fallback count, and closed-loop tracking metrics.

| Test | OpenCV | CNN v4 | Result |
|---|---:|---:|---|
| Held-out localization RMSE | 0.08675 px | 0.08690 px | no gain |
| Held-out false corrections | — | 52.5% | unacceptable for promotion |
| OOD localization RMSE | 0.99609 px | 2.20999 px | regression |
| CPU inference mean | — | 1.78 ms | real-time model only |
| Live closed-loop, 2 paired seeds | 72.72372 px | 72.72372 px | equal because all 45 frames/run safely fell back |

The live full-resolution input was outside the learned auxiliary-feature envelope, so
the runtime correctly selected OpenCV. The model is therefore an experimental
ablation. `DEFAULT_STABLE` remains unchanged. Raw summary numbers are in
`evidence/cnn_spot_correction_summary.json` and the full training report is under
`models/cnn/`.
