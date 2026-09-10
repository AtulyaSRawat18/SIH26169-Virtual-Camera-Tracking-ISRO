# Classical Vision — Member 3

Candidate generation is separate from subpixel localization. Registry choices are
the preserved legacy contour centroid, binary centroid, background-subtracted
intensity centroid, Sobel-gradient centroid and bounded 2-D Gaussian fit. All return
one `SpotMeasurement` with quality, image-SNR proxy, failure reason and optional
covariance. Truth is evaluation-only. No YOLO/CNN is implemented yet.

Track localization RMSE/p95, valid rate, distractor confusion, clipping/saturation,
confidence features and mean/p95 latency. See `docs/CLASSICAL_VISION.md`.
