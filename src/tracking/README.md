# State Estimation — Member 4

Available estimators are NONE, linear image-space KF-CV, angular EKF and UKF with
the genuine tangent camera measurement, and measurement-quality adaptive KF-R.
They share one controller-facing state and expose prediction-only status, covariance,
95% image ellipse, NIS/NEES diagnostics, numerical failures and latency.

GRU/LSTM is not implemented. Track RMSE/p95, dropout error, consistency, recovery,
rejections and mean/p95 latency. See `docs/STATE_ESTIMATION.md`.
