# Predictor benchmark method

Open-loop replay records one estimator sequence and gives the identical history to
every predictor. Future truth is interpolated at the requested physical horizon and is
used only for scoring. Closed-loop comparison then runs paired seeds through the real
controller and actuator. Horizon and history sweeps are exposed through the API and
dashboard.

The scenario-stratified held-out result at 20/50/100/200 ms was:

| Predictor | Held-out RMSE | OOD RMSE | Mean CPU latency |
|---|---:|---:|---:|
| NONE | 14.288 px | 17.254 px | ~0 ms |
| CV | 16.950 px | 21.116 px | <0.1 ms |
| GRU | 14.715 px | 11.503 px | 1.75 ms |
| LSTM | 14.354 px | 8.733 px | 1.35 ms |
| GRU residual CV | 19.908 px | 13.513 px | 1.92 ms |

The direct GRU was excellent on ground-satellite/UAV families but regressed on the
satellite-satellite family, so the global promotion gate failed. On a short nominal
live replay it slightly beat CA (6.683 vs 6.796 px RMSE), but the paired closed-loop run
was worse: GRU+FF-PID 79.759 px versus NONE+PID 72.723 px. This shows why open-loop
accuracy alone is not enough.

The current limitation is system identification/tuning across mixed motion regimes.
Recommended next experiments are scenario-conditioned normalization, longer
scenario-balanced trajectories, and controller retuning—not a larger model by default.
See `evidence/temporal_prediction_summary.json` for the concise evidence record.
