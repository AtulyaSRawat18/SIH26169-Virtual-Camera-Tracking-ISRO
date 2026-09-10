# Estimator benchmark method

Open-loop replay and closed-loop PAT are deliberately separate.

`EstimatorReplaySequence` records timestamps, Prompt 3 measurements/quality/covariance, actual camera calibration, scenario metadata and truth beside each sample for evaluation only. NONE, KF-CV, EKF-angular, UKF-angular and AKF-R then consume the exact saved measurement objects without rerendering. This removes vision randomness and timing as a confounder.

Closed-loop comparison uses the existing paired-seed engine. Each estimator drives PID/gimbal, so later camera frames legitimately differ. Results therefore report closed-loop tracking/lock metrics separately from pure replay estimation accuracy.

The required deterministic matrix covers nominal, high measurement noise, weak beacon, high vibration, manoeuvre, short/long dropout, distractor/outlier, ground-satellite stress, UAV aggressive motion and combined stress. Per-seed differences against KF-CV are retained; they are not presented as significance tests.

Open-loop metrics include pixel/angular RMSE, MAE, median, p95/max, dropout RMSE, NIS, position NEES, covariance trace, uncertainty coverage and estimator-only latency. Closed-loop metrics use tracking RMSE/p95, time and percentage locked, lock losses/reacquisition, gimbal saturation and control effort.

Q/R sweeps replay the same measurement sequence. A fair study tunes on designated training scenarios, validates once, then freezes parameters before test seeds. Failed numerical updates and rejected measurements remain in results.

Neither low RMSE alone nor algorithm complexity selects a winner. The report retains accuracy, dropout robustness, consistency/calibration, manoeuvre recovery, latency, closed-loop performance and accuracy/compute trade-off as separate operating-regime dimensions.
