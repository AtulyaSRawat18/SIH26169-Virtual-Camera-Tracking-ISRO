# Classical Vision Lab benchmark method

Two tests answer different questions:

- `ORACLE_ROI` / localization-only: the offline harness crops around evaluation truth, then passes only the crop to the localizer. The algorithm never receives the true centre. This isolates centre mathematics.
- `FULL_FRAME`: the production detection, association and localization path receives the complete camera image. Association failures are included.

For same-frame comparison, one immutable source image is copied to each algorithm and SHA-256 input checksums are recorded. Noise is not regenerated. For paired Monte Carlo, every algorithm uses the same scenario seed, target, jitter, background, attenuation, dropout and distractors. Closed-loop comparisons use the existing paired runner separately because algorithm/controller motion changes later frames.

The controlled matrix covers clean, weak beacon, high noise, blur, ellipticity, bright/gradient background, glare, distractors, clipping, saturation, dynamic vibration and combined stress. Satellite–satellite, ground–satellite and UAV profiles continue to come from Prompt 2.

Reported localization metrics are RMSE, MAE, median, p95, maximum, valid percentage and failure reason. Latency mean/p95 is kept separate from accuracy. Closed-loop metrics remain tracking RMSE, time/percentage locked, lock losses and reacquisition. A parameter sweep runs multiple deterministic seeds per value.

Fairness rules:

- ground truth is evaluation-only;
- input frames are identical within a comparison;
- parameters come from recorded configuration, never per-frame manual tuning;
- failed detections remain in availability/failure metrics;
- no universal winner is declared from one lucky run;
- accuracy, robustness and latency are retained as separate Pareto dimensions.

Machine-readable results are appended to `lab_results` in `data/experiments.sqlite3`. UI actions can export selected samples for future CNN work, but export is disabled during ordinary tracking.
