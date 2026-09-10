# Experiment Evidence Method

## Paired design

Baseline and candidate runs use the same scenario, physical configuration, master seed, run index, subsystem seeds, initial state, environment, beacon conditions, dropouts, and disturbances. Only validated algorithm-pipeline fields may differ. The comparison rejects physically unequal configurations.

For metric `m`, each pair preserves `delta_i = m_candidate_i - m_baseline_i`. Direction semantics are explicit: lower is better for error, latency, effort and reacquisition time; higher is better for lock, measurement availability and link availability.

## Confidence and effect size

Continuous paired improvements use a deterministic percentile bootstrap of the signed improvements: 2,000 resamples, 95% interval, seed equal to the experiment master seed. The report includes mean and median paired difference and paired standardized effect size `d_z = mean(signed improvement) / sample_std(signed improvement)` when variance is non-zero. A confidence interval or effect size does not replace physical magnitude.

## Monte Carlo and cumulative history

Default evidence scale is 100 deterministic runs; the API also supports 10, custom counts, and up to 10,000 headless runs. Each run is append-only. Cumulative summaries include only matching simulation-schema and physics-model versions unless incompatibility is explicitly requested. Current run, current batch, and compatible cumulative history remain distinguishable.

## Sweeps and scenario evidence

One-variable sweeps use multiple seeds per point and store a separate batch for every point. Supported headline variables include vibration, image noise, beacon intensity, beam wander, dropout, angular LOS rate, gimbal latency, UAV speed, and UAV altitude. Scenario summaries group by exact pipeline. Recommendations and regime maps may only be produced from measured groups; missing evidence remains missing.

## Failures, worst cases and Pareto interpretation

Failure counts include every complete run and separately flag zero link availability. Worst-case records retain the highest tracking RMSE together with replay metadata. Pareto analysis independently minimizes tracking error, latency, control effort and reacquisition time while maximizing lock and link availability. It deliberately produces a frontier rather than one opaque score.

## Anti-cherry-picking rules

- Preserve all seeds and failed runs.
- Never hide negative module improvement.
- Never pool incompatible model versions silently.
- Never compare different physical draws and label the result paired.
- Never imply correlation or a waterfall transition proves unique causation.
- Report latency, actuator effort, saturation and link consequence alongside accuracy.
- Keep unvalidated CNN or temporal models labelled experimental.

## Prompt 10 measured smoke evidence

The checked-in Prompt 10 evidence used five paired seeds for `DEFAULT_STABLE` versus `BEST_CLASSICAL_ESTIMATOR`, plus three seeds per sweep point. This small batch validates the evidence machinery; it is not the final 100-run acceptance batch.

- Tracking RMSE changed from 13.0662 px to 13.0643 px: 0.0146% improvement.
- p95 pointing error degraded by 0.00323%.
- Lock retention was unchanged at 60%.
- The paired tracking-improvement bootstrap interval was `[0.000027, 0.003783]` px for this short batch.
- At beacon intensity 50 DN, lock success fell to 0% and mean tracking RMSE rose to 20.1923 px.
- At 8% and 20% random beacon dropout, lock success was only 33.3% in the sampled short runs.
- The 0–50 µrad vibration sweep showed no meaningful degradation in this short regime; the slight numerical trend must not be presented as vibration improving tracking.
- Link availability remained 0% under the coarse-pointing/optical settings, demonstrating that coarse camera lock alone is insufficient for an FSOC fine-pointing claim.

The final acceptance run in Prompt 11 should expand these experiments to the preferred 100 paired seeds and presentation-selected scenarios.
