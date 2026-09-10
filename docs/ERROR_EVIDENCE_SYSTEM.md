# Error Evidence System

## Purpose

The evidence layer answers a causal engineering question: which configured physical disturbance was applied, what instantaneous realization occurred, how errors propagated through PAT, and what optical consequence followed. Ground truth is used only after each operational stage returns its output; it is never passed into vision, correction, estimation, prediction, control, or reacquisition contracts.

## Standard errors

| Name | Exact evaluation | Unit | Directly influenced by |
|---|---|---|---|
| `e_measurement` | `||p_classical - p_true||` | px | Classical localization |
| `e_cnn` | `||p_corrected - p_true||` when correction is enabled | px | CNN correction |
| `e_estimation` | `||p_est - p_true||` | px | Estimator |
| `e_prediction` | `||p_hat(t+tau) - p_true(t+tau)||` when the queued horizon arrives | px | Predictor |
| `e_control` | `||(p_est-p_centre)/f_px||` small-angle control residual | rad | Controller input/state |
| `e_actuator` | `||commanded_rate - actual_rate||` | rad/s | Gimbal dynamics, latency and limits |
| `e_pointing` | `sqrt((LOS_pan-axis_pan)^2 + (LOS_tilt-axis_tilt)^2)` | rad | Complete closed loop |

Pixel errors convert to the common angular domain using `theta ~= pixel_error / focal_length_px`. Navigation errors remain in metres unless a valid range-dependent LOS conversion is applied. The UI does not add incompatible units or claim a unique additive decomposition in this nonlinear loop.

## Live error flow

Every telemetry record contains `error_budget`, `error_waterfall`, and `resolved_error_config`. The waterfall displays the real current stage values. A same-unit transition is labelled `IMPROVED`, `DEGRADED`, or `UNCHANGED`; negative AI improvement remains visible. The run summary provides mean, RMSE, median, p95, maximum, sample count, live link margin, and PAT state.

Configured disturbance parameters and their instantaneous realizations are separate. For example, configured vibration sigma appears in `resolved_error_config`, while the current sampled/sinusoidal disturbance appears in `error_budget.vibration_rad`.

## User-controlled variables

The registry in `src/experiments/evidence.py` groups variables into geometry/navigation, platform, environment, optics, beacon, camera/sensor, actuator, and failure events. It marks the scenarios where each variable is relevant. `NOMINAL`, `LOW`, `MEDIUM`, `HIGH`, and `STRESS` resolve to explicit vibration, image noise, blur, dropout, latency, turbulence, cloud-transmission, and beam-wander values. The complete advanced configuration remains editable in the Experiment view.

Monte Carlo parameters accept:

- `fixed`: `{"distribution":"fixed","value":8e-6}`
- `uniform`: `{"distribution":"uniform","low":5e-6,"high":12e-6}`
- `normal`: `{"distribution":"normal","mean":8e-6,"std":2e-6}`

Sampling uses the isolated `parameter_sampling` RNG stream derived from master seed and run index. Each stored run keeps the configured distribution, effective sampled value, sampling seed, subsystem seeds, and fully resolved configuration.

## Causal experiments

The Evidence dashboard supports a controlled five-point, multi-seed sweep. Each point reports mean, p95, lock success, link availability, and a seeded 95% bootstrap confidence interval. A linear local sensitivity is reported as an experimental response, not a universal causal coefficient.

The paired Before/After mode holds physical configuration and common random streams constant. Its baseline is centroid → KF → PID → basic reacquisition; the candidate is the exact currently selected pipeline. It reports absolute change, direction-aware percentage improvement, degradation, paired confidence intervals, and compute cost.

## Failures and replay evidence

Compatible stored runs aggregate weak-beacon, out-of-FOV, false-target, saturation, dropout, excessive-pointing, link-unavailable, and unknown outcomes. The API preserves the five worst tracking-RMSE runs with seed, resolved configuration, pipeline, metrics, and failure cause. Those fields are sufficient to replay the exact run. Failed runs remain in denominators.

## Export

The dashboard exports scenario, seed, resolved variables, live waterfall, running metrics, paired result, sweep, and compatible cumulative history as JSON. `/api/evidence/export` provides the same machine-readable evidence path.

## Scientific limitations

The optical model is an explicit Gaussian far-field approximation, the camera uses a pinhole/small-angle mapping, and short demonstration batches are not qualification evidence. Error sources interact nonlinearly; waterfall stages are sequential evaluations, while ablations and sweeps estimate system response. No single architecture is declared universally best.
