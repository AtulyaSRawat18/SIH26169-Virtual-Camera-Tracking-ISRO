# Final System Overview

## Product and claim

SIH26169 is a software-in-the-loop Pointing, Acquisition and Tracking (PAT) testbed for mobile free-space optical communication terminals. A user selects a physical scenario, controls platform/environment/optical/sensor errors, runs one registered pipeline, and measures localization, estimation, prediction, control, reacquisition, pointing, and optical-link consequences using deterministic experiments.

It demonstrates a reproducible engineering workflow. It does not claim a new universal AI tracker, flight readiness, or hardware performance.

## Runtime architecture

```text
SCENARIO / MOTION (analytical or numerical CW; scenario kinematics)
       │
       ▼
PHYSICAL ERROR ENGINE ── navigation, attitude, vibration, atmosphere,
       │                  beam wander, beacon, sensor, failures
       ▼
TRUE STATE ──► VIRTUAL PINHOLE CAMERA ──► RAW SENSOR FRAME
                                               │
                                               ▼
OpenCV localizer → optional CNN correction → estimator → predictor
                                               │
                                               ▼
                                  PAT state / control arbitration
                                      ┌────────┴────────┐
                                      ▼                 ▼
                                  tracking     search/reacquisition
                                      └────────┬────────┘
                                               ▼
                                    constrained actuator / gimbal
                                               │
                                               └──── camera loop

Parallel evaluation only:
truth → stage errors → pointing error → Gaussian optical link
      → seeded Monte Carlo → compatible cumulative evidence
```

Operational contracts never receive ground truth. Truth enters only after stage outputs exist, for scoring and optional visibly labelled evaluation overlays.

## Scenarios and physics

- Satellite ↔ Satellite: CW relative motion, including exact analytical bounded, in-plane, drift, cross-track, and 3-D formation cases. CW is restricted to satellite pairs.
- Ground ↔ Satellite: kinematic angular pass with elevation-scaled simplified atmosphere.
- UAV ↔ Ground, UAV ↔ UAV, UAV ↔ Satellite: kinematic motion, manoeuvres, wind proxy, attitude and rotor/platform vibration.

Physical controls are grouped as geometry/navigation, platform, environment, optics, beacon, camera/sensor, actuator, and failure events. `NOMINAL`, `LOW`, `MEDIUM`, `HIGH`, and `STRESS` are resolved into numerical values; fixed, uniform, and normal Monte Carlo distributions store both their specification and sampled value.

## Selected algorithms

| Stage | Stable | Experimental / comparison |
|---|---|---|
| Vision | centroid, binary, weighted, gradient, Gaussian fit | — |
| Correction | none | CNN residual + learned uncertainty |
| Estimator | KF-CV, EKF angular, UKF angular, adaptive KF-R | — |
| Predictor | none, CV, CA | GRU, LSTM, GRU residual-CV |
| Controller | PID, FF-PID, gain-scheduled PID, LQR | constrained MPC with fallback |
| Reacquisition | hold, last-known, raster, spiral, predicted, covariance, predictive covariance, hybrid | — |

`DEFAULT_STABLE`, `REFERENCE_BASELINE`, and `CURRENT_BEST_VALIDATED` all resolve to centroid → KF → no predictor → PID → hybrid/basic reacquisition after the final promotion gate. The Gradient+UKF candidate produced a tiny mean closed-loop gain but worse localization and latency, so it was not promoted. CNN and temporal models remain visible and replayable experiments with classical fallbacks.

## PAT behavior

The state machine is `SEARCH → ACQUIRE → TRACK → LOCKED`, with `LOCKED/TRACK → LOST → REACQUIRE → TRACK`. Candidate confidence, innovation consistency, dwell, loss thresholds, FOV, gimbal limits, prediction, covariance, and fallback search govern transitions. The live UI shows time in state and the last 100 state/configuration events.

## Error chain

```text
physical disturbance → image degradation → measurement error
 → optional CNN error → estimation error → horizon-aligned prediction error
 → control residual → actuator rate error → final optical-axis pointing error
 → pointing loss → received power → link margin/state
```

Not every active stage reduces error on every frame. Same-unit transitions are shown as improved, degraded, or unchanged. Angularly compatible terms can be compared in radians/µrad; metres, pixels, and radians are never naively summed.

## Experiment and evidence architecture

Live, headless, replay, and paired runs use the same engine and error implementation. RNG streams are deterministically derived from master seed and run index, so changing camera-noise draws cannot alter dropout draws. Stored records include scenario/config versions, exact pipeline/model metadata, git hash, sampled inputs, resolved config, seeds, metrics, failures, and optional bounded telemetry.

The Evidence area provides current flow, paired Before/After, multi-seed parameter sweeps, seeded bootstrap intervals, effect size, compatible cumulative history, failure/worst-case records, six-objective Pareto results, JSON/CSV export, and a print-ready HTML report.

## Final acceptance result

Batch `a71b8273-19d6-49b0-9080-91fb5a9d765c` ran 100 paired `DEMO_6_COMBINED_STRESS` seeds per pipeline. Gradient+UKF improved mean tracking RMSE by only 0.0463% and estimator RMSE by 0.0345 px, but localization RMSE rose from 0.1175 to 0.3532 px and latency rose from 6.12 to 7.02 ms. Lock retention stayed 57.78%; both had 0% link availability. The stable reference remains the validated default. Full values and confidence intervals are in `evidence/final_acceptance_snapshot.json`.

## UI and reproducibility

The five concise navigation areas are Live Demo, Experiment Lab, Evidence, Advanced, and About. Live Demo contains the interactive Three.js view, actual raw/annotated sensor feed, evaluation-only truth toggle, active pipeline, physical controls, error waterfall, PAT events, and optical consequence. The launcher builds the React UI and starts the FastAPI engine from the project-local environment.

## Limitations

This is a software/simulation PAT testbed: it tests algorithmic robustness, compares architectures, studies error propagation, and generates reproducible synthetic experiments. It is not flight-certified software, a complete optical-terminal hardware model, a universally validated atmospheric channel, or proof of real-world performance without hardware testing.

- Simplified atmosphere, image detector and Gaussian far-field optical link.
- Coarse-gimbal/camera emphasis; no separate fine-steering mirror, so tested FSOC link availability can be zero.
- Synthetic ML training data and limited cross-domain evidence.
- Scenario fidelity varies; UAV wind/turbulence are proxies, not flight-qualified Dryden/CFD models.
- BER appears only when communication noise is explicitly configured.
- No hardware-in-loop, thermal/structural qualification, calibration campaign, or flight validation.

## Technical debt

Before demo: keep the prebuilt frontend current; rehearse the six presets; use the stable default if optional ML dependencies are absent.

After demo: add UI filters for every cumulative-history field, CSV chart-data download buttons, and selective full-telemetry retention for worst failures.

Future research: fine-steering-mirror dynamics, hardware-in-loop calibration, validated atmospheric channels, real optical imagery, domain adaptation, and flight-computer profiling.
