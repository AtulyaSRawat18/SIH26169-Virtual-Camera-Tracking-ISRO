# Experiment Guide

## Configure

Choose a scenario preset first. The resolver merges global defaults, the physical scenario, scenario-specific optical defaults, a registered pipeline, and user overrides in that order. Select only controls relevant to the active scenario; satellite-only runs hide ordinary clouds, while atmospheric and UAV controls appear where meaningful.

Select a pipeline stage by stage: Vision, Correction, Estimator, Predictor, Controller, and Reacquisition. The registry rejects unknown methods and invalid dependencies, such as CNN correction without its enabled model configuration. `DEFAULT_STABLE` is the safe startup pipeline. `CURRENT_BEST_VALIDATED` currently resolves to the same stable chain because alternatives failed the final promotion gate.

Use **View resolved configuration** before running. It includes physical values, algorithm settings/model IDs, seed, duration, optical parameters, and version fields. Use **Export rerunnable configuration JSON** to preserve it.

## Seeds and run modes

Master seed 42 is the default. Run index and subsystem name derive independent RNG streams for motion, vibration, camera noise, environment, dropout, distractors, and parameter sampling.

- **Live run:** reset and stream world/camera/telemetry.
- **Single headless:** one simulation without Three.js/DOM rendering.
- **Monte Carlo:** 100 runs by default; 1–10,000 supported.
- **Baseline vs proposed:** deterministic sequential pairs with common physical draws.
- **Parameter sweep:** one variable, multiple values, multiple seeds per point.
- **Replay:** load the stored resolved config, pair index, seed, and algorithm provenance.

Summary-only mode stores metrics and provenance without large frame telemetry. Use retained telemetry only for a representative failure or debugging run.

## Distributions

Eligible physical variables support:

```json
{"distribution":"fixed","value":0.000008}
{"distribution":"uniform","low":0.000005,"high":0.000012}
{"distribution":"normal","mean":0.000008,"std":0.000002}
```

For every run, the database stores the configured distribution, sampled effective value, parameter-sampling seed, complete resolved configuration, and subsystem seeds. Normal samples remain subject to configuration validation; choose distributions whose practical support remains physical.

## Read results

Continuous metrics report count, mean, median, standard deviation, extrema, p05, and p95. Paired comparisons also report mean/median paired difference, a seeded 2,000-resample 95% percentile-bootstrap interval, and paired `d_z` effect size when variance is non-zero. Direction semantics matter: lower is better for error/latency/effort; higher is better for lock/availability.

The live waterfall is sequential evaluation, not a claim of uniquely additive causation. Negative CNN or candidate performance is labelled `DEGRADED`. The prediction error is evaluated only when its intended future timestamp arrives. PAT lock and optical link availability are separate outcomes.

## Sweep and failure analysis

Select one principal parameter such as vibration, image noise, beacon intensity, beam wander, dropout, LOS rate, gimbal latency, UAV speed/altitude, or beam divergence. Every point runs multiple deterministic seeds and reports response plus confidence interval. Treat fitted sensitivity as local experimental response, not universal causation.

Failure analysis includes failed runs in denominators and retains worst-run seed, config, pipeline, metrics, and cause. Replay a compatible run by its run ID. Cumulative views exclude incompatible schema/physics versions unless explicitly requested.

## Export

- Dashboard: detailed current evidence JSON.
- `/api/evidence/export`: detailed JSON, per-run CSV, or summary-compatible CSV payloads.
- `/api/evidence/report`: print-ready HTML evidence report.
- `evidence/final_acceptance_snapshot.json`: frozen final paired evidence.
