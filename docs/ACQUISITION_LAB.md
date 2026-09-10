# Acquisition and Reacquisition Laboratory (Prompt 8)

## State and handoff model

`SEARCH → ACQUIRE → TRACK → LOCKED → LOST → REACQUIRE → TRACK` is the sole PAT state machine. A measurement must satisfy availability, minimum confidence, and—during acquisition/reacquisition confirmation—the estimator innovation gate. The controller owns TRACK/LOCKED commands; the selected search strategy owns commands only while SEARCH/LOST/REACQUIRE has no confirmed measurement. Re-entry to TRACK resets controller memory while preserving physical gimbal state.

## Truth-free strategy contract

`SearchInput` contains only the last accepted state, predicted state, covariance, reported gimbal state, actuator limits, image centre/intrinsics, elapsed time, time step, timestamp, and PAT state. Ground truth is not present. Every strategy emits angular-rate commands into the same `GimbalPlant` used by normal tracking.

| Strategy | Intended regime | Fallback |
|---|---|---|
| Hold | Very short interruption | None |
| Last known | Short dropout with a valid prior track | Hold |
| Raster | Broad deterministic coverage | Bounded scan |
| Spiral | Compact local coverage | Bounded scan |
| Predicted point | Fresh trajectory prediction | Last known |
| Covariance search | Anisotropic estimator uncertainty | Spiral |
| Predictive covariance | Fresh prediction plus uncertainty | Covariance-only |
| Hybrid | Hold → last-known → predictive/covariance → raster | Phase-specific |

Search rate, phase timing, hold/last-known durations, covariance scaling, prediction age, confirmation confidence, and innovation gate are validated configuration fields. Search phase, sub-strategy, fallback, effort, duration, false-lock outcome, and PAT transition are recorded per run.

## Experiments

- Same-input replay compares command geometry without a changing vision stream.
- The closed-loop common-seed matrix covers initial pointing offset, short/long dropout, uncertain prediction, and distractors.
- Scan-rate and timing/covariance/gate parameters are sweepable.
- Results are evaluated by time to lock, reacquisition time/success, lock retention, false-lock probability, effort, duration, fallback, saturation, and failure cause.

The quick evidence matrix is an architecture smoke benchmark, not a promotion run. Raster improved mean lock occupancy slightly in the injected initial-offset case but took longer to first lock overall; hold spent no search effort. A longer regime-balanced study is required before choosing a default. The production-compatible `basic` name maps to the documented hybrid strategy.

API routes: `/api/acquisition/replay`, `/api/acquisition/benchmark`, `/api/acquisition/sweep`. The dashboard Acquisition Lab exposes the same operations.
