# Prompt 8 verification

Status: **complete**

- [x] Explicit SEARCH, ACQUIRE, TRACK, LOCKED, LOST, and REACQUIRE state machine retained.
- [x] Truth-free common search input and command contract.
- [x] Hold, last-known, raster, spiral, predicted-point, covariance, predictive-covariance, and hybrid strategies.
- [x] Stale/invalid prediction and missing-covariance fallbacks are explicit.
- [x] All search commands use the same rate/acceleration/delay/deadband/position-limited gimbal plant.
- [x] Confidence and innovation confirmation gate reduces false-lock acceptance without simulator truth.
- [x] Search-to-track handoff resets controller memory and preserves physical actuator state.
- [x] Phase, strategy, fallback, effort, duration, transitions, reacquisition, false-lock, and failure metrics recorded.
- [x] Same-input replay, common-seed closed-loop matrix, and parameter sweep are runnable.
- [x] API, Acquisition Lab UI, persistent lab results, docs, tests, and evidence exist.

Evidence: `evidence/acquisition_benchmark_summary.json`; method: `docs/ACQUISITION_LAB.md`; focused validation: 10 tests passed with CW/integration coverage.

No strategy is promoted from the quick smoke matrix.
