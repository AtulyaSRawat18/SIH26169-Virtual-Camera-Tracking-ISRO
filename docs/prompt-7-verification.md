# Prompt 7 verification

Status: **complete**

## Mandatory implementation checklist

- [x] Common `ControllerInput` and `ControlCommand` contracts with explicit radians and seconds.
- [x] Pixel-to-angle conversion uses camera intrinsics and the established pan/tilt signs.
- [x] PID baseline remains configuration driven, bounded and resettable.
- [x] PID conditional anti-windup, filtered derivative and optional estimator-rate derivative.
- [x] FF-PID has bounded uncertainty-weighted feed-forward and safe invalid-prediction behavior.
- [x] Gain-scheduled PID has explicit stable regions and smooth scheduling.
- [x] LQR gain comes from a documented Riccati solution for the shared actuator model.
- [x] MPC uses a small constrained receding horizon, records solver failures/timeouts and falls back safely.
- [x] PID, FF-PID, GS-PID, LQR and MPC drive the same `GimbalPlant`.
- [x] Controller latency, deadline misses, fallback, saturation, effort and command smoothness are recorded.
- [x] Controller reset and switching preserve physical gimbal state.
- [x] Fixed-sequence open-loop replay and full closed-loop paired comparison exist.
- [x] Static-step, scenario matrix, latency, actuator-limit and prediction-quality experiments exist.
- [x] Parameter sweeps and Pareto/regime summaries exist without a forced universal winner.
- [x] Controller metadata persists in cumulative lab/run records.
- [x] Controller Lab UI exposes the selected controller and controller-specific diagnostics/actions.
- [x] `DEFAULT_STABLE` remains PID and all existing pipelines remain runnable.
- [x] Measured evidence is exported; no controller is promoted without it.
- [x] Focused and full regression tests pass.

## Explicit exclusions

- Reinforcement learning and end-to-end neural control.
- New acquisition/reacquisition strategies; these belong to Prompt 8.
- Complete optical BER/link modelling; this belongs to Prompt 9.

## Implementation plan

1. Extend contracts and validated configuration.
2. Implement controller family and common adapter.
3. Integrate the engine, registry, metrics and persistence.
4. Add replay, closed-loop benchmark, sweeps and evidence.
5. Add API/UI laboratory and documentation.
6. Run focused numerical, fairness and regression tests.

## Evidence

- Implementation: `src/control/controllers.py`, `src/experiments/controller_lab.py`
- API/UI: `backend/app.py`, `frontend/src/ControllerLab.tsx`
- Method: `docs/CONTROL_LAB.md`
- Snapshot: `evidence/controller_benchmark_summary.json`
- Tests: `8 passed` focused; `71 passed` full regression on 2026-09-10.
