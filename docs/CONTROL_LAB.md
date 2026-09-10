# Controller Laboratory (Prompt 7)

## Operational boundary

Every controller receives the same `ControllerInput`: estimated and predicted image state, covariance, image centre, reported gimbal angle/rate, actuator limits, focal length, time step, and timestamp. Simulator truth is intentionally absent. Commands are angular rates in rad/s and all variants pass through the same `GimbalPlant` with response lag, delay, deadband, quantization, and rate/acceleration/position limits.

## Runnable variants

| Controller | Role | Key implementation detail | Status |
|---|---|---|---|
| PID | Baseline feedback | Angular error, conditional-integration anti-windup, filtered derivative | Baseline |
| FF-PID | Predictive tracking | PID plus prediction velocity weighted by covariance | Runnable |
| Gain-scheduled PID | Regime adaptation | Smooth gains from LOS-rate and covariance | Experimental |
| LQR | State-feedback comparison | Discrete plant, documented Q/R, iterative Riccati solution | Experimental |
| MPC | Constraint-aware comparison | Finite-horizon dense quadratic program, command/rate/acceleration bounds, timeout fallback | Experimental |

LQR and MPC are compact educational implementations, not flight-qualified controllers. MPC falls back to PID or FF-PID for invalid numerics or a solver deadline overrun.

## Experiments and interpretation

- Same-input replay isolates controller behavior from vision and estimator variation.
- Angular step response reports rise time, settling time, overshoot, steady-state error, and peak command.
- Closed-loop matrices use common random-number seeds across nominal, vibration, latency, saturation, and dropout conditions.
- Sweeps support PID gains, FF gain, LQR effort weight, and MPC horizon/effort weight.
- The Pareto output identifies non-dominated tracking-error/control-effort choices; it does not claim a universal winner.

The dashboard Controller Lab exposes these modes. API routes are `/api/control/replay`, `/api/control/step-response`, `/api/control/benchmark`, and `/api/control/sweep`.

## Evidence status

The checked-in quick matrix is a smoke/architecture benchmark (five conditions, two seeds, 0.6 s runs). It validates fair execution and observability, not controller promotion. Its Pareto front contains FF-PID, LQR, and MPC; the short horizon and high early saturation make lock conclusions premature.

## Deliberate exclusions

This phase does not implement reacquisition policy, optical link budget, BER, fine steering mirrors, point-ahead mirrors, or hardware drivers. Those remain separate phase boundaries.
