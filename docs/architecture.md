# System Architecture

## Design goal

The system must let a team change one tracking stage without rewriting the simulator. Each operational stage consumes a small, explicit contract and produces a timestamped result with confidence, uncertainty and latency where applicable.

![Complete architecture](../assets/diagrams/system-architecture.svg)

## Operational loop

1. A scenario produces a true relative pose using CW propagation for nearby satellites or a kinematic model for ground/UAV cases.
2. The pinhole camera projects the target into image space and applies configured image/atmospheric disturbances.
3. OpenCV produces a beacon measurement and confidence without access to truth coordinates.
4. Optional CNN correction can adjust the measurement, but guard rails fall back to the classical result on high uncertainty, excessive latency or OOD input.
5. The estimator filters measurements and maintains position/velocity uncertainty through dropouts.
6. An optional predictor supplies a future image position to compensate processing and actuator delay.
7. PAT arbitration chooses tracking control or a search/reacquisition command.
8. PID/FF-PID commands a constrained pan–tilt actuator whose attitude feeds the next camera frame.

## Error cycle

![Closed-loop error cycle](../assets/diagrams/error-cycle.svg)

The controller does not minimize detector error directly. It minimizes displacement between the selected target estimate and image centre. Mechanical lag and saturation can therefore make closed-loop tracking error larger than localization error even when vision is accurate.

## Evaluation isolation

Projected truth is copied to a parallel evaluation path. It calculates localization, estimation, prediction and true pointing errors plus optical-link consequences. Truth never becomes a detector measurement, estimator input or camera command.

This separation prevents an easy but invalid simulation in which the controller is secretly given the correct target coordinates.

## Modular stages

| Stage | Stable/default | Selectable experiments | Output contract |
| --- | --- | --- | --- |
| Motion | scenario-specific kinematics/CW | analytical versus RK4 CW cases | relative position and velocity |
| Camera | pinhole projection | optics/disturbance profiles | frame and hidden projection truth |
| Vision | centroid baseline | binary, weighted, gradient, Gaussian fit | `(x, y)`, confidence, latency |
| Correction | none | CNN residual correction | corrected measurement, uncertainty |
| Estimator | Kalman | EKF, UKF, adaptive KF | state and covariance |
| Predictor | none | CV, CA, GRU, LSTM, residual GRU | future position and horizon |
| PAT | six-state arbitration | alternative gates and searches | state, reason and command mode |
| Controller | PID | FF-PID, gain scheduling, LQR/MPC labs | pan/tilt rate demand |
| Reacquisition | basic | raster, spiral, predicted, covariance-guided | search command |
| Actuator | constrained gimbal | limit/lag/deadband profiles | achieved pan/tilt attitude |

## Experiment promotion

![Evidence-gated promotion](../assets/diagrams/validation-cycle.svg)

A candidate module is not promoted based on one attractive run. It must improve paired seeded tests, held-out data, OOD cases and complete closed-loop performance while staying within latency and safety limits.

## Runtime boundaries

- `src/`: deterministic engineering core and experiment implementations
- `backend/`: configuration, experiment and WebSocket endpoints
- `frontend/`: live observer, controls, module labs and evidence views
- `configs/`: versioned scenario and pipeline definitions
- `models/`: small model checkpoints plus registry metadata
- `evidence/`: checked-in benchmark decisions, not mutable local run history

## Current recommendation

Use **centroid → Kalman → PID → constrained gimbal → basic reacquisition** for the primary demo. CNN correction and learned temporal prediction remain useful research comparisons, but current evidence does not justify making either the startup path.
