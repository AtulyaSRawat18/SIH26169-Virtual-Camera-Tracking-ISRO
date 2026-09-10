# Experiment architecture

## Repository inspection and minimum refactor

The original src/core/engine.py owned CW advancement, projection, rendering,
centroid, KF, PID, actuator, annotations and metrics. backend/app.py owns FastAPI,
the simulation thread lifecycle, MJPEG and a 20 Hz telemetry WebSocket. React /
TypeScript uses useTelemetry.ts; ObserverScene.tsx owns Three.js through React
Three Fiber/Drei. Vite builds static assets served by FastAPI. integrated.json
supplies initial numerical settings. Existing CW/KF/PID/centroid math is retained.

## Pipeline

```text
JSON / UI -> validate_config -> immutable ExperimentConfig -> registry factories
                                                       |
CW motion -> attitude disturbance -> pinhole camera -> image disturbance
                                                       |
                    centroid -> estimator -> predictor -> controller
                                                          |
                    next capture <- pan/tilt actuator <----+

truth + measurements + states + commands -> metrics/lock -> MJPEG + WebSocket
```

Truth only enters camera rendering and evaluation. Lock receives estimated error
and measurement validity. The controller receives a TrackingState, camera centre
and GimbalState. NoPrediction passes through the estimate; predicted_pixel is null.
The Kalman adapter retains prediction during missed detections. No reacquisition
returns no override, preserving existing coasting control during loss.

## Files and interfaces

- core/config.py: frozen Pydantic configuration, numerical/feature validation.
- core/contracts.py: PhysicalState, Measurement, TrackingState, GimbalState,
  Command and Protocols for MotionModel, DisturbanceModel, VisionTracker,
  StateEstimator, Predictor, Controller, ReacquisitionStrategy.
- core/stages.py: current adapters; internal state/history is owned by each stage.
- core/registry.py: Stage enum, REGISTRY, validate_config, build_stages,
  capabilities. Algorithms use names only at this configuration boundary.
- core/camera.py: projection, synthetic frame generation and annotation.
- core/metrics.py: PATStateMachine, FailureCause and Metrics.
- core/errors.py: physical, beacon, environment and detector error system.
- core/randomness.py: independent deterministic subsystem streams.
- core/scenarios.py and configs/scenarios.json: precedence and versioned profiles.
- control/gimbal.py: preserved actuator plus bias, command delay and saturation evidence.
- experiments/monte_carlo.py and storage.py: headless batches, paired comparison and SQLite history.
- core/engine.py: SimulationEngine orchestrates and synchronizes lifecycle.
- physics/cw.py, tracking/kalman.py, control/pid.py and
  simulation/stationary_demo.py: preserved mathematical implementations.
- frontend/src/ExperimentPanel.tsx: capabilities-driven configuration form.

## Configuration and lifecycle

Legacy integrated.json is validated and expanded with explicit defaults. The
immutable ExperimentConfig is authoritative; GET /api/experiment returns its
serialized snapshot and registered options. POST /api/experiment/reset accepts
a complete config and returns HTTP 422 with an explanation for invalid selections.
Invalid configuration does not interrupt the active run. Fresh stages are built
before swapping under the same lock used for stepping. Reset recreates camera,
actuator, algorithms, RNG, trajectory, lock and metrics and clears stale snapshots.
The existing thread continues; no duplicate producer is launched. UI options are
derived from registry capabilities; unavailable algorithms have no fake buttons.

## Metrics and reproducibility

Each reset creates stable subsystem seeds from SHA-256(master seed, run index,
subsystem name). Motion, attitude, vibration, environment, beacon, camera noise,
dropout, distractor and manoeuvre streams are independent. Algorithm changes do
not consume environmental streams. Same config/seed reproduces physical inputs;
runtime latency remains observational and therefore is not deterministic.

Metrics keep 1,200 recent records plus whole-run accumulators.
GET /api/experiment/metrics exports that buffer and summary. Invalid projected
truth does not count as zero error. Lock percentage is a fraction of equal-duration
ticks; reset begins a new run. See PAT_SYSTEM_SPEC.md for units and error semantics.
Headless batches instantiate the same engine and skip annotation, JPEG and web
publication. SQLite stores batches and individual summaries; optional telemetry
retention supports selected/failed runs. Compatibility filters exclude a changed
schema or physics model unless cross-version comparison is requested.

## Extension recipe

Implement the appropriate Protocol in a new module, with a (config, rng) factory.
Register the name in REGISTRY for that Stage. Add typed config fields/model-asset
validation if required, tests, and UI parameter inputs only where needed. Existing
algorithm dropdown options are populated automatically from capabilities.

| Addition | Contract and registration |
|---|---|
| EKF, UKF, AKF | StateEstimator.update; Stage.ESTIMATOR |
| CNN correction | VisionTracker.measure; Stage.VISION; validate available model |
| GRU | Predictor.predict; Stage.PREDICTOR; own sequence history/model |
| Feed-forward PID, LQR | Controller.compute; Stage.CONTROLLER |
| Search/reacquisition | ReacquisitionStrategy.search; Stage.REACQUISITION |

For example UKF: implement update, register ukf, add its configuration validation
and tests. Camera, CW, PID and dashboard need no estimator-specific edits.
Do not enable cnn_correction until its adapter and asset validation exist.

## Known debt before AI

- Orbital and control clocks differ by time_scale; record this in comparisons.
- The nominal range offset and synthetic camera mapping are illustrative; rectify
  relative geometry before claiming a physically consistent satellite link.
- Current image-plane KF ignores camera ego-motion; basic reacquisition is a
  deterministic scan rather than uncertainty-aware search.
- Atmosphere, turbulence, glare, clouds, vibration and attitude effects are
  simplified stress models, not flight-grade optics or PSD-calibrated hardware.
- Centroid uses contour moments with hard-coded area limits (20..2000 px^2).
- Three.js FOV Euler composition and capture/actuation timing need a dedicated
  geometry alignment pass. MJPEG and telemetry are separate unsynchronized feeds.
- Existing scene React updates and per-consumer JPEG encoding remain; no new
  high-frequency React state path was added. Vite's large Three.js bundle is
  existing build debt. Full traces need a streaming recorder, not an unbounded list.

## Verification (2026-09-09)

- Prompt 1 verification: `python -m pytest -q`: 19 passed, including original CW/closed-loop tests,
  reproducibility, detector/filter/controller contracts, actuator limits, invalid
  config rejection, known-value RMSE, lock dwell/loss/recovery, passthrough mode,
  truth isolation, HTTP reset, WebSocket telemetry and MJPEG JPEG bytes.
- `npm run build`: TypeScript and Vite passed; existing large-bundle warning remains.
- 210-frame comparison against the pre-refactor HEAD engine: largest difference
  across detector/Kalman pixels and pan/tilt was 0.0005, within old telemetry's
  decimal rounding. No mathematical implementation was changed.
- Browser verified live scene/camera, None estimator reset, then Kalman reset.
- `git diff --check`: passed.
