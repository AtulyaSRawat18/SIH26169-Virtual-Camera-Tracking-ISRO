# PAT engineering contracts

## Frames and transformation chain

No inertial Earth orbit or spacecraft attitude propagation currently exists. CW
uses a rotating Hill/LVLH frame: radial x outward, along-track y, cross-track z.
The preserved illustrative world mapping is W = [Hill.y, Hill.z, range + Hill.x].
This includes a nominal radial offset that is NOT propagated by CW; it is a demo
placement convention, not a physically consistent second satellite orbit.
Platform/body nominal axes coincide with world axes. Explicit attitude bias,
drift, white/rate noise and vibration perturb the physical camera orientation.
Gimbal pan rotates toward +world x; positive tilt rotates toward +world y.
Camera basis is right, up, forward; coordinates are dot products with that basis.
Pixel u = width/2 + f*x_camera/z_camera, v = height/2 - f*y_camera/z_camera.
Pixels increase right/down. Points with depth <= 0.05 m have no valid projection.
Image capture uses pre-command gimbal attitude; the resulting command updates
the next capture. Telemetry distinguishes capture attitude and updated attitude.

## Units and timing

Position m; velocity m/s; angles rad; angular rates rad/s; time s; image positions
px; focal length px. Gaussian noise sigma is intensity levels (8-bit); blur is
the existing 7x7 Gaussian kernel. PID maps pixel errors to rad/s commands.
All stages execute once per tick (default 30 Hz). Camera, estimator and controller
dt = 1/fps. CW dt = time_scale/fps (default 60x). Thus this is an accelerated
orbital demo with unaccelerated control, not a common physical-time experiment.
First capture timestamps are one dt, after the first motion integration.

## Truth boundary and state

Motion/projected truth is available only to rendering, visualization and metrics.
Vision receives pixels and timestamp; estimator receives Measurement; predictor
receives TrackingState; controller receives that state and camera/gimbal data.
TrackingState uses image x/y (u/v), vx/vy in px/control-second, timestamp in
control seconds, validity, optional acceleration, covariance and confidence.
None predictor passes through current state and reports no future prediction.
Kalman prediction on missing measurements remains part of estimation.

## Metrics and lock

Existing tracking_error_px remains estimated position to centre. Run tracking
metrics use TRUE projected position to centre so bad estimates cannot improve
the score. CV localization error compares measured and true pixels. Invalid
truth is excluded from error aggregates; dropout and lock include every tick.
Lock uses measured visibility and estimated error only, never truth. States are
SEARCH, ACQUIRE, TRACK, LOCKED, LOST and REACQUIRE. Configured dwell counts and
loss thresholds govern transitions. A short missing interval can preserve LOCKED;
max_missing_frames or excessive pointing error moves it to LOST.
Handoff threshold is a configurable simulation criterion, not hardware evidence.
Metrics retain a bounded recent-frame buffer and whole-run summary arrays.
