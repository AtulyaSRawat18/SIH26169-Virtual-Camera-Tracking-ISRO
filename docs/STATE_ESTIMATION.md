# State estimation laboratory

All estimators consume only `SpotMeasurement`, time step and configured camera calibration. Simulator truth is used after estimation for metrics and is never passed to an estimator.

## Shared output

`TrackingState` exposes validity, native state/covariance, image position/velocity, optional angular position/velocity, predicted measurement, innovation/covariance, measurement-used and prediction-only flags, confidence, estimator identity, image-space covariance/95% ellipse, NIS, current Q/R, adaptive-R scale, numerical events and update latency. PID consumes only the common image position.

## KF-CV: linear image-space model

State is `x=[u,v,u_dot,v_dot]^T`. For time step `dt`:

```text
F = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]
H = [[1,0,0,0], [0,1,0,0]]
z = [u,v]^T
```

For independent white acceleration spectral density `q`, each axis uses `q[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]` in the matching position/velocity slots. `R=rI`. The first valid measurement initializes position, zero velocity and configured covariance. The Joseph covariance update is used.

`none` passes valid measurements directly and becomes invalid on missing input. `kalman` is a compatibility alias of `kf_cv`.

## EKF/UKF: nonlinear angular camera model

State is `x=[theta_x,theta_y,omega_x,omega_y]^T` in radians, with the same constant-rate transition layout. The estimator convention makes `theta_y` positive down, matching image `v`; this is intentionally distinct from the camera renderer’s world-up projection step.

Central transforms are:

`theta_x=atan((u-cx)/fx)`, `theta_y=atan((v-cy)/fy)`

`u=cx+fx tan(theta_x)`, `v=cy+fy tan(theta_y)`.

The EKF analytic measurement Jacobian is:

```text
H = [[fx sec²(theta_x), 0, 0, 0],
     [0, fy sec²(theta_y), 0, 0]]
```

The UKF uses this same genuinely nonlinear measurement function. Defaults are `alpha=0.3`, `beta=2`, `kappa=0`, all serialized in configuration. It forms `2n+1` sigma points using `lambda=alpha²(n+kappa)-n`, propagates them through the constant-angular-rate process and tangent camera function, and reconstructs measurement/cross covariances. Covariance is symmetrized; Cholesky jitter is added only after failure and recorded as `CHOLESKY_RECOVERY`.

## AKF-R

AKF-R retains the linear pixel state and adapts measurement noise conservatively. From real Prompt 3 features:

```text
q_quality = clamp(
  .35 confidence + .30 SNR/(SNR+5) + .15/(1+fit_RMSE/20)
  + .10 unclipped + .05 unsaturated + .05 unambiguous,
  .05, 1)
s = clamp(1/q_quality², s_min, s_max)
R_k = s R_base
```

When a localizer supplies a valid 2×2 covariance, its eigenvalues are bounded to configured measurement limits and `R_k = R_floor + bounded(R_measurement)`; the quality scale is still exposed. Adaptive Q is deliberately not hidden inside AKF-R.

## Gating, dropout and reset

Innovation is `nu=z-h(x-)`, innovation covariance is `S`, and `NIS=nu^T S^-1 nu`. Optional gating rejects an update when NIS exceeds the configured threshold (`9.210`, the 99% chi-square threshold for two measured dimensions). Gating is off in `DEFAULT_STABLE` to preserve the pre-existing baseline and can be enabled explicitly for outlier experiments.

KF/EKF/UKF/AKF predict without measurement and covariance grows through Q. `none` becomes invalid. After the configured maximum missing frames, a filter emits `RESET_REQUIRED` and invalidates itself; it never consumes the known dropout endpoint or true velocity.

Numerical telemetry records `CHOLESKY_RECOVERY`, `NON_POSITIVE_COVARIANCE`, `SINGULAR_INNOVATION`, `INVALID_STATE`, `INVALID_MEASUREMENT`, `ANGLE_OUTSIDE_MODEL_DOMAIN` and `RESET_REQUIRED`. Large angles are rejected before tangent singularities.

For image filters, the 95% ellipse comes from the `[u,v]` covariance. For angular filters, `H_theta P_theta H_theta^T` is used. Ellipse axes use chi-square factor 5.991. It is a probabilistic visualization, not a guarantee.

Known limits: the angular process is constant rate, not an orbital-navigation state; focal length/calibration is ideal unless mismatch is injected; image NEES uses only the two-position block; and Gaussian/linearity assumptions can fail in multimodal distractor cases.
