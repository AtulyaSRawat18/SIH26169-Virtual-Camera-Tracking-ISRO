# Optical Link Consequence Model (Prompt 9)

## Scope and causality

This is a configurable engineering consequence model, not a flight-grade link simulator. Tracking remains `camera → measurement → estimator → predictor → controller → gimbal`. The optical model observes the true scenario LOS and the actual disturbed optical axis at the exposure time; it never feeds received power back into the controller. The existing beacon/environment path remains the only camera-strength coupling.

## Pointing geometry

The virtual camera optical axis uses the same pan/tilt convention as `VirtualCamera.basis`. For scenario vector `(X,Y,Z)`:

`theta_LOS_pan = atan2(X,Z)`

`theta_LOS_tilt = atan2(Y,sqrt(X²+Z²))`

Axis differences are wrapped to `[-π,π]`; total error is `theta_e = sqrt(delta_pan² + delta_tilt²)`. Radians are stored; the UI displays µrad. Image-domain errors remain separate and any pixel/angle visualization uses the configured pinhole focal length, never a fixed conversion constant.

## Beam and power convention

The simplified far-field radius is `w(R) = R theta_div`, where `w` is explicitly the **1/e² intensity radius** and `theta_div` is the configured half-angle divergence. This approximation omits waist/Rayleigh-range propagation.

For receiver aperture radius `a`, the on-axis circular-aperture capture approximation is:

`L_geometric = 1 - exp(-2 a² / w²)`

Lateral offset is `r_offset = R theta_e`, giving:

`L_point = exp(-2 r_offset² / w²)`

The unpointed and received powers are:

`P_rx_ideal = P_tx L_geometric eta_tx eta_rx L_atmosphere`

`P_rx = P_rx_ideal L_point`

Every factor is retained separately in telemetry. Loss is `-10 log10(L)` dB, power is `10 log10(P_W / 1 mW)` dBm, and margin is `P_rx,dBm - sensitivity_dBm`.

## Range and atmosphere

`AUTO_FROM_SCENARIO` uses the current physical relative-vector norm; isolated sweeps can select `MANUAL_OVERRIDE`. Satellite-satellite defaults have no atmospheric attenuation. Ground and UAV presets use the existing scenario atmosphere factor (including elevation-scaled attenuation/cloud factor) exactly once. Beam wander enters the actual optical axis/image geometry and is not also charged as an arbitrary power penalty.

Defaults differ across satellite-satellite, ground-satellite, UAV-ground, UAV-UAV, and UAV-satellite scenarios for transmit power, divergence, aperture, efficiencies, and sensitivity. All values resolve into the experiment configuration and remain user-overridable.

## Link state and SNR/BER limits

`link_available` is the configured received-power threshold test and is independent of PAT lock. Labels are `NO_LINK`, `MARGINAL`, `LINK_AVAILABLE`, and `GOOD_MARGIN`, using configurable margin thresholds. Telemetry preserves PAT locked/link available, PAT locked/link unavailable, PAT not locked/link unavailable, and PAT reacquiring combinations.

Communication SNR is `NOT_CONFIGURED` unless an explicit communication noise power is supplied; when supplied it is labelled approximate. It is never copied from image SNR. BER remains `NOT_CONFIGURED` because modulation, bandwidth, receiver/detection model, and noise statistics are not specified.

## Experiments and known simplifications

The lab provides pointing-error, divergence, range, and simplified ground-satellite elevation sweeps plus a paired baseline/AI-candidate link comparison. Common subsystem seeds are used. The AI candidate remains experimental because Prompts 5–6 did not clear promotion gates.

The model excludes diffraction near-field/waist evolution, aperture misalignment integration, scintillation statistics, receiver electronics, coding/modulation/BER, terminal fine-steering mirrors, and hardware calibration. Very large pointing errors intentionally drive power to a numerical floor; this honestly shows that coarse camera pointing alone may not close a narrow-beam data link.
