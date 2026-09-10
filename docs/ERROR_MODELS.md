# Error models

All values are resolved configuration, use SI/radian/pixel units, and have a
named deterministic RNG stream where randomness exists. Values affect the actual
camera/sensor pipeline; Three.js remains an observer. These are reproducible
software stress models, not qualification or flight models.

| Error | Physical origin | Applicability | Affected subsystem | Units | Current model | Fidelity now | Future fidelity |
|---|---|---|---|---|---|---|---|
| Initial state uncertainty | Navigation/orbit knowledge | All | True initial motion | m, m/s | Seeded Gaussian | Simplified | Covariance/ephemeris source |
| Navigation bias/noise | Reported state | All | Reported telemetry only | m, m/s | Bias + white noise + mismatch | Simplified | Correlated navigation filter |
| Attitude error | Platform reference | Sat/UAV/Ground | Physical optical axis | rad | bias, white/rate noise, drift | Simplified | measured PSD and attitude loop |
| Vibration | wheel/structure/rotor | Sat/UAV/Ground | Physical optical axis | rad, Hz | deterministic multi-sine + stochastic | Simplified | measured PSD/modal model |
| Gimbal error | Actuator | All | Camera axis/command | rad, rad/s | static bias, limits, delay, deadband, quantization | Intermediate | friction/backlash identification |
| Boresight | Optical assembly | All | Camera axis | rad | fixed pan/tilt offset | Simplified | thermal/alignment model |
| Point-ahead | propagation geometry | Sat links | Telemetry extension | rad | angle/estimate/error data only | Foundation | velocity and light-time model |
| Beacon variation | source/power | All | Rendered signal | intensity, px | flicker, attenuation windows, dropout, ellipse/radius | Simplified | radiometry and beam profile |
| Read/shot noise | detector | All | Pixel frame | intensity | Gaussian + signal-scaled approximation | Simplified | calibrated detector statistics |
| Exposure/quantization | camera | All | Pixel frame | gain, levels | gain, clipping, finite levels | Simplified | camera response curve |
| False sources | stars/beacons/clutter | All | Pixel frame | px, intensity | static/moving rendered spots | Simplified | scene/material model |
| Glare/background | sky/local source | Ground/UAV | Pixel frame | intensity | gradient + Gaussian glare region | Simplified | solar-angle stray-light model |
| Atmospheric attenuation | path loss/cloud | Ground/UAV links | Beacon intensity | factor | attenuation, cloud factor, capped airmass | Simplified | MODTRAN/weather data |
| Turbulence/beam wander | refractive path | Ground/UAV links | spot position/quality | rad, scalar | Gaussian wander + blur strength | Simplified | phase screen/Cn2 profile |
| Manoeuvre | target acceleration | All | True motion | m/s, m/s2 | scheduled impulse/pulse layer | Simplified | propagated force/orbit dynamics |
| Frame dropout | sensor/timing | All | Measurement availability | probability | deterministic Bernoulli stream | Simplified | measured timing/drop logs |

Failure attribution follows evidence priority: false target, explicit dropout,
out of FOV, weak signal, position/rate saturation, excessive vibration, excessive
pointing error, measurement failure, then UNKNOWN when evidence is insufficient.
Quantities in metres, radians and pixels remain separated inside error_budget.
