# Scenario models

`configs/scenarios.json` is the versioned data source. Precedence is global
integrated defaults, scenario/environment profile, algorithm preset, then user
overrides. `VIEW RESOLVED CONFIG` in the UI exposes the final experiment input.

| Profile family | Motion | Platforms | Atmosphere default | Principal enabled errors |
|---|---|---|---|---|
| SAT_SAT_NOMINAL / STRESS | CW in Hill/LVLH | satellite to satellite, 400 km | Disabled | optional attitude, vibration, navigation, boresight, weak beacon |
| GROUND_SAT_* | relative kinematic pass | ground to satellite | Enabled | elevation-scaled attenuation, turbulence proxy, beam wander, sky/glare |
| UAV_GROUND_* | relative kinematic | ground to UAV | Enabled by environment | numerical altitude/speed/wind, rotor vibration, attitude/manoeuvre |
| UAV_UAV_* | relative kinematic, both mobile | UAV to UAV | Profile-dependent | vibration, attitude, manoeuvres, changing range |
| UAV_SAT_NOMINAL | relative kinematic pass | UAV to satellite | Enabled | UAV vibration, weak beacon, atmosphere, changing elevation proxy |

CW is retained only for nearby satellites on a circular reference orbit. Other
profiles use labelled relative kinematics and do not claim orbital physics. UAV
altitude, speed, angular rate and wind are stored as numerical platform fields.
Atmospheric airmass is capped and based on configured elevation; altitude does
not automatically imply stronger turbulence. Profile values are plausible demo
stress settings and require mission/hardware calibration before scientific use.
