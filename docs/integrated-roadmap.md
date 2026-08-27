# Integrated tracking baseline

## Goal alignment

| Capability | Why it belongs |
|---|---|
| Interactive Three.js observer | Lets judges inspect geometry without changing the physical tracking camera. |
| OpenCV camera feed | Demonstrates that measurements come from pixels rather than hidden simulator state. |
| Field-of-view and trajectory | Makes target visibility, geometry and relative motion explainable. |
| Kalman filter | Smooths noisy detections and estimates image-plane velocity. |
| PID pan/tilt | Closes the loop by driving the target toward the optical axis. |
| Clohessy-Wiltshire motion | Replaces arbitrary animation with credible nearby-satellite relative dynamics. |

All six capabilities support the software-in-the-loop optical-tracking goal. Three.js is a presentation and inspection layer; OpenCV, Kalman, PID and the physical model form the experiment.

## Integrated flow

`CW physics → camera projection → noisy pixels → OpenCV → Kalman → PID → actuator → camera attitude`

The CW clock is accelerated for demonstration. It can be slowed to real time through `configs/integrated.json`.

