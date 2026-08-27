# SIH26169 — AI Virtual Camera Tracking

Minimal project structure for Team Doomsday Squad's software-in-the-loop optical tracking demo.

No implementation, datasets, model weights, generated evidence, or secrets are committed yet.

## Planned flow

`Physics → Synthetic camera → OpenCV detection → Kalman/AI estimate → PID control → Web dashboard`

## Main measurements

- Tracking error, acquisition time, lock retention and reacquisition time
- Detection confidence, false detections and processing latency
- Simulation, computer-vision, AI and dashboard frame rates
- CPU, memory and model-inference usage

Each folder contains a short README defining its task, technology and measurable parameters.
