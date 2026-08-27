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

## Step 1 — stationary baseline

This first runnable setup uses a stationary 3D target, a fixed pinhole camera and OpenCV bright-blob detection. No pretrained model is needed for this baseline because the target is a beacon, not a standard object class.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/simulation/stationary_demo.py
```

The demo displays the 3D world and the virtual camera side by side, saves `evidence/stationary_step1.png`, and fails if detection error exceeds 2 pixels.

## Step 2 — live simulation

```powershell
python src/simulation/live_demo.py
```

The target moves continuously in 3D and OpenCV detects it from each noisy virtual-camera frame. Press `Space` to pause, `R` to reset, and `Q` or `Esc` to exit. Edit `configs/live.json` to change FPS, path amplitude or motion frequency.
