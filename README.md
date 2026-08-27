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

## Integrated browser dashboard

This mode adds interactive Three.js inspection, Clohessy-Wiltshire relative motion, Kalman filtering and PID-driven camera pan/tilt.

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Drag the observer view to orbit, scroll to zoom and right-drag to pan. The observer camera is independent of the simulated tracking camera.

## One-command launch on Windows

Double-click `launch_simulation.cmd`, or run it from PowerShell:

```powershell
.\launch_simulation.cmd
```

The launcher uses the project-local virtual environment directly, builds the dashboard, opens the browser and starts the server. To stop it, press `Ctrl+C` in its terminal and then `Y` if Windows asks for confirmation.

See `docs/technology-stack.md` for the technologies currently used, their purposes and the planned YOLO/PyTorch layer.
