# Simulation and Rendering — Member 3

## Task

Convert ground truth into synthetic camera frames with beacon appearance and disturbances.

The Step 1 stationary demo is implemented in `stationary_demo.py`. It presents a fixed 3D world view beside the synthetic camera image and verifies that OpenCV can recover the projected beacon centre.

## Technologies

Python, NumPy and OpenCV.

## Track

Beacon intensity, spot size, noise, blur, jitter, occlusion, decoy count, frame ID and render latency.

## Run Step 1

From the repository root:

```powershell
python src/simulation/stationary_demo.py
```

Success means the green detection cross is centred on the beacon and the reported error is at most 2 pixels.
