# FastAPI Backend — Member 5

## Task

Expose simulation controls, configuration, telemetry, camera frames and evidence export.

`app.py` now runs the integrated physics/tracking engine, streams telemetry over WebSocket and serves the annotated OpenCV camera as MJPEG.

## Technologies

FastAPI, Uvicorn, Pydantic and WebSocket.

## Track

Backend health, simulation status, telemetry rate, stream FPS, WebSocket latency and dropped frames.

## Run

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
