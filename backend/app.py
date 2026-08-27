"""Live API for simulation telemetry and camera frames."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.core.engine import ROOT, SimulationEngine


engine = SimulationEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="SIH26169 Virtual Camera Tracking", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    telemetry = engine.telemetry_snapshot()
    return {"status": "ok", "frame": telemetry.get("frame", 0)}


def mjpeg_stream():
    while True:
        jpeg = engine.jpeg_snapshot()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / float(engine.config["fps"]))


@app.get("/api/camera.mjpeg")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            telemetry = engine.telemetry_snapshot()
            if telemetry:
                await websocket.send_json(telemetry)
            await asyncio.sleep(1.0 / 20.0)
    except WebSocketDisconnect:
        return


frontend_dist = Path(ROOT) / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
