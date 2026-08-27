# React Dashboard — Member 5

## Task

Display camera/world views, controls, live telemetry, graphs and the classical-versus-AI switch.

The current dashboard provides an orbitable Three.js observer, field-of-view cone, target trajectory, annotated OpenCV feed and live tracking metrics.

## Technologies

React, TypeScript, Vite and Plotly.js.

## Track

Connection state, render time, displayed FPS, active scenario, active model and frontend errors.

## Build

```powershell
npm install
npm run build
```

After building, FastAPI serves the dashboard at `http://127.0.0.1:8000`.
