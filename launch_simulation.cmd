@echo off
setlocal
cd /d "%~dp0"
title SIH26169 Virtual Camera Tracking

set "SIM_PYTHON=.venv\Scripts\python.exe"
set "SIM_URL=http://127.0.0.1:8000"

echo.
echo [SIH26169] Preparing the virtual camera tracking simulation...

if not exist "%SIM_PYTHON%" (
  echo [SETUP] Creating the isolated Python environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)

"%SIM_PYTHON%" -c "import cv2, fastapi, numpy, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing Python dependencies...
  "%SIM_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Node.js and npm are required for the Three.js dashboard.
  goto :error
)

pushd frontend
if not exist "node_modules" (
  echo [SETUP] Installing dashboard dependencies...
  call npm install --no-audit --no-fund
  if errorlevel 1 goto :frontend_error
)

echo [BUILD] Building the latest dashboard...
call npm run build
if errorlevel 1 goto :frontend_error
popd

echo [READY] Starting FastAPI, OpenCV, orbital physics and the dashboard.
echo [READY] URL: %SIM_URL%
echo [READY] To stop: press Ctrl+C here, then press Y if Windows asks.
echo.

if /I not "%~1"=="--no-browser" (
  start "" /b "%~dp0.venv\Scripts\pythonw.exe" -c "import time, webbrowser; time.sleep(2); webbrowser.open('%SIM_URL%')"
)

"%SIM_PYTHON%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%

:frontend_error
popd

:error
echo.
echo [FAILED] The simulation could not start. Review the message above.
pause
exit /b 1
