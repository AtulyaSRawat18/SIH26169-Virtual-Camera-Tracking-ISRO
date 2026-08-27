# Technology stack and responsibilities

## Currently implemented

| Technology | Purpose | Role in this project |
|---|---|---|
| Python 3 | Main simulation language | Runs physics, image generation, detection, filtering, control and the API. |
| NumPy | Numerical computing | Stores vectors and matrices, integrates motion, applies coordinate transforms and performs Kalman calculations. |
| OpenCV | Computer vision and image formation | Generates the synthetic sensor frame, adds blur/noise, detects the bright target, measures its centroid and annotates the camera feed. |
| Clohessy-Wiltshire equations | Relative orbital dynamics | Produces nearby-satellite motion in the Hill frame around a circular reference orbit. |
| RK4 integration | Numerical propagation | Advances the CW differential equations with better stability than a simple Euler step. |
| Pinhole camera model | 3D-to-2D projection | Converts the target's relative 3D position and camera attitude into image pixels. |
| Kalman filter | State estimation | Smooths noisy OpenCV measurements and estimates image-plane position and velocity. |
| PID controller | Closed-loop tracking | Converts filtered pixel error into pan and tilt rate commands. |
| First-order actuator model | Camera response simulation | Adds finite response time and angular-rate limits instead of instantly rotating the camera. |
| FastAPI | Backend web framework | Exposes health, camera and telemetry endpoints and manages the live simulation lifecycle. |
| Uvicorn | ASGI application server | Runs FastAPI locally and handles HTTP and WebSocket connections. |
| WebSocket | Live structured telemetry | Sends target, camera, Kalman, PID and metric state to the dashboard about 20 times per second. |
| MJPEG | Live image transport | Streams the continuously annotated OpenCV camera frames to the browser. |
| React | Dashboard UI | Organizes the observer, camera feed, status and telemetry components. |
| TypeScript | Typed frontend development | Defines telemetry contracts and catches dashboard integration mistakes during builds. |
| Three.js | Web-based 3D graphics | Renders the spacecraft, target, coordinate system, line of sight, trajectory and field-of-view geometry. |
| React Three Fiber | React renderer for Three.js | Connects live React telemetry state to the Three.js scene. |
| Drei | Three.js interaction helpers | Provides orbit, zoom and pan controls plus reusable grid, line and star components. |
| Vite | Frontend build system | Develops and compiles the React/TypeScript dashboard into static browser files. |
| HTML/CSS | Presentation and layout | Defines the dashboard shell, responsive panels, metrics and visual styling. |
| JSON configuration | Scenario configuration | Stores camera, orbit, noise, Kalman, PID and time-acceleration parameters. |
| pytest | Automated verification | Checks CW propagation and verifies acquisition, detection accuracy and closed-loop lock. |
| Git and GitHub | Version control and collaboration | Preserves changes, supports member branches and distributes the reproducible project. |
| Windows CMD launcher | Repeatable local startup | Builds missing pieces, avoids PowerShell activation restrictions, opens the browser and starts the server. |

## Planned, not implemented yet

| Technology | Intended role |
|---|---|
| Custom YOLO model | Second visual detector for targets large enough to show a satellite body or identifiable structure. |
| PyTorch | YOLO/custom-model training, evaluation and experimental AI prediction. |
| ONNX Runtime | Optional portable and faster inference for the chosen trained model. |
| COCO/YOLO dataset tooling | Exports synthetic frames and labels for training and blind comparison. |

Gazebo, Isaac Sim, Unity, ROS and SciPy are not currently part of the running system.

