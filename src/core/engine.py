"""Integrated optical-tracking simulation engine."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.control.pid import PIDAxis, PanTiltActuator
from src.physics.cw import ClohessyWiltshire
from src.simulation.stationary_demo import detect_beacon
from src.tracking.kalman import ImageKalmanFilter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "integrated.json"


class SimulationEngine:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        with config_path.open("r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_telemetry: dict = {}
        self.trajectory: deque[list[float]] = deque(maxlen=300)
        self.frame_number = 0
        self.reset()

    def reset(self) -> None:
        orbit = self.config["orbit"]
        orbital_radius = float(orbit["earth_radius_m"]) + float(orbit["altitude_m"])
        mean_motion = math.sqrt(float(orbit["earth_mu_m3_s2"]) / orbital_radius**3)
        self.physics = ClohessyWiltshire(
            mean_motion_rad_s=mean_motion,
            position_m=np.asarray(orbit["initial_hill_position_m"], dtype=float),
            velocity_m_s=np.asarray(orbit["initial_hill_velocity_m_s"], dtype=float),
        )
        kalman = self.config["kalman"]
        self.kalman = ImageKalmanFilter(kalman["process_noise"], kalman["measurement_noise"])
        pid = self.config["pid"]
        self.pan_pid = PIDAxis(pid["kp"], pid["ki"], pid["kd"], pid["max_rate_rad_s"], pid["integral_limit"])
        self.tilt_pid = PIDAxis(pid["kp"], pid["ki"], pid["kd"], pid["max_rate_rad_s"], pid["integral_limit"])
        self.actuator = PanTiltActuator(pid["actuator_response_time_s"], pid["max_rate_rad_s"])
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self.trajectory.clear()
        self.frame_number = 0

    @staticmethod
    def camera_basis(pan_rad: float, tilt_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array(
            [math.sin(pan_rad) * math.cos(tilt_rad), math.sin(tilt_rad), math.cos(pan_rad) * math.cos(tilt_rad)],
            dtype=float,
        )
        right = np.array([math.cos(pan_rad), 0.0, -math.sin(pan_rad)], dtype=float)
        up = np.cross(forward, right)
        return right, up, forward

    def project_target(self, target_world_m: np.ndarray) -> tuple[float, float, float]:
        camera = self.config["camera"]
        right, up, forward = self.camera_basis(self.actuator.pan_rad, self.actuator.tilt_rad)
        x_camera = float(target_world_m @ right)
        y_camera = float(target_world_m @ up)
        z_camera = float(target_world_m @ forward)
        if z_camera <= 0.05:
            return float("nan"), float("nan"), z_camera
        focal = float(camera["focal_length_px"])
        u = camera["width_px"] / 2.0 + focal * x_camera / z_camera
        v = camera["height_px"] / 2.0 - focal * y_camera / z_camera
        return float(u), float(v), z_camera

    def render_camera(self, projected: tuple[float, float, float]) -> np.ndarray:
        camera = self.config["camera"]
        width, height = int(camera["width_px"]), int(camera["height_px"])
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Dim stars are visual disturbances but remain below the beacon threshold.
        star_x = self.rng.integers(0, width, 45)
        star_y = self.rng.integers(0, height, 45)
        star_brightness = self.rng.integers(25, 95, 45)
        frame[star_y, star_x] = np.column_stack((star_brightness, star_brightness, star_brightness))

        u, v, z_camera = projected
        if z_camera > 0 and -20 <= u < width + 20 and -20 <= v < height + 20:
            cv2.circle(
                frame,
                (int(round(u)), int(round(v))),
                int(camera["beacon_radius_px"]),
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        noise = self.rng.normal(0.0, float(camera["noise_sigma"]), frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def step(self) -> tuple[np.ndarray, dict]:
        fps = float(self.config["fps"])
        control_dt = 1.0 / fps
        physics_dt = control_dt * float(self.config["time_scale"])
        hill_position, hill_velocity = self.physics.step(physics_dt)
        orbit = self.config["orbit"]
        target_world = np.array(
            [hill_position[1], hill_position[2], float(orbit["nominal_camera_range_m"]) + hill_position[0]],
            dtype=float,
        )
        self.trajectory.append(target_world.tolist())

        projected = self.project_target(target_world)
        frame = self.render_camera(projected)
        measurement = detect_beacon(frame, int(self.config["camera"]["threshold"]))
        estimate = self.kalman.step(measurement, control_dt)

        camera = self.config["camera"]
        centre_u, centre_v = camera["width_px"] / 2.0, camera["height_px"] / 2.0
        if estimate is not None:
            pan_command = self.pan_pid.step(float(estimate[0] - centre_u), control_dt)
            tilt_command = self.tilt_pid.step(float(centre_v - estimate[1]), control_dt)
        else:
            pan_command = tilt_command = 0.0
        self.actuator.step(pan_command, tilt_command, control_dt)

        if measurement is not None:
            cv2.drawMarker(frame, tuple(np.rint(measurement).astype(int)), (0, 255, 90), cv2.MARKER_CROSS, 22, 2)
        if estimate is not None:
            cv2.circle(frame, tuple(np.rint(estimate[:2]).astype(int)), 12, (255, 210, 40), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(centre_u), int(centre_v)), (50, 80, 255), cv2.MARKER_CROSS, 28, 1)
        cv2.putText(frame, "green: OpenCV  cyan: Kalman  red: optical axis", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1, cv2.LINE_AA)

        ground_truth = None if not np.isfinite(projected[0]) else [projected[0], projected[1]]
        detection_error = None
        if measurement is not None and ground_truth is not None:
            detection_error = float(np.linalg.norm(np.asarray(measurement) - np.asarray(ground_truth)))
        tracking_error = None if estimate is None else float(np.linalg.norm(estimate[:2] - [centre_u, centre_v]))
        field_of_view = [
            math.degrees(2.0 * math.atan(camera["width_px"] / (2.0 * camera["focal_length_px"]))),
            math.degrees(2.0 * math.atan(camera["height_px"] / (2.0 * camera["focal_length_px"]))),
        ]
        telemetry = {
            "frame": self.frame_number,
            "simulation_time_s": round(self.frame_number * physics_dt, 3),
            "target_world_m": [round(float(value), 5) for value in target_world],
            "target_hill_position_m": [round(float(value), 5) for value in hill_position],
            "target_hill_velocity_m_s": [round(float(value), 7) for value in hill_velocity],
            "ground_truth_pixel": None if ground_truth is None else [round(value, 3) for value in ground_truth],
            "opencv_pixel": None if measurement is None else [round(float(value), 3) for value in measurement],
            "kalman_pixel": None if estimate is None else [round(float(value), 3) for value in estimate[:2]],
            "detection_error_px": None if detection_error is None else round(detection_error, 3),
            "tracking_error_px": None if tracking_error is None else round(tracking_error, 3),
            "camera_pan_rad": round(self.actuator.pan_rad, 6),
            "camera_tilt_rad": round(self.actuator.tilt_rad, 6),
            "camera_pan_rate_rad_s": round(self.actuator.pan_rate_rad_s, 6),
            "camera_tilt_rate_rad_s": round(self.actuator.tilt_rate_rad_s, 6),
            "fov_deg": [round(value, 3) for value in field_of_view],
            "locked": bool(measurement is not None and tracking_error is not None and tracking_error < 20.0),
            "trajectory_m": list(self.trajectory),
        }
        self.frame_number += 1
        return frame, telemetry

    def run(self) -> None:
        period = 1.0 / float(self.config["fps"])
        while not self.stop_event.is_set():
            started = time.perf_counter()
            frame, telemetry = self.step()
            with self.lock:
                self.latest_frame = frame
                self.latest_telemetry = telemetry
            self.stop_event.wait(max(0.0, period - (time.perf_counter() - started)))

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, name="simulation-engine", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def snapshot(self) -> tuple[bytes, dict]:
        return self.jpeg_snapshot(), self.telemetry_snapshot()

    def telemetry_snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest_telemetry)

    def jpeg_snapshot(self) -> bytes:
        with self.lock:
            frame = self.latest_frame.copy()
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not encoded:
            raise RuntimeError("Could not encode camera frame")
        return jpeg.tobytes()
