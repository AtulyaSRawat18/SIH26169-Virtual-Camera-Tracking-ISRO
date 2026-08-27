"""Step 1: stationary 3D target, synthetic camera and OpenCV detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "stationary.json"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def project_tracking_camera(point: np.ndarray, camera: dict) -> tuple[float, float]:
    """Project a 3D camera-frame point into pixels using a pinhole camera."""
    x_m, y_m, z_m = point
    if z_m <= 0:
        raise ValueError("Target must be in front of the camera (z > 0).")
    focal = float(camera["focal_length_px"])
    cx = float(camera["width_px"]) / 2.0
    cy = float(camera["height_px"]) / 2.0
    return cx + focal * x_m / z_m, cy - focal * y_m / z_m


def render_camera_frame(config: dict) -> tuple[np.ndarray, tuple[float, float]]:
    camera = config["camera"]
    render = config["render"]
    target = np.asarray(config["target_position_m"], dtype=np.float64)
    ground_truth = project_tracking_camera(target, camera)

    height = int(camera["height_px"])
    width = int(camera["width_px"])
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    target_px = tuple(np.rint(ground_truth).astype(int))
    cv2.circle(
        frame,
        target_px,
        int(render["beacon_radius_px"]),
        (255, 255, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    rng = np.random.default_rng(int(config["seed"]))
    noise = rng.normal(0.0, float(render["noise_sigma"]), frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, ground_truth


def detect_beacon(frame: np.ndarray, threshold: int) -> tuple[float, float] | None:
    """Detect the largest bright blob without using simulation ground truth."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in contours if 20.0 <= cv2.contourArea(contour) <= 2000.0]
    if not valid:
        return None
    blob = max(valid, key=cv2.contourArea)
    moments = cv2.moments(blob)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def observer_project(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Project world points into a fixed third-person observer view."""
    width, height = size
    eye = np.array([9.0, 6.0, -10.0])
    centre = np.array([0.0, 0.0, 4.0])
    forward = centre - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    relative = points - eye
    camera_points = np.column_stack((relative @ right, relative @ up, relative @ forward))
    focal = 470.0
    pixels = np.column_stack(
        (
            width / 2.0 + focal * camera_points[:, 0] / camera_points[:, 2],
            height / 2.0 - focal * camera_points[:, 1] / camera_points[:, 2],
        )
    )
    return np.rint(pixels).astype(int)


def render_world_view(config: dict) -> np.ndarray:
    width, height = int(config["camera"]["width_px"]), int(config["camera"]["height_px"])
    canvas = np.full((height, width, 3), (18, 22, 32), dtype=np.uint8)

    grid_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for coordinate in np.linspace(-4, 4, 9):
        grid_lines.append((np.array([coordinate, -2.0, 0.0]), np.array([coordinate, -2.0, 10.0])))
        grid_lines.append((np.array([-4.0, -2.0, coordinate + 4]), np.array([4.0, -2.0, coordinate + 4])))
    for start, end in grid_lines:
        projected = observer_project(np.vstack((start, end)), (width, height))
        cv2.line(canvas, tuple(projected[0]), tuple(projected[1]), (45, 55, 72), 1, cv2.LINE_AA)

    origin = np.array([0.0, 0.0, 0.0])
    axes = [
        (np.array([2.0, 0.0, 0.0]), (80, 80, 255), "X"),
        (np.array([0.0, 2.0, 0.0]), (80, 255, 80), "Y"),
        (np.array([0.0, 0.0, 3.0]), (255, 160, 80), "Z"),
    ]
    origin_px = observer_project(origin[None, :], (width, height))[0]
    for endpoint, colour, label in axes:
        endpoint_px = observer_project(endpoint[None, :], (width, height))[0]
        cv2.arrowedLine(canvas, tuple(origin_px), tuple(endpoint_px), colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, tuple(endpoint_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    frustum = np.array(
        [[0, 0, 0], [-2.0, -1.5, 4], [2.0, -1.5, 4], [2.0, 1.5, 4], [-2.0, 1.5, 4]],
        dtype=float,
    )
    frustum_px = observer_project(frustum, (width, height))
    for corner in range(1, 5):
        cv2.line(canvas, tuple(frustum_px[0]), tuple(frustum_px[corner]), (190, 120, 45), 1, cv2.LINE_AA)
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 1)):
        cv2.line(canvas, tuple(frustum_px[a]), tuple(frustum_px[b]), (190, 120, 45), 1, cv2.LINE_AA)

    target = np.asarray(config["target_position_m"], dtype=float)
    target_px = observer_project(target[None, :], (width, height))[0]
    cv2.line(canvas, tuple(origin_px), tuple(target_px), (80, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(target_px), 10, (80, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "stationary target", tuple(target_px + [12, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1)
    cv2.putText(canvas, "3D WORLD VIEW", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 240, 250), 2)
    return canvas


def compose_demo(config: dict) -> tuple[np.ndarray, dict]:
    frame, ground_truth = render_camera_frame(config)
    detection = detect_beacon(frame, int(config["render"]["threshold"]))
    if detection is None:
        raise RuntimeError("OpenCV did not detect the stationary beacon.")

    error_px = float(np.linalg.norm(np.asarray(detection) - np.asarray(ground_truth)))
    camera_panel = frame.copy()
    centre = (int(round(detection[0])), int(round(detection[1])))
    cv2.drawMarker(camera_panel, centre, (0, 255, 80), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(camera_panel, "VIRTUAL CAMERA + OPENCV", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 240, 250), 2)
    cv2.putText(camera_panel, f"error: {error_px:.2f} px", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

    combined = np.hstack((render_world_view(config), camera_panel))
    metrics = {
        "target_position_m": config["target_position_m"],
        "ground_truth_pixel": [round(value, 3) for value in ground_truth],
        "detected_pixel": [round(value, 3) for value in detection],
        "detection_error_px": round(error_px, 3),
        "detected": True,
    }
    return combined, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Save output without opening a window.")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "stationary_step1.png")
    parser.add_argument("--max-error-px", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, metrics = compose_demo(load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Could not save output to {args.output}")
    print(json.dumps(metrics, indent=2))
    if metrics["detection_error_px"] > args.max_error_px:
        raise SystemExit(f"Detection error exceeded {args.max_error_px} px")
    if not args.headless:
        cv2.imshow("SIH26169 - Stationary Step 1", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

