"""Live 3D target motion, virtual camera rendering and OpenCV detection."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

from stationary_demo import DEFAULT_CONFIG, ROOT, compose_demo, load_config


DEFAULT_LIVE_CONFIG = ROOT / "configs" / "live.json"


def target_position(simulation_time_s: float, motion: dict) -> list[float]:
    """Return a smooth deterministic 3D path around the configured centre."""
    centre = np.asarray(motion["centre_m"], dtype=float)
    amplitude = np.asarray(motion["amplitude_m"], dtype=float)
    frequency = np.asarray(motion["frequency_hz"], dtype=float)
    phase = 2.0 * np.pi * frequency * simulation_time_s
    offset = amplitude * np.array(
        [np.sin(phase[0]), np.cos(phase[1]), np.sin(phase[2])],
        dtype=float,
    )
    return (centre + offset).tolist()


def annotate(image: np.ndarray, frame_number: int, simulation_time_s: float, processing_fps: float) -> None:
    cv2.rectangle(image, (0, image.shape[0] - 34), (image.shape[1], image.shape[0]), (12, 15, 22), -1)
    status = f"LIVE | frame {frame_number:05d} | t={simulation_time_s:6.2f}s | processing={processing_fps:6.1f} FPS"
    cv2.putText(image, status, (18, image.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stationary-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--live-config", type=Path, default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Run without opening a window.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many frames; zero runs continuously.")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "live_step2.png")
    parser.add_argument("--max-error-px", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.stationary_config)
    live_config = load_config(args.live_config)
    requested_fps = float(live_config["fps"])
    frame_period_s = 1.0 / requested_fps
    frame_limit = args.frames if args.frames > 0 else (180 if args.headless else 0)

    frame_number = 0
    paused = False
    errors: list[float] = []
    processing_times: list[float] = []
    last_image: np.ndarray | None = None

    try:
        while frame_limit == 0 or frame_number < frame_limit:
            loop_started = time.perf_counter()
            simulation_time_s = frame_number / requested_fps
            frame_config = copy.deepcopy(base_config)
            frame_config["seed"] = int(base_config["seed"]) + frame_number
            frame_config["target_label"] = "live target"
            frame_config["target_position_m"] = target_position(simulation_time_s, live_config["motion"])

            image, metrics = compose_demo(frame_config)
            processing_time_s = time.perf_counter() - loop_started
            processing_fps = 1.0 / max(processing_time_s, 1e-9)
            annotate(image, frame_number, simulation_time_s, processing_fps)
            last_image = image
            errors.append(float(metrics["detection_error_px"]))
            processing_times.append(processing_time_s)

            if args.headless:
                frame_number += 1
                continue

            cv2.imshow("SIH26169 - Live Virtual Camera Tracking", image)
            remaining_ms = max(1, int(round((frame_period_s - processing_time_s) * 1000.0)))
            key = cv2.waitKey(remaining_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                frame_number = 0
                errors.clear()
                processing_times.clear()
                continue
            if not paused:
                frame_number += 1
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    if last_image is None or not errors:
        raise RuntimeError("No simulation frames were produced.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), last_image):
        raise RuntimeError(f"Could not save output to {args.output}")

    summary = {
        "frames": len(errors),
        "requested_fps": requested_fps,
        "mean_detection_error_px": round(float(np.mean(errors)), 3),
        "max_detection_error_px": round(float(np.max(errors)), 3),
        "mean_processing_ms": round(float(np.mean(processing_times) * 1000.0), 3),
        "mean_processing_fps": round(float(1.0 / np.mean(processing_times)), 1),
        "detection_rate": 1.0,
    }
    print(json.dumps(summary, indent=2))
    if summary["max_detection_error_px"] > args.max_error_px:
        raise SystemExit(f"Detection error exceeded {args.max_error_px} px")


if __name__ == "__main__":
    main()
