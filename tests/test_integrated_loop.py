from __future__ import annotations

import numpy as np

from src.core.engine import SimulationEngine
from src.physics.cw import ClohessyWiltshire


def test_cw_propagation_remains_finite() -> None:
    model = ClohessyWiltshire(
        mean_motion_rad_s=0.00113,
        position_m=np.array([0.0, 1.5, 0.8]),
        velocity_m_s=np.array([0.0, 0.0005, 0.0002]),
    )
    for _ in range(120):
        position, velocity = model.step(1.0)
    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(velocity))
    assert not np.allclose(position, [0.0, 1.5, 0.8])


def test_visual_tracking_acquires_and_centres_target() -> None:
    engine = SimulationEngine()
    telemetry = [engine.step()[1] for _ in range(210)]
    settled = telemetry[150:]
    assert all(frame["opencv_pixel"] is not None for frame in telemetry)
    assert max(frame["detection_error_px"] for frame in telemetry) < 2.0
    assert sum(frame["locked"] for frame in settled) / len(settled) > 0.95
    assert np.mean([frame["tracking_error_px"] for frame in settled]) < 5.0

