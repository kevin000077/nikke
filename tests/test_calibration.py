from __future__ import annotations

import numpy as np

from marble_aim.calibration import TrackPoint, fit_calibration
from marble_aim.geometry import Rect


def make_v_bounce() -> list[TrackPoint]:
    points: list[TrackPoint] = []
    coordinates = [
        (50, 80),
        (40, 65),
        (30, 50),
        (20, 35),
        (10, 20),
        (7, 12),
        (14, 20),
        (25, 35),
        (36, 50),
        (47, 65),
        (58, 80),
        (70, 65),
        (82, 50),
        (91, 35),
        (93, 20),
        (86, 30),
        (75, 45),
        (64, 60),
        (53, 75),
    ]
    for index, (x, y) in enumerate(coordinates):
        points.append(TrackPoint(index / 30, x, y, 6.0))
    return points


def test_fit_calibration_reports_sparse_data():
    result = fit_calibration(
        [TrackPoint(0, 50, 50, 7), TrackPoint(0.1, 55, 45, 7)],
        Rect(0, 0, 100, 100),
        7,
    )
    assert result.success is False


def test_fit_calibration_uses_observed_radius():
    result = fit_calibration(make_v_bounce(), Rect(0, 0, 100, 100), 7)
    assert np.isclose(result.ball_radius, 6.0)
    assert result.event_count >= 1
