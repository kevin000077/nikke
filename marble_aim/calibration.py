from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .geometry import Rect, reflect, unit, vec

Image = NDArray[np.uint8]


@dataclass(slots=True)
class TrackPoint:
    timestamp: float
    x: float
    y: float
    observed_radius: float

    @property
    def position(self) -> NDArray[np.float64]:
        return vec(self.x, self.y)


@dataclass(slots=True)
class CalibrationResult:
    success: bool
    ball_radius: float
    boundary_offsets: list[float]
    reflection_bias_deg: float
    event_count: int
    message: str


class MotionBallTracker:
    def __init__(self):
        self.previous_gray: Image | None = None
        self.points: list[TrackPoint] = []
        self._last_position: NDArray[np.float64] | None = None
        self._last_seen = 0.0

    def reset(self) -> None:
        self.previous_gray = None
        self.points.clear()
        self._last_position = None
        self._last_seen = 0.0

    def observe(self, bgr: Image, board: Rect, timestamp: float | None = None) -> TrackPoint | None:
        timestamp = time.monotonic() if timestamp is None else timestamp
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous_gray is None:
            self.previous_gray = gray
            return None
        delta = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray
        _, mask = cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        roi = np.zeros_like(mask)
        cv2.rectangle(
            roi,
            (round(board.left), round(board.top)),
            (round(board.right), round(board.bottom)),
            255,
            -1,
        )
        mask = cv2.bitwise_and(mask, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, TrackPoint]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 12 <= area <= 700:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / max(1, height)
            if not 0.35 <= aspect <= 2.8 or max(width, height) > 45:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            point = TrackPoint(timestamp, float(center_x), float(center_y), max(2.0, radius))
            if self._last_position is None or timestamp - self._last_seen > 0.4:
                center_bias = abs(center_x - board.center[0]) / max(1.0, board.width)
                score = area - center_bias * 15
            else:
                distance = float(np.linalg.norm(point.position - self._last_position))
                if distance > 140:
                    continue
                score = 200.0 - distance + area * 0.05
            candidates.append((score, point))
        if not candidates:
            if timestamp - self._last_seen > 0.4:
                self._last_position = None
            return None
        point = max(candidates, key=lambda item: item[0])[1]
        self.points.append(point)
        self._last_position = point.position
        self._last_seen = timestamp
        return point


def _signed_angle_degrees(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.dot(first, second))
    return math.degrees(math.atan2(cross, dot))


@dataclass(slots=True)
class _BounceEvent:
    point: NDArray[np.float64]
    incoming: NDArray[np.float64]
    outgoing: NDArray[np.float64]
    wall_index: int
    normal: NDArray[np.float64]


def _extract_wall_events(points: list[TrackPoint], board: Rect) -> list[_BounceEvent]:
    if len(points) < 7:
        return []
    positions = np.array([[item.x, item.y] for item in points], dtype=np.float64)
    events: list[_BounceEvent] = []
    walls = [
        (0, board.left, vec(1, 0)),
        (0, board.right, vec(-1, 0)),
        (1, board.top, vec(0, 1)),
        (1, board.bottom, vec(0, -1)),
    ]
    for index in range(2, len(positions) - 2):
        if (
            points[index].timestamp - points[index - 2].timestamp > 0.2
            or points[index + 2].timestamp - points[index].timestamp > 0.2
        ):
            continue
        before = positions[index] - positions[index - 2]
        after = positions[index + 2] - positions[index]
        if np.linalg.norm(before) < 3 or np.linalg.norm(after) < 3:
            continue
        incoming, outgoing = unit(before), unit(after)
        turn = abs(_signed_angle_degrees(incoming, outgoing))
        if turn < 25:
            continue
        distances = [abs(positions[index][axis] - value) for axis, value, _ in walls]
        wall_index = int(np.argmin(distances))
        if distances[wall_index] > 38:
            continue
        normal = walls[wall_index][2]
        if float(np.dot(incoming, normal)) >= -0.05:
            continue
        if events and np.linalg.norm(events[-1].point - positions[index]) < 15:
            continue
        events.append(_BounceEvent(positions[index], incoming, outgoing, wall_index, normal))
    return events


def fit_calibration(
    points: list[TrackPoint],
    board: Rect,
    default_radius: float,
) -> CalibrationResult:
    events = _extract_wall_events(points, board)
    observed_radii = [
        point.observed_radius for point in points if 2.0 <= point.observed_radius <= 25.0
    ]
    radius = float(np.median(observed_radii)) if observed_radii else default_radius
    radius = float(np.clip(radius, 2.0, 25.0))
    if len(events) < 2:
        return CalibrationResult(
            False,
            radius,
            [0.0, 0.0, 0.0, 0.0],
            0.0,
            len(events),
            "墙壁反弹样本不足；请在校准期间至少向左右两侧发射。",
        )

    wall_values = [board.left, board.right, board.top, board.bottom]
    wall_signs = [1.0, -1.0, 1.0, -1.0]

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        offsets = parameters[:4]
        bias = float(parameters[4])
        values: list[float] = []
        for event in events:
            axis = 0 if event.wall_index < 2 else 1
            predicted_contact = (
                wall_values[event.wall_index]
                + offsets[event.wall_index]
                + wall_signs[event.wall_index] * radius
            )
            values.append((float(event.point[axis]) - predicted_contact) / 3.0)
            ideal = reflect(event.incoming, event.normal, bias)
            values.append(_signed_angle_degrees(ideal, event.outgoing) / 3.0)
        values.extend((offsets / 15.0).tolist())
        return np.asarray(values, dtype=np.float64)

    solution = least_squares(
        residual,
        np.zeros(5, dtype=np.float64),
        bounds=([-20, -20, -20, -20, -10], [20, 20, 20, 20, 10]),
        loss="soft_l1",
    )
    return CalibrationResult(
        True,
        radius,
        [round(float(value), 3) for value in solution.x[:4]],
        round(float(solution.x[4]), 3),
        len(events),
        f"已使用 {len(events)} 个墙壁反弹事件完成校准。",
    )
