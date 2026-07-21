from __future__ import annotations

import cv2
import numpy as np

from marble_aim.config import VisionConfig
from marble_aim.geometry import Rect
from marble_aim.vision import (
    BoardDetector,
    MotionAimDetector,
    TemporalDetector,
    _detect_aim_reticle_radius,
)


def hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    pixel = np.uint8([[[h, s, v]]])
    converted = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in converted)


def synthetic_board() -> np.ndarray:
    image = np.zeros((360, 260, 3), dtype=np.uint8)
    image[:] = (35, 180, 230)
    water = hsv_to_bgr(98, 180, 220)
    green = hsv_to_bgr(60, 220, 180)
    cv2.rectangle(image, (30, 20), (230, 330), water, -1)
    cv2.rectangle(image, (55, 70), (115, 125), green, -1)
    cv2.rectangle(image, (145, 105), (205, 160), green, -1)
    return image


def test_detector_finds_board_and_blocks():
    result = BoardDetector(VisionConfig()).detect(synthetic_board())
    assert result.board is not None
    assert abs(result.board.left - 30) <= 2
    assert abs(result.board.right - 230) <= 2
    assert len(result.obstacles) == 2
    assert result.launch_origin is not None


def test_temporal_detector_rejects_single_frame_false_positive():
    detector = TemporalDetector(BoardDetector(VisionConfig()), frames=3)
    base = synthetic_board()
    detector.detect(base)
    detector.detect(base)
    noisy = base.copy()
    green = hsv_to_bgr(60, 220, 180)
    cv2.rectangle(noisy, (40, 200), (100, 255), green, -1)
    result = detector.detect(noisy)
    assert len(result.obstacles) == 2


def synthetic_collision_scene() -> np.ndarray:
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    image[:] = hsv_to_bgr(100, 210, 235)
    pale = hsv_to_bgr(95, 35, 245)
    green = hsv_to_bgr(55, 220, 210)
    purple = hsv_to_bgr(145, 200, 210)
    cv2.rectangle(image, (240, 90), (660, 100), pale, -1)
    cv2.rectangle(image, (240, 500), (660, 510), pale, -1)
    cv2.rectangle(image, (240, 90), (250, 510), pale, -1)
    cv2.rectangle(image, (650, 90), (660, 510), pale, -1)
    cv2.rectangle(image, (260, 180), (319, 239), green, -1)
    cv2.rectangle(image, (320, 180), (379, 239), green, -1)
    cv2.rectangle(image, (440, 250), (499, 309), purple, -1)
    start, end = np.array([450.0, 490.0]), np.array([550.0, 110.0])
    direction = (end - start) / np.linalg.norm(end - start)
    for distance in np.arange(0, np.linalg.norm(end - start), 22):
        first = start + direction * distance
        second = start + direction * min(distance + 12, np.linalg.norm(end - start))
        cv2.line(
            image,
            tuple(np.round(first).astype(int)),
            tuple(np.round(second).astype(int)),
            pale,
            3,
        )
    return image


def test_collision_frame_grid_blocks_and_aim_line():
    result = BoardDetector(VisionConfig()).detect(synthetic_collision_scene())
    assert result.board is not None
    assert abs(result.board.left - 250) <= 2
    assert abs(result.board.right - 650) <= 2
    assert abs(result.board.top - 100) <= 2
    assert abs(result.board.bottom - 500) <= 2
    assert len(result.obstacles) == 3
    assert all(obstacle.corner_radius == 0 for obstacle in result.obstacles)
    assert result.aim_line is not None


def test_collision_scene_scales_to_near_4k_resolution():
    image = cv2.resize(
        synthetic_collision_scene(),
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_NEAREST,
    )

    result = BoardDetector(VisionConfig()).detect(image)

    assert result.collision_frame_detected
    assert result.board is not None
    assert abs(result.board.left - 1000) <= 5
    assert abs(result.board.right - 2600) <= 5
    assert len(result.obstacles) == 3
    assert result.aim_line is not None


def test_single_pixel_inner_vertical_edges_define_collision_frame():
    image = np.full((600, 900, 3), hsv_to_bgr(100, 210, 235), dtype=np.uint8)
    pale = hsv_to_bgr(95, 35, 245)
    cv2.rectangle(image, (220, 90), (239, 510), pale, -1)
    cv2.line(image, (250, 90), (250, 510), pale, 1)
    cv2.line(image, (650, 90), (650, 510), pale, 1)
    cv2.rectangle(image, (651, 90), (670, 510), pale, -1)
    cv2.rectangle(image, (250, 90), (650, 100), pale, -1)
    cv2.rectangle(image, (250, 500), (650, 510), pale, -1)

    result = BoardDetector(VisionConfig()).detect(image)

    assert result.collision_frame_detected
    assert result.board is not None
    assert abs(result.board.left - 250) <= 1
    assert abs(result.board.right - 650) <= 1


def test_manual_board_skips_automatic_collision_frame_detection():
    image = np.full((600, 900, 3), hsv_to_bgr(100, 210, 235), dtype=np.uint8)
    green = hsv_to_bgr(55, 220, 210)
    cv2.rectangle(image, (260, 180), (319, 239), green, -1)
    manual = [250 / 900, 100 / 600, 650 / 900, 500 / 600]

    result = BoardDetector(
        VisionConfig(),
        manual_board_normalized=manual,
    ).detect(image)

    assert result.collision_frame_detected
    assert result.board is not None
    assert abs(result.board.left - 250) <= 1
    assert abs(result.board.top - 100) <= 1
    assert abs(result.board.right - 650) <= 1
    assert abs(result.board.bottom - 500) <= 1
    assert len(result.obstacles) == 1


def test_aim_reticle_radius_scales_from_visible_circle():
    image = np.full((600, 900, 3), hsv_to_bgr(100, 210, 235), dtype=np.uint8)
    board = Rect(250.0, 100.0, 650.0, 500.0)
    line = ((450.0, 500.0), (554.0, 100.0))
    direction = np.array(line[1]) - np.array(line[0])
    direction /= np.linalg.norm(direction)
    radius = 14
    center = np.array(line[1]) - direction * radius
    pale = hsv_to_bgr(95, 35, 245)
    cv2.circle(image, tuple(np.round(center).astype(int)), radius, pale, 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    measured = _detect_aim_reticle_radius(image, hsv, board, line, [])

    assert measured is not None
    assert abs(measured - radius) <= 2


def test_cropped_frame_and_water_coloured_blocks_are_detected():
    image = np.zeros((636, 529, 3), dtype=np.uint8)
    water = hsv_to_bgr(100, 210, 235)
    pale = hsv_to_bgr(95, 35, 245)
    orange = hsv_to_bgr(20, 220, 235)
    image[:] = water
    cv2.rectangle(image, (35, 18), (480, 28), pale, -1)
    cv2.rectangle(image, (35, 590), (480, 602), pale, -1)
    cv2.rectangle(image, (35, 18), (47, 602), pale, -1)
    cv2.rectangle(image, (466, 18), (480, 602), pale, -1)
    for x, y, color in [
        (118, 100, water),
        (188, 100, orange),
        (48, 170, water),
        (188, 170, water),
    ]:
        cv2.rectangle(image, (x, y), (x + 67, y + 67), (15, 35, 60), 3)
        cv2.rectangle(image, (x + 4, y + 4), (x + 63, y + 55), color, -1)
        cv2.putText(
            image,
            "5",
            (x + 23, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (245, 245, 245),
            3,
        )
    start, end = np.array([170.0, 590.0]), np.array([450.0, 320.0])
    direction = (end - start) / np.linalg.norm(end - start)
    for distance in np.arange(0, np.linalg.norm(end - start), 20):
        first = start + direction * distance
        second = start + direction * min(distance + 11, np.linalg.norm(end - start))
        cv2.line(
            image,
            tuple(np.round(first).astype(int)),
            tuple(np.round(second).astype(int)),
            pale,
            3,
        )
    result = BoardDetector(VisionConfig()).detect(image)
    assert result.collision_frame_detected
    assert result.board is not None
    assert result.board.top < image.shape[0] * 0.06
    assert result.board.bottom > image.shape[0] * 0.90
    assert len(result.obstacles) == 4
    assert result.aim_line is not None


def test_shallow_aim_uses_mouse_when_dotted_line_is_absent():
    image = synthetic_collision_scene()
    # Remove the synthetic dotted line while preserving the collision frame.
    water = hsv_to_bgr(100, 210, 235)
    cv2.rectangle(image, (252, 105), (648, 498), water, -1)
    green = hsv_to_bgr(55, 220, 210)
    cv2.rectangle(image, (260, 180), (319, 239), green, -1)
    arrow = hsv_to_bgr(31, 35, 254)
    cv2.line(image, (440, 495), (575, 465), arrow, 16)
    cv2.fillConvexPoly(
        image,
        np.array([(610, 457), (570, 445), (580, 480)], dtype=np.int32),
        arrow,
    )

    detector = BoardDetector(VisionConfig())
    result = detector.detect(image)

    assert result.aim_marker_present
    assert result.aim_line is None
    line = detector.detect_aim_only(
        image,
        result.board,
        result.obstacles,
        launch_origin=(450.0, result.board.bottom),
        cursor_position=(720.0, 540.0),
    )
    assert line == ((450.0, result.board.bottom), (720.0, 540.0))


def test_dotted_origin_and_mouse_direction_ignore_arrow_axis():
    image = synthetic_collision_scene()
    arrow = hsv_to_bgr(31, 35, 254)
    cv2.line(image, (445, 495), (500, 468), arrow, 16)
    cv2.fillConvexPoly(
        image,
        np.array([(530, 452), (495, 454), (510, 485)], dtype=np.int32),
        arrow,
    )
    board = Rect(250.0, 100.0, 650.0, 500.0)

    line = BoardDetector(VisionConfig()).detect_aim_only(
        image,
        board,
        [],
        launch_origin=(300.0, 500.0),
        cursor_position=(700.0, 300.0),
    )

    assert line is not None
    lower, upper = line
    assert abs(lower[0] - 450.0) <= 8.0
    assert upper == (700.0, 300.0)


def test_thin_cream_decoration_is_not_mistaken_for_aim_arrow():
    image = synthetic_collision_scene()
    water = hsv_to_bgr(100, 210, 235)
    cv2.rectangle(image, (252, 105), (648, 498), water, -1)
    arrow = hsv_to_bgr(31, 35, 254)
    cv2.rectangle(image, (620, 430), (626, 480), arrow, -1)

    result = BoardDetector(VisionConfig()).detect(image)

    assert result.aim_line is None


def test_motion_aim_detector_tracks_changed_current_dotted_line():
    first = synthetic_collision_scene()
    second = first.copy()
    water = hsv_to_bgr(100, 210, 235)
    pale = hsv_to_bgr(95, 35, 245)
    cv2.rectangle(second, (380, 320), (590, 499), water, -1)
    start, end = np.array([450.0, 490.0]), np.array([610.0, 300.0])
    direction = (end - start) / np.linalg.norm(end - start)
    for distance in np.arange(0, np.linalg.norm(end - start), 18):
        segment_start = start + direction * distance
        segment_end = start + direction * min(
            distance + 10,
            np.linalg.norm(end - start),
        )
        cv2.line(
            second,
            tuple(np.round(segment_start).astype(int)),
            tuple(np.round(segment_end).astype(int)),
            pale,
            3,
        )
    arrow = hsv_to_bgr(31, 35, 254)
    for image in (first, second):
        cv2.line(image, (445, 495), (500, 468), arrow, 16)
        cv2.fillConvexPoly(
            image,
            np.array([(530, 452), (495, 454), (510, 485)], dtype=np.int32),
            arrow,
        )
    tracker = MotionAimDetector()
    board = Rect(250.0, 100.0, 650.0, 500.0)
    origin = (450.0, 500.0)

    first_line = tracker.detect(
        first,
        board,
        [],
        origin,
        active=True,
        cursor_position=(720.0, 280.0),
    )
    line = tracker.detect(
        second,
        board,
        [],
        origin,
        active=True,
        cursor_position=(720.0, 280.0),
    )

    assert first_line == ((450.0, 500.0), (720.0, 280.0))
    assert line is not None
    lower, upper = line
    expected_bottom_x = start[0] + (board.bottom - start[1]) * (
        (end[0] - start[0]) / (end[1] - start[1])
    )
    assert abs(lower[0] - expected_bottom_x) <= 3
    assert upper == (720.0, 280.0)


def test_motion_aim_detector_rejects_moving_pale_line_without_arrow_marker():
    first = synthetic_collision_scene()
    second = first.copy()
    pale = hsv_to_bgr(95, 35, 245)
    cv2.line(second, (450, 490), (610, 300), pale, 4)
    tracker = MotionAimDetector()
    board = Rect(250.0, 100.0, 650.0, 500.0)
    origin = (450.0, 500.0)

    assert tracker.detect(
        first,
        board,
        [],
        origin,
        active=True,
        cursor_position=(720.0, 280.0),
    ) is None
    assert tracker.detect(
        second,
        board,
        [],
        origin,
        active=True,
        cursor_position=(720.0, 280.0),
    ) is None
    assert not tracker.last_marker_present
