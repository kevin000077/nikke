from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import VisionConfig
from .geometry import Obstacle, Rect

Image = NDArray[np.uint8]


@dataclass(slots=True)
class DetectionResult:
    board: Rect | None
    obstacles: list[Obstacle]
    launch_origin: tuple[float, float] | None
    aim_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    water_mask: Image | None = None
    block_mask: Image | None = None
    confidence: float = 0.0
    collision_frame_detected: bool = False
    aim_radius: float | None = None


def _largest_contour(mask: Image) -> tuple[NDArray[np.int32] | None, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    return contour, float(cv2.contourArea(contour))


def _board_from_row_spans(mask: Image) -> tuple[Rect | None, float]:
    """Find the central playfield even when it touches same-colour UI at the top."""
    height, width = mask.shape
    x_min = round(width * 0.04)
    x_max = round(width * 0.96)
    spans: list[tuple[int, int, int]] = []
    for y in range(height):
        xs = np.flatnonzero(mask[y, x_min:x_max] > 0)
        if xs.size < width * 0.15:
            continue
        left = int(xs[0] + x_min)
        right = int(xs[-1] + x_min)
        span_width = right - left + 1
        midpoint = (left + right) / 2
        if not (width * 0.45 <= span_width <= width * 0.82):
            continue
        if abs(midpoint - width / 2) > width * 0.15:
            continue
        spans.append((y, left, right))
    if not spans:
        return None, 0.0

    # Reject isolated rows caused by sprites: require a local vertical consensus.
    valid_y = {item[0] for item in spans}
    stable = [
        item
        for item in spans
        if sum((item[0] + offset) in valid_y for offset in range(-3, 4)) >= 4
    ]
    if not stable:
        return None, 0.0
    top, bottom = stable[0][0], stable[-1][0]
    middle = [
        item
        for item in stable
        if top + (bottom - top) * 0.08 <= item[0] <= bottom - (bottom - top) * 0.08
    ] or stable
    left = float(np.percentile([item[1] for item in middle], 30))
    right = float(np.percentile([item[2] for item in middle], 70))
    coverage = len(stable) / max(1, bottom - top + 1)
    width_stability = 1.0 - min(
        1.0,
        float(np.std([item[2] - item[1] for item in middle])) / max(1.0, right - left),
    )
    return Rect(left, float(top), right, float(bottom)), coverage * width_stability


def _runs(indices: NDArray[np.int64]) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]


def _detect_collision_frame(hsv: Image) -> tuple[Rect | None, float, Image]:
    """Detect the inner edges of the pale rounded collision frame."""
    height, width = hsv.shape[:2]
    pale = np.uint8((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 205)) * 255
    y0, y1 = round(height * 0.20), round(height * 0.72)
    vertical_counts = (pale[y0:y1] > 0).sum(axis=0)
    strong_columns = np.flatnonzero(vertical_counts >= (y1 - y0) * 0.72)
    column_runs = [
        run
        for run in _runs(strong_columns)
        # The actual inner collision edge can be a single bright pixel even
        # though the decorative outer rim is much wider.
        if run[1] - run[0] >= 0
        and width * 0.015 <= (run[0] + run[1]) / 2 <= width * 0.995
    ]
    center = width / 2
    left_runs = [run for run in column_runs if run[1] < center]
    right_runs = [run for run in column_runs if run[0] > center]
    pairs: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    for left_run in left_runs:
        for right_run in right_runs:
            left, right = left_run[1], right_run[0]
            frame_width = right - left
            if not width * 0.20 <= frame_width <= width * 0.96:
                continue
            symmetry = abs((left + right) / 2 - center)
            expected_width_penalty = min(
                abs(frame_width - width * 0.255),
                abs(frame_width - width * 0.68),
                abs(frame_width - width * 0.90),
            ) * 0.2
            pairs.append((symmetry + expected_width_penalty, left_run, right_run))
    if not pairs:
        return None, 0.0, pale
    _, left_run, right_run = min(pairs, key=lambda item: item[0])
    left, right = float(left_run[1]), float(right_run[0])
    if right - left < 50:
        return None, 0.0, pale

    x0, x1 = round(left), round(right) + 1
    row_counts = (pale[:, x0:x1] > 0).sum(axis=1)
    frame_width = x1 - x0
    strong_rows = np.flatnonzero(row_counts >= frame_width * 0.72)
    row_runs = [run for run in _runs(strong_rows) if run[1] - run[0] >= 1]
    top_candidates = [
        run
        for run in row_runs
        if height * 0.02 <= (run[0] + run[1]) / 2 <= height * 0.45
    ]
    bottom_candidates = [
        run
        for run in row_runs
        if height * 0.55 <= (run[0] + run[1]) / 2 <= height * 0.98
    ]
    if not top_candidates or not bottom_candidates:
        return None, 0.0, pale
    top_run = min(top_candidates, key=lambda run: run[0])
    bottom_run = min(bottom_candidates, key=lambda run: run[0])
    top = float(top_run[1])
    bottom = float(bottom_run[0])
    if bottom - top < height * 0.35:
        return None, 0.0, pale
    vertical_support = min(
        1.0,
        (
            np.mean(vertical_counts[left_run[0] : left_run[1] + 1])
            + np.mean(vertical_counts[right_run[0] : right_run[1] + 1])
        )
        / (2 * (y1 - y0)),
    )
    horizontal_support = min(
        1.0,
        (row_counts[round(top)] + row_counts[round(bottom)]) / (2 * frame_width),
    )
    return Rect(left, top, right, bottom), float(
        0.55 * vertical_support + 0.45 * horizontal_support
    ), pale


def _detect_grid_blocks(
    bgr: Image, hsv: Image, board: Rect
) -> tuple[list[Obstacle], Image]:
    """Find occupied cells on the six-column lattice defined by the white frame."""
    height, width = hsv.shape[:2]
    grid_step = board.width / 6.0
    nominal_cell = grid_step * 0.96
    min_component_area = max(80.0, nominal_cell * nominal_cell * 0.05)
    min_component_width = max(12.0, nominal_cell * 0.45)
    min_component_height = max(10.0, nominal_cell * 0.40)
    saturated_non_water = (
        (hsv[:, :, 1] > 75)
        & (hsv[:, :, 2] > 45)
        & ((hsv[:, :, 0] < 82) | (hsv[:, :, 0] > 112))
    )
    mask = np.uint8(saturated_non_water) * 255
    roi = np.zeros_like(mask)
    cv2.rectangle(
        roi,
        (max(0, round(board.left)), max(0, round(board.top))),
        (min(width - 1, round(board.right)), min(height - 1, round(board.bottom))),
        255,
        -1,
    )
    mask = cv2.bitwise_and(mask, roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[tuple[NDArray[np.int32], int, int, int, int, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, component_width, component_height = cv2.boundingRect(contour)
        if area < min_component_area:
            continue
        if (
            component_width < min_component_width
            or component_height < min_component_height
        ):
            continue
        if component_width > board.width * 0.55 or component_height > board.height * 0.45:
            continue
        components.append((contour, x, y, component_width, component_height, area))
    single_widths = [
        item[3]
        for item in components
        if nominal_cell * 0.78 <= item[3] <= nominal_cell * 1.25
    ]
    single_heights = [
        item[4]
        for item in components
        if nominal_cell * 0.72 <= item[4] <= nominal_cell * 1.30
    ]
    if not components:
        return [], mask
    cell_width = (
        float(np.median(single_widths)) if single_widths else nominal_cell
    )
    cell_height = (
        float(np.median(single_heights)) if single_heights else nominal_cell
    )
    seed_obstacles: list[Obstacle] = []
    for _, x, y, component_width, component_height, _ in components:
        columns = max(1, round(component_width / cell_width))
        rows = max(1, round(component_height / cell_height))
        if columns > 6 or rows > 6:
            continue
        split_width = component_width / columns
        split_height = component_height / rows
        for row in range(rows):
            for column in range(columns):
                left = round(x + column * split_width)
                top = round(y + row * split_height)
                right = round(x + (column + 1) * split_width - 1)
                bottom = round(y + (row + 1) * split_height - 1)
                cell = mask[top : bottom + 1, left : right + 1]
                fill = float(np.count_nonzero(cell)) / max(1, cell.size)
                if fill < 0.20:
                    continue
                rect = Rect(float(left), float(top), float(right), float(bottom))
                seed_obstacles.append(
                    Obstacle(
                        rect,
                        corner_radius=0.0,
                        confidence=float(np.clip((fill - 0.18) / 0.55, 0.35, 1.0)),
                    )
                )
    obstacles: list[Obstacle] = []
    if seed_obstacles:
        pitch_candidates: list[float] = []
        for coordinates in (
            sorted({item.rect.center[0] for item in seed_obstacles}),
            sorted({item.rect.center[1] for item in seed_obstacles}),
        ):
            for first_index, first in enumerate(coordinates):
                for second in coordinates[first_index + 1 :]:
                    delta = second - first
                    multiple = round(delta / grid_step)
                    if multiple < 1 or multiple > 6:
                        continue
                    pitch = delta / multiple
                    if grid_step * 0.92 <= pitch <= grid_step * 1.06:
                        pitch_candidates.append(pitch)
        if pitch_candidates:
            grid_step = float(np.clip(
                np.median(pitch_candidates),
                grid_step * 0.94,
                grid_step * 1.04,
            ))
        step_x = grid_step
        step_y = grid_step
        # Estimate the lattice phase from every reliable seed instead of using
        # one animated sprite as the global anchor.  Quantising only the phase
        # (not the scale) removes the common 1-2 px idle bob.
        x_phases: list[float] = []
        y_phases: list[float] = []
        for obstacle in seed_obstacles:
            if (
                obstacle.rect.width < cell_width * 0.72
                or obstacle.rect.height < cell_height * 0.68
            ):
                continue
            column = round((obstacle.rect.left - board.left) / step_x)
            x_phases.append(obstacle.rect.left - column * step_x)
            row = round((obstacle.rect.top - board.top - step_y) / step_y)
            y_phases.append(obstacle.rect.top - row * step_y)
        if not x_phases or not y_phases:
            return [], mask
        phase_quantum = max(1.0, grid_step * 0.025)
        anchor_x = (
            round(float(np.median(x_phases)) / phase_quantum) * phase_quantum
        )
        anchor_y = (
            round(float(np.median(y_phases)) / phase_quantum) * phase_quantum
        )
        cell_width = float(np.clip(cell_width, grid_step * 0.84, grid_step * 0.99))
        cell_height = float(np.clip(cell_height, grid_step * 0.82, grid_step * 0.99))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 65, 165)
        dark_map = np.uint8((hsv[:, :, 2] < 105) & (hsv[:, :, 1] > 45)) * 255
        columns = 6
        rows = max(
            1,
            math.floor(
                (board.bottom - anchor_y - cell_height * 0.50) / step_y
            )
            + 1,
        )
        occupied: set[tuple[int, int]] = set()
        direct_cells: set[tuple[int, int]] = set()
        for obstacle in seed_obstacles:
            if (
                obstacle.rect.width < cell_width * 0.75
                or obstacle.rect.height < cell_height * 0.70
            ):
                continue
            column = round(
                (obstacle.rect.center[0] - anchor_x - cell_width / 2) / step_x
            )
            row = round(
                (obstacle.rect.center[1] - anchor_y - cell_height / 2) / step_y
            )
            if 0 <= column < columns and 0 <= row < rows:
                direct_cells.add((row, column))
        occupied.update(direct_cells)
        for row in range(rows):
            for column in range(columns):
                left = round(anchor_x + column * step_x)
                top = round(anchor_y + row * step_y)
                right = min(bgr.shape[1] - 1, round(left + cell_width - 1))
                bottom = min(bgr.shape[0] - 1, round(top + cell_height - 1))
                if right <= left or bottom <= top:
                    continue
                edges = edge_map[top : bottom + 1, left : right + 1]
                edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
                dark = dark_map[top : bottom + 1, left : right + 1]
                dark_density = float(np.count_nonzero(dark)) / max(1, dark.size)
                # Blue/water-coloured blocks do not survive the hue mask, but
                # their square outline, shadow and central artwork provide both
                # edge and dark-pixel evidence. Requiring both rejects ripples.
                if edge_density >= 0.085 and dark_density >= 0.045:
                    occupied.add((row, column))
        obstacles = []
        for row, column in sorted(occupied):
            left = round(anchor_x + column * step_x)
            top = round(anchor_y + row * step_y)
            rect = Rect(
                float(left),
                float(top),
                float(round(left + cell_width - 1)),
                float(round(top + cell_height - 1)),
            )
            obstacles.append(
                Obstacle(
                    rect,
                    0.0,
                    0.85,
                    len(obstacles),
                )
            )
    obstacles = [
        Obstacle(
            item.rect,
            item.corner_radius,
            item.confidence,
            index,
        )
        for index, item in enumerate(obstacles)
    ]
    return obstacles, mask


def _clip_line_to_rect(
    first: tuple[float, float],
    second: tuple[float, float],
    rect: Rect,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x1, y1 = first
    x2, y2 = second
    dx, dy = x2 - x1, y2 - y1
    candidates: list[tuple[float, float, float]] = []
    for x in (rect.left, rect.right):
        if abs(dx) > 1e-9:
            t = (x - x1) / dx
            y = y1 + t * dy
            if rect.top - 1 <= y <= rect.bottom + 1:
                candidates.append((t, x, y))
    for y in (rect.top, rect.bottom):
        if abs(dy) > 1e-9:
            t = (y - y1) / dy
            x = x1 + t * dx
            if rect.left - 1 <= x <= rect.right + 1:
                candidates.append((t, x, y))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item[0])
    start = candidates[0][1:]
    end = candidates[-1][1:]
    if start[1] < end[1]:
        start, end = end, start
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def _detect_aim_line(
    pale_mask: Image,
    board: Rect,
    obstacles: list[Obstacle],
    launch_hint: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    mask = pale_mask.copy()
    margin = max(4, round(board.width * 0.024))
    mask[: max(0, round(board.top + margin))] = 0
    mask[min(mask.shape[0], round(board.bottom - margin)) :] = 0
    mask[:, : max(0, round(board.left + margin))] = 0
    mask[:, min(mask.shape[1], round(board.right - margin)) :] = 0
    for obstacle in obstacles:
        rect = obstacle.rect
        cv2.rectangle(
            mask,
            (max(0, round(rect.left - 2)), max(0, round(rect.top - 2))),
            (
                min(mask.shape[1] - 1, round(rect.right + 2)),
                min(mask.shape[0] - 1, round(rect.bottom + 2)),
            ),
            0,
            -1,
        )
    # The solid arrow/character is much larger than one dotted-line segment.
    # Removing large connected pale blobs leaves a cleaner set of collinear
    # dash pixels and prevents the arrow head from flattening shallow angles.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    max_component_area = max(180.0, board.width * board.height * 0.0015)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] > max_component_area:
            mask[labels == label] = 0
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 720,
        threshold=max(12, round(board.height * 0.020)),
        minLineLength=max(24, round(board.width * 0.10)),
        maxLineGap=max(14, round(board.height * 0.040)),
    )
    if lines is None:
        return None
    anchor = np.array(
        launch_hint or (float(board.center[0]), board.bottom),
        dtype=np.float64,
    )
    anchor_limit = board.width * (0.12 if launch_hint is not None else 0.25)
    candidates: list[
        tuple[
            float,
            tuple[tuple[float, float], tuple[float, float]],
            tuple[float, float],
        ]
    ] = []
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = (float(value) for value in raw)
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < board.width * 0.08:
            continue
        unit_direction = direction / length
        if abs(unit_direction[1]) < 0.025:
            continue
        anchor_distance = abs(
            float(
                unit_direction[0] * (anchor[1] - y1)
                - unit_direction[1] * (anchor[0] - x1)
            )
        )
        if anchor_distance > anchor_limit:
            continue
        clipped = _clip_line_to_rect((x1, y1), (x2, y2), board)
        if clipped is None:
            continue
        lower, _ = clipped
        if abs(lower[1] - board.bottom) > 2.0:
            continue
        score = length - anchor_distance * 2.5
        candidates.append((score, clipped, (x1, y1)))
    if not candidates:
        return None
    _, best, best_point = max(candidates, key=lambda item: item[0])

    # Refit all dash pixels close to the best Hough segment. This turns several
    # noisy integer segments into one sub-pixel centreline.
    lower, upper = best
    fit_direction = (
        np.array(upper, dtype=np.float64) - np.array(lower, dtype=np.float64)
    )
    fit_direction /= max(1e-9, float(np.linalg.norm(fit_direction)))
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)
    if len(points) >= 12:
        relative = points - np.array(best_point, dtype=np.float64)
        distances = np.abs(
            relative[:, 0] * fit_direction[1]
            - relative[:, 1] * fit_direction[0]
        )
        nearby = points[distances <= max(3.0, board.width * 0.009)]
        if len(nearby) >= 12:
            # The hint is only for selecting the correct dotted line. Never
            # force the fitted direction through it: a stale round origin would
            # rotate the entire predicted trajectory.
            vx, vy, x0, y0 = (
                float(value)
                for value in cv2.fitLine(
                    nearby.astype(np.float32),
                    cv2.DIST_L2,
                    0,
                    0.01,
                    0.01,
                ).reshape(-1)
            )
            reach = board.diagonal * 1.5
            refined = _clip_line_to_rect(
                (x0 - vx * reach, y0 - vy * reach),
                (x0 + vx * reach, y0 + vy * reach),
                board,
            )
            if refined is not None and abs(refined[0][1] - board.bottom) <= 2.0:
                return refined
    return best


def _detect_aim_reticle_radius(
    bgr: Image,
    hsv: Image,
    board: Rect,
    aim_line: tuple[tuple[float, float], tuple[float, float]] | None,
    obstacles: list[Obstacle],
) -> float | None:
    """Measure the current aim reticle; all limits scale with the detected board."""
    if aim_line is None:
        return None
    min_radius = max(4, round(board.width * 0.018))
    max_radius = max(min_radius + 2, round(board.width * 0.050))
    endpoint = np.array(aim_line[1], dtype=np.float64)
    search_span = max(24, max_radius * 4)
    height, width = bgr.shape[:2]
    x0 = max(0, math.floor(endpoint[0] - search_span))
    y0 = max(0, math.floor(endpoint[1] - search_span))
    x1 = min(width, math.ceil(endpoint[0] + search_span) + 1)
    y1 = min(height, math.ceil(endpoint[1] + search_span) + 1)
    if x1 - x0 < min_radius * 2 or y1 - y0 < min_radius * 2:
        return None

    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.8), 55, 145)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    wall_distances = (
        abs(endpoint[0] - board.left),
        abs(endpoint[0] - board.right),
        abs(endpoint[1] - board.top),
        abs(endpoint[1] - board.bottom),
    )
    wall_index = int(np.argmin(wall_distances))
    inward_normals = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    normal = inward_normals[wall_index]
    tangent = np.array([-normal[1], normal[0]])
    angles = np.linspace(0.0, math.tau, 96, endpoint=False)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates: list[tuple[float, float]] = []
    for radius in range(min_radius, max_radius + 1):
        for tangent_offset in range(-3, 4):
            center = endpoint + normal * radius + tangent * tangent_offset
            if any(
                obstacle.rect.left - radius <= center[0] <= obstacle.rect.right + radius
                and obstacle.rect.top - radius <= center[1] <= obstacle.rect.bottom + radius
                for obstacle in obstacles
            ):
                continue
            samples = np.rint(
                center - np.array([x0, y0]) + unit_circle * radius
            ).astype(np.int32)
            valid = (
                (samples[:, 0] >= 0)
                & (samples[:, 0] < edges.shape[1])
                & (samples[:, 1] >= 0)
                & (samples[:, 1] < edges.shape[0])
            )
            if np.count_nonzero(valid) < len(samples) * 0.75:
                continue
            support = float(np.mean(edges[samples[valid, 1], samples[valid, 0]] > 0))
            candidates.append((support, float(radius)))
    if not candidates:
        return None
    support, radius = max(candidates, key=lambda item: (item[0], item[1]))
    return radius if support >= 0.22 else None


def _detect_arrow_axis(
    hsv: Image,
    board: Rect,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Use the large cream aiming arrow as a fallback for shallow dotted lines."""
    height, width = hsv.shape[:2]
    arrow_mask = np.uint8(
        (hsv[:, :, 0] >= 14)
        & (hsv[:, :, 0] <= 45)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 1] <= 180)
        & (hsv[:, :, 2] >= 190)
    ) * 255
    roi = np.zeros_like(arrow_mask)
    cv2.rectangle(
        roi,
        (
            max(0, round(board.left - board.width * 0.025)),
            max(0, round(board.bottom - board.height * 0.23)),
        ),
        (
            min(width - 1, round(board.right + board.width * 0.025)),
            min(height - 1, round(board.bottom + board.height * 0.04)),
        ),
        255,
        -1,
    )
    arrow_mask = cv2.bitwise_and(arrow_mask, roi)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(arrow_mask)
    candidates: list[
        tuple[float, tuple[tuple[float, float], tuple[float, float]]]
    ] = []
    board_area = max(1.0, board.width * board.height)
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if not board_area * 0.0015 <= area <= board_area * 0.025:
            continue
        center_x, center_y = (float(value) for value in centroids[label])
        if not (
            board.left + board.width * 0.025
            <= center_x
            <= board.right - board.width * 0.025
        ):
            continue
        if not (
            board.bottom - board.height * 0.16
            <= center_y
            <= board.bottom + board.height * 0.02
        ):
            continue
        if max(component_width, component_height) < board.width * 0.08:
            continue
        if min(component_width, component_height) < board.width * 0.032:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float64)
        if len(points) < 20:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
        major_index = int(np.argmax(eigenvalues))
        minor_index = 1 - major_index
        elongation = float(
            eigenvalues[major_index] / max(1e-6, eigenvalues[minor_index])
        )
        if elongation < 2.5:
            continue
        direction = eigenvectors[:, major_index]
        if direction[1] > 0:
            direction = -direction
        if direction[1] > -0.08:
            continue
        reach = board.diagonal * 1.5
        center = np.array([center_x, center_y], dtype=np.float64)
        first = center - direction * reach
        second = center + direction * reach
        clipped = _clip_line_to_rect(
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
            board,
        )
        if clipped is None:
            continue
        lower, upper = clipped
        # The selected arrow must extrapolate back to the launch edge, not merely
        # be a similarly coloured elongated decoration near a side wall.
        if abs(lower[1] - board.bottom) > 2.0:
            continue
        bottom_distance = abs(center_y - board.bottom)
        score = (
            math.log1p(area) * 4.0
            + min(8.0, elongation)
            - bottom_distance / max(1.0, board.height) * 12.0
        )
        candidates.append((score, clipped))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


class BoardDetector:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.aim_radius_ratios: deque[float] = deque(maxlen=5)

    def detect_aim_only(
        self,
        bgr: Image,
        board: Rect,
        obstacles: list[Obstacle] | tuple[Obstacle, ...],
        launch_origin: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        pale = np.uint8((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 205)) * 255
        arrow_line = _detect_arrow_axis(hsv, board)
        # The visible arrow belongs to the current frame and therefore wins
        # over a launch point cached when the scene was locked.
        launch_hint = arrow_line[0] if arrow_line is not None else launch_origin
        return _detect_aim_line(
            pale,
            board,
            list(obstacles),
            launch_hint,
        ) or arrow_line

    def detect(self, bgr: Image, *, debug_masks: bool = False) -> DetectionResult:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        collision_board, collision_confidence, pale_mask = _detect_collision_frame(hsv)
        water_mask = cv2.inRange(
            hsv,
            np.array(self.config.water_hsv_low, dtype=np.uint8),
            np.array(self.config.water_hsv_high, dtype=np.uint8),
        )
        water_mask = cv2.morphologyEx(
            water_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        )
        water_mask = cv2.morphologyEx(
            water_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        water_board, water_confidence = _board_from_row_spans(water_mask)
        board = collision_board or water_board
        board_confidence = (
            collision_confidence if collision_board is not None else water_confidence
        )
        if board is not None and collision_board is None:
            # A pale horizontal rim separates the pool from a same-colour header in
            # the reference game. It is a stronger top-boundary cue than hue alone.
            pale = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 180)
            left = max(0, round(board.left))
            right = min(bgr.shape[1], round(board.right) + 1)
            search_bottom = min(round(bgr.shape[0] * 0.35), round(board.bottom))
            row_counts = pale[:search_bottom, left:right].sum(axis=1)
            if row_counts.size:
                rim_y = int(np.argmax(row_counts))
                if row_counts[rim_y] >= max(20, board.width * 0.55) and rim_y > board.top:
                    board = Rect(board.left, float(rim_y + 1), board.right, board.bottom)

        obstacles: list[Obstacle] = []
        if collision_board is not None:
            obstacles, block_mask = _detect_grid_blocks(bgr, hsv, collision_board)
        else:
            block_mask = cv2.inRange(
                hsv,
                np.array(self.config.block_hsv_low, dtype=np.uint8),
                np.array(self.config.block_hsv_high, dtype=np.uint8),
            )
            # Do not close this mask: adjacent platforms can be only 1–2 px apart.
            block_mask = cv2.morphologyEx(
                block_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
            )
            contours, _ = cv2.findContours(
                block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contour_area = float(cv2.contourArea(contour))
                x, y, width, height = cv2.boundingRect(contour)
                if not (
                    self.config.min_block_area
                    <= contour_area
                    <= self.config.max_block_area
                ):
                    continue
                if not (
                    self.config.block_min_width <= width <= self.config.block_max_width
                    and self.config.block_min_height
                    <= height
                    <= self.config.block_max_height
                ):
                    continue
                if board is not None and not (
                    board.left - 5 <= x <= board.right
                    and board.top - 5 <= y <= board.bottom
                ):
                    continue
                fill = contour_area / max(1, width * height)
                confidence = max(0.2, min(1.0, (fill - 0.35) / 0.5))
                obstacles.append(
                    Obstacle(
                        Rect(
                            float(x),
                            float(y),
                            float(x + width - 1),
                            float(y + height - 1),
                        ),
                        corner_radius=0.0,
                        confidence=confidence,
                    )
                )
            obstacles.sort(key=lambda item: (item.rect.top, item.rect.left))
            obstacles = [
                Obstacle(item.rect, 0.0, item.confidence, index)
                for index, item in enumerate(obstacles)
            ]
        if board is not None and obstacles and collision_board is None:
            margin = max(2.0, bgr.shape[1] * 0.005)
            board = Rect(
                min(board.left, min(item.rect.left for item in obstacles) - margin),
                min(board.top, min(item.rect.top for item in obstacles) - margin),
                max(board.right, max(item.rect.right for item in obstacles) + margin),
                max(board.bottom, max(item.rect.bottom for item in obstacles) + margin),
            )

        launch_origin = None
        if board is not None:
            launch_origin = (
                (board.left + board.right) / 2,
                board.top + board.height * 0.94,
            )
        aim_line = None
        aim_radius = None
        if board is not None and collision_board is not None:
            arrow_line = _detect_arrow_axis(hsv, board)
            launch_hint = arrow_line[0] if arrow_line is not None else None
            aim_line = _detect_aim_line(
                pale_mask,
                board,
                obstacles,
                launch_hint,
            ) or arrow_line
            aim_radius = _detect_aim_reticle_radius(
                bgr,
                hsv,
                board,
                aim_line,
                obstacles,
            )
            if aim_radius is not None:
                # Hough sometimes locks onto the inner white ring. The outer
                # reticle-to-board ratio is the scale-invariant lower bound.
                aim_radius = max(aim_radius, board.width * 0.0345)
                self.aim_radius_ratios.append(aim_radius / board.width)
            elif self.aim_radius_ratios:
                aim_radius = (
                    float(np.median(self.aim_radius_ratios)) * board.width
                )
        overall = board_confidence
        if obstacles:
            overall = min(1.0, 0.65 * board_confidence + 0.35 * np.mean(
                [item.confidence for item in obstacles]
            ))
        return DetectionResult(
            board=board,
            obstacles=obstacles,
            launch_origin=launch_origin,
            aim_line=aim_line,
            water_mask=water_mask if debug_masks else None,
            block_mask=block_mask if debug_masks else None,
            confidence=float(overall),
            collision_frame_detected=collision_board is not None,
            aim_radius=aim_radius,
        )


class MotionAimDetector:
    """Fast cropped frame-difference channel used while the mouse is moving."""

    def __init__(self) -> None:
        self.previous_gray: Image | None = None
        self.previous_geometry: tuple[int, int, int, int] | None = None
        self.last_marker_present = False

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_geometry = None
        self.last_marker_present = False

    def detect(
        self,
        bgr: Image,
        board: Rect,
        obstacles: list[Obstacle] | tuple[Obstacle, ...],
        launch_origin: tuple[float, float],
        *,
        active: bool,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        height, width = bgr.shape[:2]
        x0 = max(0, math.floor(board.left - board.width * 0.03))
        y0 = max(0, math.floor(board.top))
        x1 = min(width, math.ceil(board.right + board.width * 0.03) + 1)
        y1 = min(height, math.ceil(board.bottom + board.height * 0.05) + 1)
        geometry = (x0, y0, x1, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            self.reset()
            return None
        crop = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        previous = (
            self.previous_gray
            if self.previous_geometry == geometry
            and self.previous_gray is not None
            and self.previous_gray.shape == gray.shape
            else None
        )
        self.previous_gray = gray.copy()
        self.previous_geometry = geometry
        local_board = Rect(
            board.left - x0,
            board.top - y0,
            board.right - x0,
            board.bottom - y0,
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        arrow_line = _detect_arrow_axis(hsv, local_board)
        self.last_marker_present = arrow_line is not None
        if not self.last_marker_present or not active or previous is None:
            return None

        difference = cv2.absdiff(gray, previous)
        motion = np.uint8(difference >= 10) * 255
        motion_kernel_size = max(3, round(board.width * 0.017))
        if motion_kernel_size % 2 == 0:
            motion_kernel_size += 1
        motion = cv2.dilate(
            motion,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (motion_kernel_size, motion_kernel_size),
            ),
            iterations=1,
        )
        pale = np.uint8((hsv[:, :, 1] < 110) & (hsv[:, :, 2] > 200)) * 255
        source = cv2.bitwise_and(pale, motion)
        minimum_motion_pixels = max(20, round(board.width * board.width * 0.00017))
        if cv2.countNonZero(source) < minimum_motion_pixels:
            return None

        local_obstacles = [
            Obstacle(
                Rect(
                    item.rect.left - x0,
                    item.rect.top - y0,
                    item.rect.right - x0,
                    item.rect.bottom - y0,
                ),
                item.corner_radius,
                item.confidence,
                item.identifier,
            )
            for item in obstacles
        ]
        local_origin = (
            arrow_line[0]
            if arrow_line is not None
            else (launch_origin[0] - x0, launch_origin[1] - y0)
        )
        line = _detect_aim_line(
            source,
            local_board,
            local_obstacles,
            local_origin,
        )
        if line is None:
            return None
        lower, upper = line
        return (
            (lower[0] + x0, lower[1] + y0),
            (upper[0] + x0, upper[1] + y0),
        )


class TemporalDetector:
    """Stabilises detections by median geometry and nearest-centre voting."""

    def __init__(self, detector: BoardDetector, frames: int = 3):
        self.detector = detector
        self.history: deque[DetectionResult] = deque(maxlen=max(1, frames))

    def detect(self, bgr: Image, *, debug_masks: bool = False) -> DetectionResult:
        current = self.detector.detect(bgr, debug_masks=debug_masks)
        if current.collision_frame_detected and current.aim_line is None:
            return current
        self.history.append(current)
        valid_boards = [result.board for result in self.history if result.board is not None]
        if not valid_boards:
            return current
        board = Rect(
            float(np.median([item.left for item in valid_boards])),
            float(np.median([item.top for item in valid_boards])),
            float(np.median([item.right for item in valid_boards])),
            float(np.median([item.bottom for item in valid_boards])),
        )
        groups: list[list[Obstacle]] = []
        tolerance = max(8.0, min(board.width, board.height) * 0.025)
        for result in self.history:
            for obstacle in result.obstacles:
                center = obstacle.rect.center
                target = next(
                    (
                        group
                        for group in groups
                        if np.linalg.norm(group[0].rect.center - center) <= tolerance
                    ),
                    None,
                )
                if target is None:
                    groups.append([obstacle])
                else:
                    target.append(obstacle)
        required = max(1, math.ceil(len(self.history) / 2))
        stable: list[Obstacle] = []
        for group in groups:
            if len(group) < required:
                continue
            stable.append(
                Obstacle(
                    Rect(
                        float(np.median([item.rect.left for item in group])),
                        float(np.median([item.rect.top for item in group])),
                        float(np.median([item.rect.right for item in group])),
                        float(np.median([item.rect.bottom for item in group])),
                    ),
                    float(np.median([item.corner_radius for item in group])),
                    float(np.mean([item.confidence for item in group])),
                    len(stable),
                )
            )
        return DetectionResult(
            board=board,
            obstacles=stable or current.obstacles,
            launch_origin=(board.center[0], board.top + board.height * 0.94),
            aim_line=current.aim_line,
            water_mask=current.water_mask,
            block_mask=current.block_mask,
            confidence=float(np.mean([item.confidence for item in self.history])),
            collision_frame_detected=current.collision_frame_detected,
            aim_radius=(
                float(np.median(
                    [
                        item.aim_radius
                        for item in self.history
                        if item.aim_radius is not None
                    ]
                ))
                if any(item.aim_radius is not None for item in self.history)
                else current.aim_radius
            ),
        )


def render_debug(bgr: Image, result: DetectionResult) -> Image:
    output = bgr.copy()
    if result.board:
        board = result.board
        cv2.rectangle(
            output,
            (round(board.left), round(board.top)),
            (round(board.right), round(board.bottom)),
            (255, 255, 0),
            2,
        )
    for obstacle in result.obstacles:
        rect = obstacle.rect
        cv2.rectangle(
            output,
            (round(rect.left), round(rect.top)),
            (round(rect.right), round(rect.bottom)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            output,
            f"BLOCK {obstacle.identifier}",
            (round(rect.left), round(rect.top) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if result.launch_origin:
        cv2.circle(
            output,
            (round(result.launch_origin[0]), round(result.launch_origin[1])),
            5,
            (255, 0, 255),
            -1,
        )
    if result.aim_line:
        start, end = result.aim_line
        cv2.line(
            output,
            (round(start[0]), round(start[1])),
            (round(end[0]), round(end[1])),
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if result.aim_radius is not None:
            start, end = result.aim_line
            direction = np.array(end) - np.array(start)
            direction /= max(1e-6, float(np.linalg.norm(direction)))
            center = np.array(end) - direction * result.aim_radius
            cv2.circle(
                output,
                (round(center[0]), round(center[1])),
                round(result.aim_radius),
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
    return output
