from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


def vec(x: float, y: float) -> Vec:
    return np.array([x, y], dtype=np.float64)


def unit(value: Vec) -> Vec:
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        raise ValueError("zero-length direction")
    return value / length


def clamp_launch_elevation(
    direction: Vec,
    minimum_elevation_deg: float,
) -> Vec:
    """Clamp a launch direction to the game's minimum angle above horizontal."""
    normalized = unit(direction)
    minimum = math.radians(float(np.clip(minimum_elevation_deg, 0.0, 89.9)))
    elevation = math.atan2(-float(normalized[1]), abs(float(normalized[0])))
    if elevation >= minimum:
        return normalized
    horizontal_sign = -1.0 if normalized[0] < 0 else 1.0
    return vec(
        horizontal_sign * math.cos(minimum),
        -math.sin(minimum),
    )


def estimate_ball_radius(
    board: "Rect",
    obstacles: Iterable["Obstacle"],
    *,
    block_ratio: float = 0.1725,
    board_width_ratio: float = 0.0277,
    fallback: float = 7.0,
) -> float:
    sizes = [
        min(obstacle.rect.width, obstacle.rect.height)
        for obstacle in obstacles
        if obstacle.rect.width > 0 and obstacle.rect.height > 0
    ]
    if sizes:
        return max(2.0, float(np.median(sizes)) * block_ratio)
    if board.width > 0:
        return max(2.0, board.width * board_width_ratio)
    return max(2.0, fallback)


@dataclass(frozen=True, slots=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def center(self) -> Vec:
        return vec((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def inset(self, amount: float) -> "Rect":
        return Rect(
            self.left + amount,
            self.top + amount,
            self.right - amount,
            self.bottom - amount,
        )


@dataclass(frozen=True, slots=True)
class Obstacle:
    rect: Rect
    corner_radius: float = 8.0
    confidence: float = 1.0
    identifier: int = -1
    durability: int | None = None
    durability_confidence: float = 0.0


@dataclass(slots=True)
class Collision:
    distance: float
    point: Vec
    normal: Vec
    kind: str
    obstacle_index: int | None = None
    confidence: float = 1.0
    grazing: bool = False


@dataclass(slots=True)
class Trajectory:
    points: list[Vec]
    collisions: list[Collision] = field(default_factory=list)
    angle_deg: float | None = None
    score: float = 0.0
    looped: bool = False
    unique_block_hits: int = 0
    effective_damage: int = 0
    destroyed_blocks: int = 0
    stable_balls_before_change: int = 0

    @property
    def block_hits(self) -> int:
        return sum(c.kind == "block" for c in self.collisions)

    @property
    def confidence(self) -> float:
        block_confidences = [c.confidence for c in self.collisions if c.kind == "block"]
        if not block_confidences:
            return 0.5
        return float(sum(block_confidences) / len(block_confidences))


def reflect(direction: Vec, normal: Vec, bias_deg: float = 0.0) -> Vec:
    direction = unit(direction)
    normal = unit(normal)
    reflected = direction - 2.0 * float(np.dot(direction, normal)) * normal
    if abs(bias_deg) > 1e-9:
        theta = math.radians(bias_deg)
        c, s = math.cos(theta), math.sin(theta)
        reflected = np.array(
            [c * reflected[0] - s * reflected[1], s * reflected[0] + c * reflected[1]],
            dtype=np.float64,
        )
    return unit(reflected)


def _ray_circle(
    origin: Vec, direction: Vec, center: Vec, radius: float, epsilon: float
) -> tuple[float, Vec, Vec] | None:
    offset = origin - center
    b = 2.0 * float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant < 0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [(-b - root) / 2.0, (-b + root) / 2.0]
    for distance in candidates:
        if distance > epsilon:
            point = origin + direction * distance
            return distance, point, unit(point - center)
    return None


def ray_rounded_rect(
    origin: Vec,
    direction: Vec,
    obstacle: Obstacle,
    ball_radius: float,
    epsilon: float = 1e-5,
) -> Collision | None:
    """First contact of a moving circle with an axis-aligned rounded rectangle."""
    direction = unit(direction)
    rect = obstacle.rect
    base_radius = max(0.0, min(obstacle.corner_radius, rect.width / 2, rect.height / 2))
    radius = base_radius + ball_radius
    left = rect.left - ball_radius
    right = rect.right + ball_radius
    top = rect.top - ball_radius
    bottom = rect.bottom + ball_radius
    cx_left, cx_right = rect.left + base_radius, rect.right - base_radius
    cy_top, cy_bottom = rect.top + base_radius, rect.bottom - base_radius
    candidates: list[tuple[float, Vec, Vec]] = []

    def add_side(axis: int, value: float, low: float, high: float, normal: Vec) -> None:
        component = float(direction[axis])
        if abs(component) < 1e-12:
            return
        distance = (value - float(origin[axis])) / component
        if distance <= epsilon:
            return
        point = origin + direction * distance
        other = float(point[1 - axis])
        if low - 1e-6 <= other <= high + 1e-6 and float(np.dot(direction, normal)) < 0:
            candidates.append((distance, point, normal))

    add_side(0, left, cy_top, cy_bottom, vec(-1, 0))
    add_side(0, right, cy_top, cy_bottom, vec(1, 0))
    add_side(1, top, cx_left, cx_right, vec(0, -1))
    add_side(1, bottom, cx_left, cx_right, vec(0, 1))

    corners = [
        (vec(cx_left, cy_top), lambda p: p[0] <= cx_left and p[1] <= cy_top),
        (vec(cx_right, cy_top), lambda p: p[0] >= cx_right and p[1] <= cy_top),
        (vec(cx_left, cy_bottom), lambda p: p[0] <= cx_left and p[1] >= cy_bottom),
        (vec(cx_right, cy_bottom), lambda p: p[0] >= cx_right and p[1] >= cy_bottom),
    ]
    for center, valid_quadrant in corners:
        hit = _ray_circle(origin, direction, center, radius, epsilon)
        if hit is not None and valid_quadrant(hit[1]) and float(np.dot(direction, hit[2])) < 0:
            candidates.append(hit)

    if not candidates:
        return None
    distance, point, normal = min(candidates, key=lambda item: item[0])
    incidence = abs(float(np.dot(direction, normal)))
    return Collision(
        distance=distance,
        point=point,
        normal=normal,
        kind="block",
        confidence=obstacle.confidence,
        grazing=incidence < 0.2,
    )


def ray_inner_rect(
    origin: Vec,
    direction: Vec,
    board: Rect,
    ball_radius: float,
    epsilon: float = 1e-5,
    *,
    reflect_bottom: bool = True,
) -> Collision | None:
    direction = unit(direction)
    inner = board.inset(ball_radius)
    candidates: list[tuple[float, Vec, Vec]] = []
    walls = [
        (0, inner.left, vec(1, 0), inner.top, inner.bottom),
        (0, inner.right, vec(-1, 0), inner.top, inner.bottom),
        (1, inner.top, vec(0, 1), inner.left, inner.right),
    ]
    if reflect_bottom:
        walls.append((1, inner.bottom, vec(0, -1), inner.left, inner.right))
    for axis, value, normal, low, high in walls:
        component = float(direction[axis])
        if abs(component) < 1e-12:
            continue
        distance = (value - float(origin[axis])) / component
        if distance <= epsilon:
            continue
        point = origin + direction * distance
        other = float(point[1 - axis])
        if low - 1e-6 <= other <= high + 1e-6 and float(np.dot(direction, normal)) < 0:
            candidates.append((distance, point, normal))
    if not candidates:
        return None
    distance, point, normal = min(candidates, key=lambda item: item[0])
    simultaneous = [
        candidate for candidate in candidates if abs(candidate[0] - distance) <= 1e-4
    ]
    if len(simultaneous) > 1:
        combined = np.sum([candidate[2] for candidate in simultaneous], axis=0)
        if float(np.linalg.norm(combined)) > 1e-9:
            normal = unit(combined)
    return Collision(distance, point, normal, "wall")


def ray_bottom_exit(
    origin: Vec,
    direction: Vec,
    board: Rect,
    epsilon: float = 1e-5,
) -> tuple[float, Vec] | None:
    """Return where the ball centre leaves through the non-reflective bottom."""
    direction = unit(direction)
    if direction[1] <= 1e-12:
        return None
    distance = (board.bottom - float(origin[1])) / float(direction[1])
    if distance <= epsilon:
        return None
    point = origin + direction * distance
    if board.left - 1e-6 <= point[0] <= board.right + 1e-6:
        return distance, point
    return None


def simulate_trajectory(
    origin: Vec,
    direction: Vec,
    board: Rect,
    obstacles: Iterable[Obstacle],
    *,
    ball_radius: float = 7.0,
    max_collisions: int = 40,
    max_distance: float | None = None,
    reflection_bias_deg: float = 0.0,
    epsilon: float = 0.05,
    reflect_bottom: bool = False,
) -> Trajectory:
    obstacles = list(obstacles)
    direction = unit(direction)
    current = np.array(origin, dtype=np.float64)
    result = Trajectory(points=[current.copy()])
    travelled = 0.0
    max_distance = max_distance or board.diagonal * 12.0
    visited: dict[tuple[int, int, int], int] = {}

    for _ in range(max_collisions):
        wall_hit = ray_inner_rect(
            current,
            direction,
            board,
            ball_radius,
            epsilon,
            reflect_bottom=reflect_bottom,
        )
        hits: list[Collision] = [wall_hit] if wall_hit else []
        for index, obstacle in enumerate(obstacles):
            hit = ray_rounded_rect(current, direction, obstacle, ball_radius, epsilon)
            if hit is not None:
                hit.obstacle_index = index
                hits.append(hit)
        bottom_exit = (
            None
            if reflect_bottom
            else ray_bottom_exit(current, direction, board, epsilon)
        )
        nearest_hit_distance = (
            min(candidate.distance for candidate in hits) if hits else math.inf
        )
        if bottom_exit is not None and bottom_exit[0] < nearest_hit_distance:
            remaining = max_distance - travelled
            if bottom_exit[0] <= remaining:
                result.points.append(bottom_exit[1].copy())
            elif remaining > 0:
                result.points.append(current + direction * remaining)
            break
        if not hits:
            break
        hit = min(hits, key=lambda candidate: candidate.distance)
        simultaneous = [
            candidate
            for candidate in hits
            if abs(candidate.distance - hit.distance) <= 1e-4
        ]
        if len(simultaneous) > 1:
            combined = np.sum([candidate.normal for candidate in simultaneous], axis=0)
            if float(np.linalg.norm(combined)) > 1e-9:
                blocks = [candidate for candidate in simultaneous if candidate.kind == "block"]
                primary = blocks[0] if blocks else hit
                hit = Collision(
                    distance=hit.distance,
                    point=hit.point,
                    normal=unit(combined),
                    kind=primary.kind,
                    obstacle_index=primary.obstacle_index,
                    confidence=min(candidate.confidence for candidate in simultaneous),
                    grazing=any(candidate.grazing for candidate in simultaneous),
                )
        if travelled + hit.distance > max_distance:
            remaining = max_distance - travelled
            if remaining > 0:
                result.points.append(current + direction * remaining)
            break
        travelled += hit.distance
        result.points.append(hit.point.copy())
        result.collisions.append(hit)
        outgoing = reflect(direction, hit.normal, reflection_bias_deg)
        state = (
            round(float(hit.point[0]) / 2),
            round(float(hit.point[1]) / 2),
            round(math.degrees(math.atan2(outgoing[1], outgoing[0]))),
        )
        visited[state] = visited.get(state, 0) + 1
        if visited[state] >= 3:
            result.looped = True
            break
        current = hit.point + outgoing * epsilon
        direction = outgoing

    hit_counts: dict[int, int] = {}
    for collision in result.collisions:
        if collision.kind == "block" and collision.obstacle_index is not None:
            hit_counts[collision.obstacle_index] = (
                hit_counts.get(collision.obstacle_index, 0) + 1
            )
    result.unique_block_hits = len(hit_counts)
    progress = 0.0
    for obstacle_index, hits in hit_counts.items():
        durability = obstacles[obstacle_index].durability
        if durability is None:
            result.effective_damage += hits
            continue
        useful_hits = min(hits, max(1, durability))
        result.effective_damage += useful_hits
        progress += min(1.0, hits / max(1, durability))
        if hits >= durability:
            result.destroyed_blocks += 1
    grazing = sum(c.grazing for c in result.collisions if c.kind == "block")
    low_confidence_penalty = sum(
        (1.0 - c.confidence) * 0.5 for c in result.collisions if c.kind == "block"
    )
    result.score = (
        result.unique_block_hits * 100.0
        + result.effective_damage
        + progress * 10.0
        + result.destroyed_blocks * 8.0
        - grazing * 0.25
        - low_confidence_penalty
    )
    if result.looped:
        result.score -= 0.5
    return result


def direction_from_angle(angle_deg: float) -> Vec:
    radians = math.radians(angle_deg)
    return vec(math.sin(radians), -math.cos(radians))


def simulate_volley_trajectory(
    origin: Vec,
    direction: Vec,
    board: Rect,
    obstacles: Iterable[Obstacle],
    *,
    volley_count: int,
    **simulation_options: float | int | bool,
) -> Trajectory:
    """Score one angle across a volley while removing depleted blocks in batches."""
    original = list(obstacles)
    # Unknown digits are treated conservatively as fragile. This prevents an
    # unreadable block from being assumed indestructible and inflating a route.
    remaining_hp = {
        index: obstacle.durability if obstacle.durability is not None else 1
        for index, obstacle in enumerate(original)
    }
    active_indices = list(range(len(original)))
    balls_left = max(1, int(volley_count))
    balls_processed = 0
    unique_damaged: set[int] = set()
    effective_damage = 0
    destroyed: set[int] = set()
    first_change: int | None = None
    first_trajectory: Trajectory | None = None

    while balls_left > 0 and active_indices:
        active = [original[index] for index in active_indices]
        trajectory = simulate_trajectory(
            origin,
            direction,
            board,
            active,
            **simulation_options,
        )
        if first_trajectory is None:
            first_trajectory = trajectory
        hit_counts: dict[int, int] = {}
        for collision in trajectory.collisions:
            if collision.kind != "block" or collision.obstacle_index is None:
                continue
            original_index = active_indices[collision.obstacle_index]
            hit_counts[original_index] = hit_counts.get(original_index, 0) + 1
        if not hit_counts:
            break

        safe_batches: list[int] = []
        for index, hits in hit_counts.items():
            hp = remaining_hp[index]
            safe_batches.append(max(0, (hp - 1) // max(1, hits)))
        if safe_batches:
            batch = min(balls_left, min(safe_batches))
        else:
            batch = balls_left
        if batch <= 0:
            batch = 1

        for index, hits in hit_counts.items():
            unique_damaged.add(index)
            total_hits = hits * batch
            hp = remaining_hp[index]
            useful = min(total_hits, hp)
            effective_damage += useful
            remaining_hp[index] = hp - useful
        balls_processed += batch
        balls_left -= batch

        newly_destroyed = {
            index for index, hp in remaining_hp.items() if hp <= 0 and index in active_indices
        }
        if newly_destroyed:
            destroyed.update(newly_destroyed)
            if first_change is None:
                first_change = balls_processed
            active_indices = [
                index for index in active_indices if index not in newly_destroyed
            ]

    if first_trajectory is None:
        first_trajectory = simulate_trajectory(
            origin,
            direction,
            board,
            original,
            **simulation_options,
        )
    first_trajectory.unique_block_hits = len(unique_damaged)
    first_trajectory.effective_damage = effective_damage
    first_trajectory.destroyed_blocks = len(destroyed)
    first_trajectory.stable_balls_before_change = (
        first_change if first_change is not None else max(1, volley_count)
    )
    first_trajectory.score = (
        len(unique_damaged) * 1000.0
        + min(first_trajectory.stable_balls_before_change, volley_count) * 10.0
        + effective_damage
        + len(destroyed) * 25.0
    )
    return first_trajectory


def search_recommendations(
    origin: Vec,
    board: Rect,
    obstacles: Iterable[Obstacle],
    *,
    angle_min: float = -80.0,
    angle_max: float = 80.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
    count: int = 3,
    separation: float = 2.0,
    volley_count: int = 1,
    **simulation_options: float | int,
) -> list[Trajectory]:
    obstacles = list(obstacles)

    def evaluate(angle: float) -> Trajectory:
        if volley_count > 1:
            trajectory = simulate_volley_trajectory(
                origin,
                direction_from_angle(angle),
                board,
                obstacles,
                volley_count=volley_count,
                **simulation_options,
            )
        else:
            trajectory = simulate_trajectory(
                origin,
                direction_from_angle(angle),
                board,
                obstacles,
                **simulation_options,
            )
        trajectory.angle_deg = angle
        return trajectory

    coarse_angles = np.arange(angle_min, angle_max + coarse_step / 2, coarse_step)
    coarse = [evaluate(float(angle)) for angle in coarse_angles]
    seeds = sorted(coarse, key=lambda item: (item.score, item.confidence), reverse=True)[
        : max(12, count * 5)
    ]
    fine_angles: set[float] = set()
    for seed in seeds:
        assert seed.angle_deg is not None
        start = max(angle_min, seed.angle_deg - coarse_step)
        stop = min(angle_max, seed.angle_deg + coarse_step)
        for angle in np.arange(start, stop + fine_step / 2, fine_step):
            fine_angles.add(round(float(angle), 6))
    candidates = coarse + [evaluate(angle) for angle in sorted(fine_angles)]
    candidates.sort(key=lambda item: (item.score, item.confidence), reverse=True)
    selected: list[Trajectory] = []
    for candidate in candidates:
        assert candidate.angle_deg is not None
        if all(
            abs(candidate.angle_deg - existing.angle_deg) >= separation
            for existing in selected
            if existing.angle_deg is not None
        ):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected
