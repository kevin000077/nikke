from __future__ import annotations

import math

import numpy as np

from marble_aim.geometry import (
    Obstacle,
    Rect,
    direction_from_angle,
    estimate_ball_radius,
    ray_rounded_rect,
    reflect,
    search_recommendations,
    simulate_volley_trajectory,
    simulate_trajectory,
    vec,
)


def test_reflect_from_vertical_wall():
    result = reflect(vec(1, -0.25), vec(-1, 0))
    expected = np.array([-1, -0.25], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert np.allclose(result, expected)


def test_wall_collision_respects_ball_radius():
    board = Rect(0, 0, 100, 100)
    trajectory = simulate_trajectory(
        vec(50, 80),
        vec(0, -1),
        board,
        [],
        ball_radius=5,
        max_collisions=1,
    )
    assert len(trajectory.collisions) == 1
    assert trajectory.collisions[0].kind == "wall"
    assert np.allclose(trajectory.points[1], [50, 5])


def test_simultaneous_corner_collision_combines_normals():
    board = Rect(0, 0, 100, 100)
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(-1, -1),
        board,
        [],
        ball_radius=5,
        max_collisions=1,
    )
    collision = trajectory.collisions[0]
    assert np.allclose(collision.point, [5, 5])
    assert np.allclose(collision.normal, np.array([1, 1]) / math.sqrt(2))


def test_square_block_front_collision():
    obstacle = Obstacle(Rect(40, 35, 60, 55), corner_radius=0)
    hit = ray_rounded_rect(vec(50, 80), vec(0, -1), obstacle, ball_radius=5)
    assert hit is not None
    assert math.isclose(hit.point[1], 60, abs_tol=1e-6)
    assert np.allclose(hit.normal, [0, 1])


def test_rounded_corner_collision_has_diagonal_normal():
    obstacle = Obstacle(Rect(40, 40, 60, 60), corner_radius=5)
    hit = ray_rounded_rect(vec(25, 25), vec(1, 1), obstacle, ball_radius=2)
    assert hit is not None
    assert hit.normal[0] < 0
    assert hit.normal[1] < 0
    assert math.isclose(float(np.linalg.norm(hit.normal)), 1.0, abs_tol=1e-8)


def test_block_reflection_is_counted():
    board = Rect(0, 0, 100, 120)
    obstacle = Obstacle(Rect(40, 35, 60, 55), corner_radius=3)
    trajectory = simulate_trajectory(
        vec(50, 100),
        vec(0, -1),
        board,
        [obstacle],
        ball_radius=4,
        max_collisions=2,
        reflect_bottom=True,
    )
    assert trajectory.block_hits == 1
    assert trajectory.collisions[0].kind == "block"
    assert trajectory.collisions[1].kind == "wall"


def test_simulation_stops_at_collision_limit():
    trajectory = simulate_trajectory(
        vec(50, 50),
        direction_from_angle(35),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=3,
        max_collisions=5,
        max_distance=10000,
        reflect_bottom=True,
    )
    assert len(trajectory.collisions) == 5
    assert len(trajectory.points) == 6


def test_recommendations_are_separated():
    board = Rect(0, 0, 160, 220)
    obstacles = [
        Obstacle(Rect(35, 50, 65, 75), corner_radius=4),
        Obstacle(Rect(95, 50, 125, 75), corner_radius=4),
        Obstacle(Rect(67, 100, 93, 125), corner_radius=4),
    ]
    recommendations = search_recommendations(
        vec(80, 205),
        board,
        obstacles,
        ball_radius=4,
        max_collisions=12,
        max_distance=1600,
        coarse_step=4,
        fine_step=1,
        count=3,
        separation=2,
    )
    assert len(recommendations) == 3
    angles = [item.angle_deg for item in recommendations]
    assert all(angle is not None for angle in angles)
    assert min(
        abs(float(first) - float(second))
        for index, first in enumerate(angles)
        for second in angles[index + 1 :]
    ) >= 2


def test_bottom_is_an_open_exit_without_reflection():
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(0, 1),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=7,
        max_collisions=10,
    )
    assert len(trajectory.collisions) == 0
    assert np.allclose(trajectory.points[-1], [50, 100])


def test_side_wall_contact_uses_ball_edge_not_center():
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(-1, 0),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=7,
        max_collisions=1,
    )
    assert np.allclose(trajectory.points[1], [7, 50])
    assert np.allclose(trajectory.collisions[0].normal, [1, 0])


def test_durability_caps_effective_damage_and_tracks_coverage():
    board = Rect(0, 0, 120, 160)
    obstacle = Obstacle(
        Rect(45, 50, 75, 75),
        corner_radius=4,
        durability=1,
    )
    trajectory = simulate_trajectory(
        vec(60, 140),
        vec(0, -1),
        board,
        [obstacle],
        ball_radius=4,
        max_collisions=4,
        reflect_bottom=True,
    )
    assert trajectory.unique_block_hits == 1
    assert trajectory.effective_damage == 1
    assert trajectory.destroyed_blocks == 1


def test_volley_removes_block_after_durability_is_consumed():
    board = Rect(0, 0, 120, 160)
    obstacle = Obstacle(
        Rect(45, 50, 75, 75),
        corner_radius=4,
        durability=2,
    )
    trajectory = simulate_volley_trajectory(
        vec(60, 160),
        vec(0, -1),
        board,
        [obstacle],
        volley_count=5,
        ball_radius=4,
        max_collisions=8,
    )
    assert trajectory.unique_block_hits == 1
    assert trajectory.effective_damage == 2
    assert trajectory.destroyed_blocks == 1
    assert trajectory.stable_balls_before_change == 2


def test_ball_radius_scales_with_detected_block_size():
    board = Rect(0, 0, 430, 574)
    obstacle = Obstacle(Rect(20, 20, 89, 89))
    radius = estimate_ball_radius(board, [obstacle])
    scaled_radius = estimate_ball_radius(
        Rect(0, 0, 860, 1148),
        [Obstacle(Rect(40, 40, 178, 178))],
    )
    assert math.isclose(radius, 69 * 0.1725, rel_tol=1e-6)
    assert math.isclose(scaled_radius, radius * 2, rel_tol=1e-6)


def test_ball_radius_falls_back_to_board_scale_without_blocks():
    radius = estimate_ball_radius(Rect(0, 0, 430, 574), [])
    assert math.isclose(radius, 430 * 0.0277, rel_tol=1e-6)
