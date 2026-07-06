from __future__ import annotations

from marble_aim.config import AppConfig


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig(window_title="Example Game")
    config.physics.ball_radius = 8.25
    config.calibration.boundary_offsets = [1.0, -2.0, 3.0, -4.0]
    config.calibration.manual_board_normalized = [0.1, 0.2, 0.8, 0.9]
    config.save(path)

    loaded = AppConfig.load(path)
    assert loaded.window_title == "Example Game"
    assert loaded.physics.ball_radius == 8.25
    assert loaded.calibration.boundary_offsets == [1.0, -2.0, 3.0, -4.0]
    assert loaded.calibration.manual_board_normalized == [0.1, 0.2, 0.8, 0.9]


def test_missing_config_uses_defaults(tmp_path):
    loaded = AppConfig.load(tmp_path / "missing.json")
    assert loaded.physics.max_collisions == 40
    assert loaded.overlay.visible is True
    assert loaded.overlay.show_collision_frame is True
    assert loaded.vision.aim_transition_delay_ms == 600
