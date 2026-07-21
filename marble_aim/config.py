from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PhysicsConfig:
    ball_radius: float = 7.0
    ball_radius_to_block_ratio: float = 0.215
    ball_radius_to_board_width_ratio: float = 0.0345
    max_collisions: int = 40
    max_distance_factor: float = 12.0
    reflection_bias_deg: float = 0.0
    collision_epsilon: float = 0.05
    minimum_launch_elevation_deg: float = 17.1
    angle_min_deg: float = -80.0
    angle_max_deg: float = 80.0
    coarse_step_deg: float = 1.0
    fine_step_deg: float = 0.1
    recommendation_count: int = 2
    recommendation_separation_deg: float = 2.0
    volley_count: int = 67


@dataclass(slots=True)
class VisionConfig:
    water_hsv_low: list[int] = field(default_factory=lambda: [82, 70, 80])
    water_hsv_high: list[int] = field(default_factory=lambda: [112, 255, 255])
    block_hsv_low: list[int] = field(default_factory=lambda: [38, 90, 45])
    block_hsv_high: list[int] = field(default_factory=lambda: [82, 255, 255])
    min_block_area: int = 700
    max_block_area: int = 9000
    block_min_width: int = 38
    block_max_width: int = 120
    block_min_height: int = 28
    block_max_height: int = 110
    block_corner_radius_ratio: float = 0.0
    detection_interval_ms: int = 250
    aim_transition_delay_ms: int = 600
    temporal_frames: int = 3


@dataclass(slots=True)
class OverlayConfig:
    current_color: str = "#FF35F5"
    recommendation_colors: list[str] = field(
        default_factory=lambda: ["#50FF7A", "#FFE45B", "#FF9D42"]
    )
    line_width: float = 3.2
    opacity: float = 0.92
    show_collision_points: bool = True
    show_locked_boxes: bool = True
    show_collision_frame: bool = True
    visible: bool = True


@dataclass(slots=True)
class HotkeyConfig:
    toggle_overlay: str = "F7"
    pause: str = "F8"
    calibrate: str = "F9"
    settings: str = "F10"


@dataclass(slots=True)
class CalibrationConfig:
    launch_origin_normalized: list[float] = field(default_factory=lambda: [0.5, 0.94])
    boundary_offsets: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    manual_board_normalized: list[float] | None = None
    duration_seconds: int = 20
    min_track_points: int = 12
    calibrated: bool = False


@dataclass(slots=True)
class AppConfig:
    window_title: str = ""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            window_title=raw.get("window_title", ""),
            physics=PhysicsConfig(**raw.get("physics", {})),
            vision=VisionConfig(**raw.get("vision", {})),
            overlay=OverlayConfig(**raw.get("overlay", {})),
            hotkeys=HotkeyConfig(**raw.get("hotkeys", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_from_mapping(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)
