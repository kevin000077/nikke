from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox, QWidget

from .calibration import MotionBallTracker, fit_calibration
from .capture import HotkeyPoller, WindowCapture, list_visible_windows
from .config import AppConfig
from .geometry import (
    Obstacle,
    Rect,
    Trajectory,
    estimate_ball_radius,
    search_recommendations,
    simulate_trajectory,
    unit,
    vec,
)
from .overlay import OverlayWindow, RefreshButton, SettingsDialog
from .vision import (
    BoardDetector,
    DetectionResult,
    MotionAimDetector,
    TemporalDetector,
    render_debug,
)


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    board: Rect
    obstacles: tuple[Obstacle, ...]
    origin: tuple[float, float]
    ball_radius: float


def choose_window_title(parent: QWidget | None = None) -> str | None:
    windows = list_visible_windows()
    if not windows:
        QMessageBox.critical(parent, "弹珠轨迹助手", "没有找到可捕获的可见窗口。")
        return None
    titles = [title for _, title in windows]
    title, accepted = QInputDialog.getItem(
        parent,
        "选择游戏窗口",
        "请选择游戏所在窗口：",
        titles,
        0,
        False,
    )
    return str(title) if accepted and title else None


class ApplicationController(QObject):
    def __init__(
        self,
        app: QApplication,
        config: AppConfig,
        config_path: Path,
        *,
        debug_view: bool = False,
        start_calibration: bool = False,
    ):
        super().__init__()
        self.app = app
        self.config = config
        self.config_path = config_path
        self.debug_view = debug_view
        self.start_calibration_requested = start_calibration
        self.capture = WindowCapture(config.window_title)
        self.detector = TemporalDetector(
            BoardDetector(config.vision), config.vision.temporal_frames
        )
        self.motion_aim_detector = MotionAimDetector()
        self.overlay = OverlayWindow(config)
        self.overlay.debug_view = debug_view
        self.refresh_button = RefreshButton()
        self.refresh_button.refresh_requested.connect(self.refresh_scene)
        self.refresh_button.select_window_requested.connect(self.select_target_window)
        self.refresh_button.exit_requested.connect(self.app.quit)
        self.hotkeys = HotkeyPoller(
            [
                config.hotkeys.toggle_overlay,
                config.hotkeys.pause,
                config.hotkeys.calibrate,
                config.hotkeys.settings,
            ]
        )
        self.tracker = MotionBallTracker()
        self.calibration_deadline: float | None = None
        self.paused = False
        self.last_detection_at = 0.0
        self.last_debug_save_at = 0.0
        self.latest_detection: DetectionResult | None = None
        self.latest_scene: SceneSnapshot | None = None
        self.current_trajectory: Trajectory | None = None
        self.recommendations: list[Trajectory] = []
        self.recommendation_signature: tuple[object, ...] | None = None
        self.stable_scene_signature: tuple[object, ...] | None = None
        self.pending_scene_signature: tuple[object, ...] | None = None
        self.pending_scene_count = 0
        self.recommendation_cache: dict[
            tuple[object, ...], list[Trajectory]
        ] = {}
        self.recommendation_future_signature: tuple[object, ...] | None = None
        self.fast_aim_direction: np.ndarray | None = None
        self.fast_aim_angle_samples: deque[float] = deque(maxlen=12)
        self.fast_aim_locked_angle: float | None = None
        self.last_cursor_position: tuple[int, int] | None = None
        self.cursor_motion_frames = 0
        self.fast_origin_x: float | None = None
        self.fast_origin_samples: deque[float] = deque(maxlen=7)
        self.fast_aim_line: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None
        self.aim_missing_frames = 0
        self.aim_marker_missing_frames = 0
        self.scene_locked = False
        self.locked_block_signature: tuple[object, ...] | None = None
        self.aim_was_absent = False
        self.transition_check_active = False
        self.transition_same_count = 0
        self.aim_missing_since: float | None = None
        self.aim_reappeared_since: float | None = None
        self.last_client_size: tuple[int, int] | None = None
        self.rounds_completed = 0
        self.round_candidate_signature: tuple[object, ...] | None = None
        self.round_candidate_count = 0
        self.round_candidate_origins: list[float] = []
        self.shot_missing_frames = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory")
        self.recommendation_future: Future[list[Trajectory]] | None = None
        self.status = "正在定位游戏画面…"
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.app.aboutToQuit.connect(self.close)

    def start(self) -> None:
        frame, geometry = self.capture.grab()
        self.overlay.setGeometry(
            geometry.left, geometry.top, geometry.width, geometry.height
        )
        self.refresh_button.move(geometry.left + 8, geometry.top + 8)
        self.refresh_button.show()
        self.last_client_size = (geometry.width, geometry.height)
        if self.config.overlay.visible:
            self.overlay.show()
        self._detect(frame)
        self.timer.start()

    def close(self) -> None:
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.capture.close()
        self.refresh_button.close()
        self.config.save(self.config_path)

    def _physics_options(self) -> dict[str, float | int]:
        physics = self.config.physics
        return {
            "ball_radius": (
                self.latest_scene.ball_radius
                if self.latest_scene is not None
                else physics.ball_radius
            ),
            "max_collisions": physics.max_collisions,
            "max_distance": (
                self.latest_scene.board.diagonal * physics.max_distance_factor
                if self.latest_scene
                else None
            ),
            "reflection_bias_deg": physics.reflection_bias_deg,
            "epsilon": physics.collision_epsilon,
        }

    def _adjust_board(self, board: Rect) -> Rect:
        left, right, top, bottom = self.config.calibration.boundary_offsets
        return Rect(
            board.left + left,
            board.top + top,
            board.right + right,
            board.bottom + bottom,
        )

    def _make_scene(self, detection: DetectionResult) -> SceneSnapshot | None:
        if detection.board is None:
            return None
        board = self._adjust_board(detection.board)
        normalized = self.config.calibration.launch_origin_normalized
        launch_x = board.left + board.width * normalized[0]
        launch_y = board.bottom
        if detection.aim_line is not None:
            lower, _ = detection.aim_line
            launch_x = lower[0]
        # The launch point is frozen after the multi-frame scene lock, so it can
        # retain sub-pixel precision without introducing per-frame jitter.
        origin = (float(launch_x), launch_y)
        if detection.aim_radius is not None:
            ball_radius = float(np.clip(
                detection.aim_radius,
                board.width * 0.015,
                board.width * 0.060,
            ))
        else:
            ball_radius = estimate_ball_radius(
                board,
                detection.obstacles,
                block_ratio=self.config.physics.ball_radius_to_block_ratio,
                board_width_ratio=self.config.physics.ball_radius_to_board_width_ratio,
                fallback=self.config.physics.ball_radius,
            )
        return SceneSnapshot(
            board,
            tuple(detection.obstacles),
            origin,
            ball_radius,
        )

    def _scene_signature(self, scene: SceneSnapshot) -> tuple[object, ...]:
        return (
            round(scene.board.left / 3.0),
            round(scene.board.top / 3.0),
            round(scene.board.right / 3.0),
            round(scene.board.bottom / 3.0),
            round(scene.ball_radius, 1),
            round(self.config.physics.reflection_bias_deg, 2),
            self.config.physics.volley_count,
            self._block_signature(scene),
        )

    def _block_signature(self, scene: SceneSnapshot) -> tuple[object, ...]:
        step = max(1.0, scene.board.width / 6.0)
        cells: list[tuple[int, int]] = []
        for obstacle in scene.obstacles:
            cells.append(
                (
                    round((obstacle.rect.top - scene.board.top) / step),
                    round((obstacle.rect.left - scene.board.left) / step),
                )
            )
        return (len(cells), *sorted(cells))

    def _detect(self, frame: np.ndarray) -> None:
        detection = self.detector.detect(frame, debug_masks=self.debug_view)
        self.latest_detection = detection
        if detection.aim_line is None:
            self.round_candidate_signature = None
            self.round_candidate_count = 0
            self.current_trajectory = None
            self.status = "等待新回合瞄准线｜出现后识别并冻结一次方块场景"
            return
        scene = self._make_scene(detection)
        if scene is None:
            self.status = "未识别到水池边界；按 F10 调整颜色阈值或确认窗口无遮挡。"
            return
        candidate_signature = self._scene_signature(scene)
        if candidate_signature == self.round_candidate_signature:
            self.round_candidate_count += 1
            self.round_candidate_origins.append(scene.origin[0])
        else:
            self.round_candidate_signature = candidate_signature
            self.round_candidate_count = 1
            self.round_candidate_origins = [scene.origin[0]]
        required_frames = 2
        if self.round_candidate_count >= required_frames:
            stable_origin_x = float(np.median(self.round_candidate_origins))
            scene = SceneSnapshot(
                scene.board,
                scene.obstacles,
                (stable_origin_x, scene.board.bottom),
                scene.ball_radius,
            )
            self.latest_scene = scene
            self.stable_scene_signature = candidate_signature
            self.scene_locked = True
            self.locked_block_signature = self._block_signature(scene)
            self.aim_was_absent = False
            self.transition_check_active = False
            self.transition_same_count = 0
            self.aim_missing_since = None
            self.aim_reappeared_since = None
            self.recommendations = []
            self.shot_missing_frames = 0
            self.fast_aim_angle_samples.clear()
            self.fast_origin_samples.clear()
            self.fast_origin_x = None
            self.fast_aim_locked_angle = None
            self.motion_aim_detector.reset()
            self.aim_marker_missing_frames = 0
            self._accept_fast_aim(detection.aim_line)
            if self.config.overlay.visible:
                self.overlay.show()
        else:
            self.status = (
                f"正在确认下一回合静止场景 "
                f"{self.round_candidate_count}/{required_frames}"
            )
            return
        self.status = (
            f"本回合场景已冻结｜{len(scene.obstacles)} 个方块｜"
            f"球半径 {scene.ball_radius:.1f}px｜"
            f"手控轨迹快速刷新｜置信度 {detection.confidence:.0%}｜"
            "F7显示 F8暂停 F9校准 F10设置"
        )
        if self.debug_view and time.monotonic() - self.last_debug_save_at >= 1.0:
            debug_dir = self.config_path.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "latest_detection.png"), render_debug(frame, detection))
            self.last_debug_save_at = time.monotonic()

    def _check_scene_transition(self, frame: np.ndarray) -> str:
        """Relock only for changed blocks plus a stable new aiming state."""
        detection = self.detector.detect(frame, debug_masks=self.debug_view)
        if detection.board is None or detection.aim_line is None:
            self.round_candidate_signature = None
            self.round_candidate_count = 0
            self.round_candidate_origins = []
            self.status = "弹珠飞行中｜保持本回合冻结场景"
            return "waiting"
        scene = self._make_scene(detection)
        if scene is None:
            return "waiting"
        # Automatic round changes reuse the initially locked collision frame.
        # Only the explicit "restart recognition" button (or a window resize)
        # is allowed to detect the outer frame again.
        if self.latest_scene is not None:
            locked_scene = self.latest_scene
            scene = SceneSnapshot(
                locked_scene.board,
                scene.obstacles,
                (scene.origin[0], locked_scene.board.bottom),
                locked_scene.ball_radius,
            )
        block_signature = self._block_signature(scene)
        blocks_changed = block_signature != self.locked_block_signature
        origin_changed = (
            self.latest_scene is None
            or abs(scene.origin[0] - self.latest_scene.origin[0]) >= 6.0
        )
        aim_appeared = detection.aim_line is not None
        if not blocks_changed or not (aim_appeared or origin_changed):
            self.round_candidate_signature = None
            self.round_candidate_count = 0
            self.round_candidate_origins = []
            self.transition_same_count += 1
            self.status = (
                "新瞄准线已出现：正在用干净画面比较方块变化 "
                f"{self.transition_same_count}/4"
            )
            return "same" if self.transition_same_count >= 4 else "pending"
        self.transition_same_count = 0
        candidate_signature = self._scene_signature(scene)
        if candidate_signature == self.round_candidate_signature:
            self.round_candidate_count += 1
            self.round_candidate_origins.append(scene.origin[0])
        else:
            self.round_candidate_signature = candidate_signature
            self.round_candidate_count = 1
            self.round_candidate_origins = [scene.origin[0]]
        required_frames = 2
        if self.round_candidate_count < required_frames:
            self.status = (
                f"检测到新方块场景，正在确认 "
                f"{self.round_candidate_count}/{required_frames}"
            )
            return "pending"
        stable_origin_x = float(np.median(self.round_candidate_origins))
        scene = SceneSnapshot(
            scene.board,
            scene.obstacles,
            (stable_origin_x, scene.board.bottom),
            scene.ball_radius,
        )
        self.latest_detection = detection
        self.latest_scene = scene
        self.locked_block_signature = self._block_signature(scene)
        self.stable_scene_signature = self._scene_signature(scene)
        self.shot_missing_frames = 0
        self.rounds_completed += 1
        self.round_candidate_signature = None
        self.round_candidate_count = 0
        self.round_candidate_origins = []
        self.detector.history.clear()
        self.fast_aim_angle_samples.clear()
        self.fast_origin_samples.clear()
        self.fast_origin_x = None
        self.fast_aim_locked_angle = None
        self.motion_aim_detector.reset()
        self.aim_marker_missing_frames = 0
        self._accept_fast_aim(detection.aim_line)
        self.aim_was_absent = False
        self.transition_check_active = False
        self.transition_same_count = 0
        self.aim_missing_since = None
        self.aim_reappeared_since = None
        self.status = (
            f"新回合已锁定｜{len(scene.obstacles)} 个方块｜"
            f"球半径 {scene.ball_radius:.1f}px｜手控轨迹快速刷新"
        )
        return "relocked"

    def _search_for_scene(self, scene: SceneSnapshot) -> list[Trajectory]:
        physics = self.config.physics
        options = {
            "ball_radius": physics.ball_radius,
            "max_collisions": physics.max_collisions,
            "max_distance": scene.board.diagonal * physics.max_distance_factor,
            "reflection_bias_deg": physics.reflection_bias_deg,
            "epsilon": physics.collision_epsilon,
        }
        return search_recommendations(
            vec(*scene.origin),
            scene.board,
            scene.obstacles,
            angle_min=physics.angle_min_deg,
            angle_max=physics.angle_max_deg,
            coarse_step=physics.coarse_step_deg,
            fine_step=physics.fine_step_deg,
            count=2,
            separation=physics.recommendation_separation_deg,
            volley_count=physics.volley_count,
            **options,
        )

    def _accept_fast_aim(
        self,
        line: tuple[tuple[float, float], tuple[float, float]],
        *,
        fast_response: bool = False,
    ) -> None:
        lower, upper = line
        candidate = unit(vec(*upper) - vec(*lower))
        if candidate[1] >= -0.05:
            return
        candidate_angle = math.atan2(float(candidate[1]), float(candidate[0]))
        if self.cursor_motion_frames > 0:
            self.fast_aim_locked_angle = None
        if self.fast_aim_direction is not None:
            current_angle = math.atan2(
                float(self.fast_aim_direction[1]),
                float(self.fast_aim_direction[0]),
            )
            raw_delta_deg = abs(
                math.degrees(
                    math.atan2(
                        math.sin(candidate_angle - current_angle),
                        math.cos(candidate_angle - current_angle),
                    )
                )
            )
            if raw_delta_deg >= 4.0:
                self.fast_aim_angle_samples.clear()
                self.fast_aim_locked_angle = None
        self.fast_aim_angle_samples.append(candidate_angle)
        recent = list(self.fast_aim_angle_samples)
        target_angle = (
            candidate_angle
            if fast_response
            else float(np.median(recent[-5:]))
        )
        if self.fast_aim_direction is None:
            filtered_angle = target_angle
        else:
            current_angle = math.atan2(
                float(self.fast_aim_direction[1]),
                float(self.fast_aim_direction[0]),
            )
            delta = math.atan2(
                math.sin(target_angle - current_angle),
                math.cos(target_angle - current_angle),
            )
            delta_deg = abs(math.degrees(delta))
            if self.fast_aim_locked_angle is not None:
                locked_delta = math.atan2(
                    math.sin(target_angle - self.fast_aim_locked_angle),
                    math.cos(target_angle - self.fast_aim_locked_angle),
                )
                if (
                    self.cursor_motion_frames == 0
                    or abs(math.degrees(locked_delta)) <= 0.38
                ):
                    filtered_angle = self.fast_aim_locked_angle
                    delta_deg = 0.0
                else:
                    self.fast_aim_locked_angle = None
            # A five-frame median removes pixel-level Hough chatter. The small
            # hysteresis band keeps a held aim completely still, while sustained
            # mouse movement accumulates past the band and remains responsive.
            if self.fast_aim_locked_angle is None:
                deadband_deg = 0.04 if fast_response else 0.14
                if delta_deg <= deadband_deg:
                    filtered_angle = current_angle
                elif delta_deg >= 6.0:
                    filtered_angle = target_angle
                else:
                    effective_delta = math.copysign(
                        math.radians(delta_deg - deadband_deg),
                        delta,
                    )
                    alpha = (
                        float(np.clip(0.58 + delta_deg * 0.10, 0.58, 0.94))
                        if fast_response
                        else float(np.clip(0.20 + delta_deg * 0.22, 0.20, 0.86))
                    )
                    filtered_angle = current_angle + effective_delta * alpha
                if len(recent) >= 10:
                    unwrapped = np.unwrap(np.array(recent[-10:], dtype=np.float64))
                    spread_deg = math.degrees(float(np.ptp(unwrapped)))
                    drift_deg = abs(
                        math.degrees(
                            float(
                                np.median(unwrapped[-3:])
                                - np.median(unwrapped[:3])
                            )
                        )
                    )
                    if spread_deg <= 0.24 and drift_deg <= 0.10:
                        self.fast_aim_locked_angle = filtered_angle
                if (
                    self.fast_aim_locked_angle is None
                    and self.cursor_motion_frames == 0
                    and len(recent) >= 5
                ):
                    short_window = np.unwrap(
                        np.array(recent[-5:], dtype=np.float64)
                    )
                    if math.degrees(float(np.ptp(short_window))) <= 0.45:
                        self.fast_aim_locked_angle = filtered_angle
        self.fast_aim_direction = vec(
            math.cos(filtered_angle),
            math.sin(filtered_angle),
        )
        # Recompute the unique launch point from the current line/bottom-edge
        # intersection. Blocks stay scene-locked, but the origin must not remain
        # tied to the previous round. A short median suppresses Hough chatter.
        origin_x = float(lower[0])
        if self.latest_scene is not None:
            board = self.latest_scene.board
            if board.left - 2.0 <= origin_x <= board.right + 2.0:
                if self.fast_origin_samples:
                    previous = float(np.median(self.fast_origin_samples))
                    if abs(origin_x - previous) > board.width * 0.08:
                        self.fast_origin_samples.clear()
                self.fast_origin_samples.append(origin_x)
                self.fast_origin_x = float(np.median(self.fast_origin_samples))
        else:
            self.fast_origin_x = origin_x
        self.fast_aim_line = line
        self.aim_missing_frames = 0

    def _update_fast_aim(self, frame: np.ndarray) -> bool:
        if self.latest_scene is None:
            return False
        motion_line = self.motion_aim_detector.detect(
            frame,
            self.latest_scene.board,
            self.latest_scene.obstacles,
            self.latest_scene.origin,
            active=self.cursor_motion_frames > 0,
        )
        if not self.motion_aim_detector.last_marker_present:
            self.aim_marker_missing_frames += 1
            if self.aim_marker_missing_frames >= 2:
                self.fast_aim_direction = None
                self.fast_aim_angle_samples.clear()
                self.fast_aim_locked_angle = None
                self.fast_aim_line = None
                self.aim_missing_frames = 0
                return False
            return self.fast_aim_direction is not None
        self.aim_marker_missing_frames = 0
        if motion_line is not None:
            self._accept_fast_aim(motion_line, fast_response=True)
            return True
        if (
            self.cursor_motion_frames == 0
            and self.fast_aim_locked_angle is not None
            and self.motion_aim_detector.last_marker_present
        ):
            self.aim_missing_frames = 0
            return True
        line = self.detector.detector.detect_aim_only(
            frame,
            self.latest_scene.board,
            self.latest_scene.obstacles,
            self.latest_scene.origin,
        )
        if line is None:
            self.aim_missing_frames += 1
            if self.aim_missing_frames >= 3:
                self.fast_aim_direction = None
                self.fast_aim_angle_samples.clear()
                self.fast_aim_locked_angle = None
                self.fast_aim_line = None
            return False
        self._accept_fast_aim(line)
        return True

    def _update_current_trajectory(self) -> None:
        if (
            self.latest_scene is None
            or self.fast_aim_direction is None
        ):
            self.current_trajectory = None
            return
        origin = vec(*self.latest_scene.origin)
        if self.fast_origin_x is not None:
            origin = vec(self.fast_origin_x, self.latest_scene.board.bottom)
        delta = self.fast_aim_direction
        if float(np.linalg.norm(delta)) < 0.5 or delta[1] >= -0.01:
            self.current_trajectory = None
            return
        self.current_trajectory = simulate_trajectory(
            origin,
            unit(delta),
            self.latest_scene.board,
            self.latest_scene.obstacles,
            **self._physics_options(),
        )

    def _poll_hotkeys(self) -> None:
        actions = {
            self.config.hotkeys.toggle_overlay.upper(): self.toggle_overlay,
            self.config.hotkeys.pause.upper(): self.toggle_pause,
            self.config.hotkeys.calibrate.upper(): self.start_calibration,
            self.config.hotkeys.settings.upper(): self.open_settings,
        }
        for key in self.hotkeys.pressed():
            action = actions.get(key)
            if action:
                action()

    def refresh_scene(self) -> None:
        """Restart recognition manually without changing automatic round relocking."""
        self.scene_locked = False
        self.latest_scene = None
        self.latest_detection = None
        self.current_trajectory = None
        self.recommendations = []
        self.recommendation_signature = None
        self.recommendation_cache.clear()
        self.recommendation_future_signature = None
        if self.recommendation_future is not None:
            self.recommendation_future.cancel()
            self.recommendation_future = None
        self.stable_scene_signature = None
        self.pending_scene_signature = None
        self.pending_scene_count = 0
        self.locked_block_signature = None
        self.round_candidate_signature = None
        self.round_candidate_count = 0
        self.round_candidate_origins = []
        self.fast_aim_direction = None
        self.fast_aim_angle_samples.clear()
        self.fast_aim_locked_angle = None
        self.fast_aim_line = None
        self.fast_origin_x = None
        self.fast_origin_samples.clear()
        self.aim_missing_frames = 0
        self.aim_marker_missing_frames = 0
        self.shot_missing_frames = 0
        self.aim_was_absent = False
        self.transition_check_active = False
        self.transition_same_count = 0
        self.aim_missing_since = None
        self.aim_reappeared_since = None
        self.detector = TemporalDetector(
            BoardDetector(self.config.vision),
            self.config.vision.temporal_frames,
        )
        self.motion_aim_detector = MotionAimDetector()
        self.tracker.reset()
        self.last_detection_at = 0.0
        self.status = "已重启识别：等待发射线并重新识别外框、方块和发射点"
        self.overlay.update_scene(None, [], None, [], self.status)
        self.overlay.hide()

    def select_target_window(self) -> None:
        """Let the user switch capture targets without restarting the process."""
        timer_was_active = self.timer.isActive()
        self.timer.stop()
        try:
            selected = choose_window_title(self.refresh_button)
            if not selected:
                return
            replacement = WindowCapture(selected)
            previous = self.capture
            self.capture = replacement
            previous.close()
            self.config.window_title = selected
            self.config.save(self.config_path)
            self.last_client_size = None
            self.refresh_scene()
            self.status = f"已选择窗口：{selected}｜等待重新识别"
        except Exception as error:
            QMessageBox.critical(self.refresh_button, "切换窗口失败", str(error))
        finally:
            if timer_was_active:
                self.timer.start()

    def _poll_cursor_motion(self) -> None:
        position = QCursor.pos()
        current = (position.x(), position.y())
        if self.last_cursor_position is not None:
            dx = current[0] - self.last_cursor_position[0]
            dy = current[1] - self.last_cursor_position[1]
            if dx * dx + dy * dy >= 1:
                self.cursor_motion_frames = 4
            elif self.cursor_motion_frames > 0:
                self.cursor_motion_frames -= 1
        self.last_cursor_position = current

    def toggle_overlay(self) -> None:
        if self.overlay.isVisible():
            self.overlay.hide()
            self.config.overlay.visible = False
        else:
            self.overlay.show()
            self.config.overlay.visible = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.status = "识别已暂停（F8继续）" if self.paused else "识别已继续"

    def open_settings(self) -> None:
        was_visible = self.overlay.isVisible()
        self.overlay.hide()
        dialog = SettingsDialog(self.config)
        dialog.debug.setChecked(self.debug_view)
        if dialog.exec():
            dialog.apply()
            self.debug_view = dialog.debug.isChecked()
            self.overlay.debug_view = self.debug_view
            self.recommendation_signature = None
            self.stable_scene_signature = None
            self.pending_scene_signature = None
            self.pending_scene_count = 0
            self.recommendation_cache.clear()
            self.config.save(self.config_path)
        if was_visible:
            self.overlay.show()

    def start_calibration(self) -> None:
        if self.latest_scene is None:
            self.status = "尚未识别到水池，无法开始校准。"
            return
        self.tracker.reset()
        self.calibration_deadline = (
            time.monotonic() + self.config.calibration.duration_seconds
        )
        self.overlay.calibrating = True
        self.status = (
            f"剩余 {self.config.calibration.duration_seconds} 秒｜"
            "请完成 3–5 次左/中/右发射"
        )

    def _calibration_tick(self, frame: np.ndarray) -> None:
        assert self.calibration_deadline is not None
        assert self.latest_scene is not None
        self.tracker.observe(frame, self.latest_scene.board)
        remaining = max(0, math.ceil(self.calibration_deadline - time.monotonic()))
        self.status = f"剩余 {remaining} 秒｜已采集 {len(self.tracker.points)} 个轨迹点"
        if remaining > 0:
            return
        result = fit_calibration(
            self.tracker.points,
            self.latest_scene.board,
            self.config.physics.ball_radius,
        )
        self.calibration_deadline = None
        self.overlay.calibrating = False
        self.config.physics.ball_radius = result.ball_radius
        if result.success:
            if self.latest_scene.obstacles:
                block_sizes = [
                    min(item.rect.width, item.rect.height)
                    for item in self.latest_scene.obstacles
                ]
                median_block_size = float(np.median(block_sizes))
                if median_block_size > 0:
                    self.config.physics.ball_radius_to_block_ratio = (
                        result.ball_radius / median_block_size
                    )
            self.config.calibration.boundary_offsets = result.boundary_offsets
            self.config.physics.reflection_bias_deg = result.reflection_bias_deg
            self.config.calibration.calibrated = True
            self.recommendation_signature = None
        self.config.save(self.config_path)
        self.status = result.message
        QMessageBox.information(
            None,
            "校准完成" if result.success else "校准未完成",
            (
                f"{result.message}\n"
                f"弹珠半径：{result.ball_radius:.2f}px\n"
                f"反射修正：{result.reflection_bias_deg:.2f}°"
            ),
        )

    def tick(self) -> None:
        self._poll_hotkeys()
        self._poll_cursor_motion()
        try:
            frame, geometry = self.capture.grab()
        except RuntimeError as error:
            self.status = str(error)
            self.overlay.update_scene(None, [], None, [], self.status)
            return
        if self.overlay.geometry().getRect() != (
            geometry.left,
            geometry.top,
            geometry.width,
            geometry.height,
        ):
            self.overlay.setGeometry(
                geometry.left, geometry.top, geometry.width, geometry.height
            )
        self.refresh_button.move(geometry.left + 8, geometry.top + 8)
        current_size = (geometry.width, geometry.height)
        if self.last_client_size is not None and current_size != self.last_client_size:
            self.scene_locked = False
            self.latest_scene = None
            self.latest_detection = None
            self.current_trajectory = None
            self.fast_aim_direction = None
            self.fast_aim_angle_samples.clear()
            self.fast_aim_locked_angle = None
            self.fast_aim_line = None
            self.fast_origin_x = None
            self.fast_origin_samples.clear()
            self.motion_aim_detector.reset()
            self.aim_marker_missing_frames = 0
            self.detector.history.clear()
            self.round_candidate_signature = None
            self.round_candidate_count = 0
            self.round_candidate_origins = []
            self.aim_was_absent = False
            self.transition_check_active = False
            self.transition_same_count = 0
            self.aim_missing_since = None
            self.aim_reappeared_since = None
            self.status = "检测到窗口尺寸变化｜正在按新比例重新识别"
        self.last_client_size = current_size
        now = time.monotonic()
        if self.calibration_deadline is not None and self.latest_scene is not None:
            self._calibration_tick(frame)
        elif not self.paused:
            if self.scene_locked:
                aim_found = self._update_fast_aim(frame)
                transition_delay = (
                    self.config.vision.aim_transition_delay_ms / 1000.0
                )
                if aim_found:
                    self.aim_missing_since = None
                    self.shot_missing_frames = 0
                    self._update_current_trajectory()
                    if self.aim_was_absent and not self.transition_check_active:
                        if self.aim_reappeared_since is None:
                            self.aim_reappeared_since = now
                        stable_for = now - self.aim_reappeared_since
                        if stable_for >= transition_delay:
                            self.transition_check_active = True
                            self.transition_same_count = 0
                            self.detector.history.clear()
                            self.last_detection_at = now
                            self.overlay.hide()
                            self.aim_was_absent = False
                            self.aim_reappeared_since = None
                        else:
                            remaining_ms = max(
                                0,
                                round((transition_delay - stable_for) * 1000),
                            )
                            self.status = (
                                "新瞄准线防抖确认中｜"
                                f"{remaining_ms}ms 后比较方块"
                            )
                    elif not self.aim_was_absent:
                        self.aim_reappeared_since = None
                    if (
                        self.transition_check_active
                        and (now - self.last_detection_at) * 1000
                        >= self.config.vision.detection_interval_ms
                    ):
                        outcome = self._check_scene_transition(frame)
                        self.last_detection_at = now
                        if outcome in {"same", "relocked"}:
                            self.transition_check_active = False
                            if self.config.overlay.visible:
                                self.overlay.show()
                else:
                    self.current_trajectory = None
                    self.shot_missing_frames += 1
                    self.aim_reappeared_since = None
                    if self.transition_check_active:
                        self.transition_check_active = False
                        self.transition_same_count = 0
                        if self.config.overlay.visible:
                            self.overlay.show()
                    if self.aim_missing_since is None:
                        self.aim_missing_since = now
                    missing_for = now - self.aim_missing_since
                    if not self.aim_was_absent and missing_for >= transition_delay:
                        self.detector.history.clear()
                        self.round_candidate_signature = None
                        self.round_candidate_count = 0
                        self.round_candidate_origins = []
                        self.transition_same_count = 0
                        self.aim_was_absent = True
                    self.status = (
                        "小球飞行中｜场景保持锁定"
                        if self.aim_was_absent
                        else "瞄准线短暂丢失｜等待防抖确认"
                    )
            elif (
                (now - self.last_detection_at) * 1000
                >= self.config.vision.detection_interval_ms
            ):
                self._detect(frame)
                self.last_detection_at = now
        scene = self.latest_scene
        self.overlay.update_scene(
            self.current_trajectory,
            [],
            scene.board if scene else None,
            list(scene.obstacles) if scene else [],
            self.status,
            scene.ball_radius if scene else None,
        )
        if self.start_calibration_requested and scene is not None:
            self.start_calibration_requested = False
            self.start_calibration()
