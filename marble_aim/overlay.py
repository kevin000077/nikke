from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .config import AppConfig
from .geometry import Obstacle, Rect, Trajectory


class RefreshButton(QWidget):
    refresh_requested = Signal()
    exit_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        refresh_button = QPushButton("重启识别")
        refresh_button.setFixedSize(92, 34)
        refresh_button.setStyleSheet(
            "QPushButton {"
            "background:#163A50; color:white; border:2px solid #00E7FF;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#235A77; }"
            "QPushButton:pressed { background:#0B2635; }"
        )
        refresh_button.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(refresh_button)

        exit_button = QPushButton("退出助手")
        exit_button.setFixedSize(82, 34)
        exit_button.setStyleSheet(
            "QPushButton {"
            "background:#4A2028; color:white; border:2px solid #FF6075;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#70313D; }"
            "QPushButton:pressed { background:#32151B; }"
        )
        exit_button.clicked.connect(self.exit_requested.emit)
        layout.addWidget(exit_button)
        self.adjustSize()


class OverlayWindow(QWidget):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.current: Trajectory | None = None
        self.recommendations: list[Trajectory] = []
        self.board: Rect | None = None
        self.obstacles: list[Obstacle] = []
        self.ball_radius = config.physics.ball_radius
        self.status = "正在等待游戏画面…"
        self.debug_view = False
        self.calibrating = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def update_scene(
        self,
        current: Trajectory | None,
        recommendations: list[Trajectory],
        board: Rect | None,
        obstacles: list[Obstacle],
        status: str,
        ball_radius: float | None = None,
    ) -> None:
        self.current = current
        self.recommendations = recommendations
        self.board = board
        self.obstacles = obstacles
        self.status = status
        if ball_radius is not None:
            self.ball_radius = ball_radius
        self.update()

    def _draw_trajectory(
        self, painter: QPainter, trajectory: Trajectory, color: QColor, width: float
    ) -> None:
        if len(trajectory.points) < 2:
            return
        path = QPainterPath(QPointF(*trajectory.points[0]))
        for point in trajectory.points[1:]:
            path.lineTo(QPointF(*point))
        visual_scale = (
            max(0.70, min(2.75, self.board.width / 420.0))
            if self.board is not None
            else 1.0
        )
        painter.setPen(
            QPen(color, width * visual_scale, Qt.PenStyle.SolidLine)
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        if self.config.overlay.show_collision_points:
            radius = self.ball_radius
            ball_fill = QColor(255, 255, 255, 150)
            painter.setBrush(ball_fill)
            painter.setPen(QPen(color, 1.8 * visual_scale))
            for collision in trajectory.collisions:
                painter.drawEllipse(QPointF(*collision.point), radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self.config.overlay.opacity)
        if self.calibrating:
            painter.setPen(QColor("#FFFFFF"))
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(15, 15, min(560, self.width() - 30), 58, 10, 10)
            painter.drawText(
                30,
                38,
                "校准录制中：请分别向左、中、右发射，尽量制造墙壁反弹。",
            )
            painter.drawText(30, 60, self.status)
            return
        if self.board and self.config.overlay.show_collision_frame:
            board = self.board
            visual_scale = max(0.70, min(2.75, board.width / 420.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    QColor("#00E7FF"),
                    1.0 * visual_scale,
                    Qt.PenStyle.SolidLine,
                )
            )
            painter.drawLine(
                QPointF(board.left, board.bottom),
                QPointF(board.left, board.top),
            )
            painter.drawLine(
                QPointF(board.left, board.top),
                QPointF(board.right, board.top),
            )
            painter.drawLine(
                QPointF(board.right, board.top),
                QPointF(board.right, board.bottom),
            )
            painter.setPen(
                QPen(
                    QColor("#FFB13B"),
                    1.8 * visual_scale,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.drawLine(
                QPointF(board.left, board.bottom),
                QPointF(board.right, board.bottom),
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(
                QPointF(board.left + 6, board.bottom - 6),
                "OPEN BOTTOM",
            )
        if self.current:
            self._draw_trajectory(
                painter,
                self.current,
                QColor(self.config.overlay.current_color),
                self.config.overlay.line_width,
            )
        for index, trajectory in enumerate(self.recommendations):
            color = QColor(
                self.config.overlay.recommendation_colors[
                    index % len(self.config.overlay.recommendation_colors)
                ]
            )
            self._draw_trajectory(
                painter, trajectory, color, self.config.overlay.line_width + 0.5
            )
            if len(trajectory.points) > 1:
                point = trajectory.points[1]
                painter.setPen(color)
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.drawText(
                    QPointF(float(point[0] + 6), float(point[1] - 6)),
                    (
                        f"#{index + 1} 覆盖{trajectory.unique_block_hits}块 "
                        f"有效伤害{trajectory.effective_damage} "
                        f"稳定{trajectory.stable_balls_before_change}球 "
                        f"{trajectory.angle_deg:.1f}°"
                    ),
                )
        if self.config.overlay.show_locked_boxes:
            visual_scale = (
                max(0.70, min(2.75, self.board.width / 420.0))
                if self.board is not None
                else 1.0
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    QColor("#54FF9A"),
                    2.0 * visual_scale,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            for index, obstacle in enumerate(self.obstacles, start=1):
                rect = obstacle.rect
                painter.drawRect(
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                )
                painter.drawText(
                    QPointF(rect.left + 4, rect.top + 14),
                    f"LOCK {index}",
                )
        if self.debug_view:
            painter.setPen(QPen(QColor("#00FFFF"), 1, Qt.PenStyle.DashLine))
            if self.board:
                painter.drawRect(
                    self.board.left,
                    self.board.top,
                    self.board.width,
                    self.board.height,
                )
            painter.setPen(QPen(QColor("#FF4060"), 1))
            for obstacle in self.obstacles:
                rect = obstacle.rect
                painter.drawRect(
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                )
                painter.drawText(
                    QPointF(rect.left + 3, rect.top + 13),
                    f"BLOCK {obstacle.identifier}",
                )
        painter.setPen(QColor("#FFFFFF"))
        painter.setBrush(QColor(0, 0, 0, 145))
        painter.drawRoundedRect(10, max(10, self.height() - 34), min(620, self.width() - 20), 25, 7, 7)
        painter.drawText(20, self.height() - 16, self.status)


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("弹珠轨迹助手设置")
        layout = QFormLayout(self)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1.0, 30.0)
        self.radius.setDecimals(2)
        self.radius.setValue(config.physics.ball_radius)
        self.max_collisions = QSpinBox()
        self.max_collisions.setRange(1, 100)
        self.max_collisions.setValue(config.physics.max_collisions)
        self.reflection_bias = QDoubleSpinBox()
        self.reflection_bias.setRange(-10.0, 10.0)
        self.reflection_bias.setDecimals(2)
        self.reflection_bias.setValue(config.physics.reflection_bias_deg)
        self.origin_x = QDoubleSpinBox()
        self.origin_x.setRange(0.0, 1.0)
        self.origin_x.setSingleStep(0.01)
        self.origin_x.setValue(config.calibration.launch_origin_normalized[0])
        self.origin_y = QDoubleSpinBox()
        self.origin_y.setRange(0.0, 1.0)
        self.origin_y.setSingleStep(0.01)
        self.origin_y.setValue(config.calibration.launch_origin_normalized[1])
        self.boundary_offsets: list[QDoubleSpinBox] = []
        for value in config.calibration.boundary_offsets:
            field = QDoubleSpinBox()
            field.setRange(-30.0, 30.0)
            field.setDecimals(1)
            field.setValue(value)
            self.boundary_offsets.append(field)
        self.water_h_low = QSpinBox()
        self.water_h_low.setRange(0, 179)
        self.water_h_low.setValue(config.vision.water_hsv_low[0])
        self.water_h_high = QSpinBox()
        self.water_h_high.setRange(0, 179)
        self.water_h_high.setValue(config.vision.water_hsv_high[0])
        self.block_h_low = QSpinBox()
        self.block_h_low.setRange(0, 179)
        self.block_h_low.setValue(config.vision.block_hsv_low[0])
        self.block_h_high = QSpinBox()
        self.block_h_high.setRange(0, 179)
        self.block_h_high.setValue(config.vision.block_hsv_high[0])
        self.debug = QCheckBox()
        layout.addRow("无方块时后备半径（像素）", self.radius)
        layout.addRow("最大碰撞次数", self.max_collisions)
        layout.addRow("反射角修正（度）", self.reflection_bias)
        layout.addRow("发射点 X（0–1）", self.origin_x)
        layout.addRow("发射点 Y（0–1）", self.origin_y)
        layout.addRow("左边界偏移", self.boundary_offsets[0])
        layout.addRow("右边界偏移", self.boundary_offsets[1])
        layout.addRow("上边界偏移", self.boundary_offsets[2])
        layout.addRow("下边界偏移", self.boundary_offsets[3])
        layout.addRow("水池色相下限", self.water_h_low)
        layout.addRow("水池色相上限", self.water_h_high)
        layout.addRow("方块色相下限", self.block_h_low)
        layout.addRow("方块色相上限", self.block_h_high)
        layout.addRow("显示识别框", self.debug)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def apply(self) -> None:
        self.config.physics.ball_radius = self.radius.value()
        self.config.physics.max_collisions = self.max_collisions.value()
        self.config.physics.reflection_bias_deg = self.reflection_bias.value()
        self.config.calibration.launch_origin_normalized = [
            self.origin_x.value(),
            self.origin_y.value(),
        ]
        self.config.calibration.boundary_offsets = [
            field.value() for field in self.boundary_offsets
        ]
        self.config.vision.water_hsv_low[0] = self.water_h_low.value()
        self.config.vision.water_hsv_high[0] = self.water_h_high.value()
        self.config.vision.block_hsv_low[0] = self.block_h_low.value()
        self.config.vision.block_hsv_high[0] = self.block_h_high.value()
