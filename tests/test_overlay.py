from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QPushButton

from marble_aim.overlay import ManualFrameSelector, RefreshButton


def test_control_panel_exposes_restart_and_exit_buttons():
    app = QApplication.instance() or QApplication([])
    panel = RefreshButton()
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    frame_spy = QSignalSpy(panel.manual_frame_requested)
    select_spy = QSignalSpy(panel.select_window_requested)
    exit_spy = QSignalSpy(panel.exit_requested)

    assert "重启识别" in buttons
    assert "设置白框" in buttons
    assert "选择窗口" in buttons
    assert "退出助手" in buttons

    buttons["设置白框"].click()
    buttons["选择窗口"].click()
    buttons["退出助手"].click()

    assert frame_spy.count() == 1
    assert select_spy.count() == 1
    assert exit_spy.count() == 1
    panel.close()
    app.processEvents()


def test_manual_frame_selector_collects_four_lines():
    app = QApplication.instance() or QApplication([])
    selector = ManualFrameSelector()
    selector.resize(400, 300)
    selected_spy = QSignalSpy(selector.frame_selected)
    selector.begin()
    app.processEvents()

    for point in (
        QPoint(50, 120),
        QPoint(350, 120),
        QPoint(200, 40),
        QPoint(200, 260),
    ):
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=point)

    assert selected_spy.count() == 1
    values = selected_spy.at(0)
    assert abs(values[0] - 0.125) < 0.01
    assert abs(values[1] - (40 / 300)) < 0.01
    assert abs(values[2] - 0.875) < 0.01
    assert abs(values[3] - (260 / 300)) < 0.01
    selector.close()
    app.processEvents()
