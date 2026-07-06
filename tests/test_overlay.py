from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QPushButton

from marble_aim.overlay import RefreshButton


def test_control_panel_exposes_restart_and_exit_buttons():
    app = QApplication.instance() or QApplication([])
    panel = RefreshButton()
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    select_spy = QSignalSpy(panel.select_window_requested)
    exit_spy = QSignalSpy(panel.exit_requested)

    assert "重启识别" in buttons
    assert "选择窗口" in buttons
    assert "退出助手" in buttons

    buttons["选择窗口"].click()
    buttons["退出助手"].click()

    assert select_spy.count() == 1
    assert exit_spy.count() == 1
    panel.close()
    app.processEvents()
