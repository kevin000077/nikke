from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .app import ApplicationController, choose_window_title
from .capture import find_window
from .config import AppConfig


def default_config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / "MarbleAim" / "config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marble-aim",
        description="弹珠游戏实时轨迹识别与透明覆盖层",
    )
    parser.add_argument("--window-title", help="游戏窗口标题中包含的文字")
    parser.add_argument("--calibrate", action="store_true", help="识别成功后启动校准")
    parser.add_argument("--debug-view", action="store_true", help="显示并保存识别框")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径（默认：当前用户 LocalAppData/MarbleAim/config.json）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config or default_config_path()
    # Qt 6 configures PER_MONITOR_AWARE_V2 itself. Calling the legacy Windows
    # DPI API first only makes Qt's second request fail with ACCESS_DENIED.
    app = QApplication(sys.argv[:1])
    app.setApplicationName("弹珠轨迹助手")
    config = AppConfig.load(config_path)
    if args.window_title:
        config.window_title = args.window_title
    if not config.window_title or find_window(config.window_title) is None:
        selected = choose_window_title()
        if not selected:
            return 1
        config.window_title = selected
    config.save(config_path)
    try:
        controller = ApplicationController(
            app,
            config,
            config_path,
            debug_view=args.debug_view,
            start_calibration=args.calibrate,
        )
        controller.start()
    except Exception as error:
        QMessageBox.critical(None, "启动失败", str(error))
        return 2
    app._marble_controller = controller  # type: ignore[attr-defined]
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
