from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
from typing import Iterable

import mss
import numpy as np
from numpy.typing import NDArray
import win32con
import win32gui

Image = NDArray[np.uint8]


def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def list_visible_windows() -> list[tuple[int, str]]:
    windows: list[tuple[int, str]] = []

    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if title and win32gui.GetClientRect(hwnd)[2:] != (0, 0):
            windows.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)
    return sorted(windows, key=lambda item: item[1].lower())


def find_window(title_fragment: str) -> tuple[int, str] | None:
    fragment = title_fragment.casefold()
    matches = [
        item for item in list_visible_windows() if fragment in item[1].casefold()
    ]
    if not matches:
        return None
    exact = next((item for item in matches if item[1].casefold() == fragment), None)
    return exact or matches[0]


def client_geometry(hwnd: int, title: str = "") -> WindowGeometry:
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
    width = client_right - client_left
    height = client_bottom - client_top
    if width <= 0 or height <= 0:
        raise RuntimeError("目标窗口已最小化或客户区大小为零")
    return WindowGeometry(
        hwnd,
        title or win32gui.GetWindowText(hwnd),
        left,
        top,
        width,
        height,
    )


class WindowCapture:
    def __init__(self, title_fragment: str):
        match = find_window(title_fragment)
        if match is None:
            raise RuntimeError(f"找不到标题包含“{title_fragment}”的可见窗口")
        self.hwnd, self.title = match
        self._mss = mss.mss()

    def geometry(self) -> WindowGeometry:
        if not win32gui.IsWindow(self.hwnd):
            raise RuntimeError("目标窗口已经关闭")
        return client_geometry(self.hwnd, self.title)

    def grab(self) -> tuple[Image, WindowGeometry]:
        geometry = self.geometry()
        bgra = np.asarray(self._mss.grab(geometry.monitor), dtype=np.uint8)
        return np.ascontiguousarray(bgra[:, :, :3]), geometry

    def cursor_client_position(self) -> tuple[float, float]:
        x, y = win32gui.GetCursorPos()
        client_x, client_y = win32gui.ScreenToClient(self.hwnd, (x, y))
        return float(client_x), float(client_y)

    def close(self) -> None:
        self._mss.close()


class HotkeyPoller:
    """Global edge-triggered function-key polling without keyboard injection."""

    KEY_CODES = {
        "F7": win32con.VK_F7,
        "F8": win32con.VK_F8,
        "F9": win32con.VK_F9,
        "F10": win32con.VK_F10,
        "F11": win32con.VK_F11,
        "F12": win32con.VK_F12,
    }

    def __init__(self, keys: Iterable[str]):
        self.keys = {key.upper(): self.KEY_CODES[key.upper()] for key in keys}
        self._down = {key: False for key in self.keys}

    def pressed(self) -> list[str]:
        result: list[str] = []
        for name, code in self.keys.items():
            down = bool(ctypes.windll.user32.GetAsyncKeyState(code) & 0x8000)
            if down and not self._down[name]:
                result.append(name)
            self._down[name] = down
        return result
