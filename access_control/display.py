from __future__ import annotations

import ctypes
import sys

import cv2
import numpy as np


WINDOW_NAME = "Controle de acesso"
WINDOW_MARGIN = 0.9


class PreviewRenderer:
    """Resize only the GUI image; inference and optional recording keep full resolution."""

    def __init__(self, max_side: int = 1280, screen_size: tuple[int, int] | None = None):
        self.max_side = max_side
        self.screen_size = screen_size or get_screen_size()
        self._frame_size = None
        self._preview_size = None

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if self.max_side <= 0:
            return frame
        if self._frame_size != (width, height):
            self._preview_size = fit_window_size(width, height, self.screen_size,
                                                 max_side=self.max_side)
            self._frame_size = (width, height)
        if self._preview_size == (width, height):
            return frame
        return cv2.resize(frame, self._preview_size, interpolation=cv2.INTER_LINEAR)


def poll_display_key() -> int:
    # Win32 pollKey processes GUI messages without waitKey's minimum sleep.
    poll = getattr(cv2, "pollKey", None)
    return (poll() if poll is not None else cv2.waitKey(1)) & 0xFF


def create_display_window(frame: np.ndarray, window_name: str = WINDOW_NAME) -> None:
    frame_height, frame_width = frame.shape[:2]
    window_width, window_height = fit_window_size(
        frame_width,
        frame_height,
        get_screen_size(),
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, window_width, window_height)


def fit_window_size(
    frame_width: int,
    frame_height: int,
    screen_size: tuple[int, int],
    margin: float = WINDOW_MARGIN,
    max_side: int | None = None,
) -> tuple[int, int]:
    screen_width, screen_height = screen_size
    max_width = max(1, round(screen_width * margin))
    max_height = max(1, round(screen_height * margin))
    scale = min(1.0, max_width / frame_width, max_height / frame_height)
    if max_side is not None and max_side > 0:
        scale = min(scale, max_side / max(frame_width, frame_height))
    return max(1, round(frame_width * scale)), max(1, round(frame_height * scale))


def map_display_point(
    point: tuple[int, int],
    display_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert mouse coordinates from an explicitly resized preview."""
    x, y = point
    display_width, display_height = display_size
    frame_width, frame_height = frame_size
    mapped_x = round(x * frame_width / max(display_width, 1))
    mapped_y = round(y * frame_height / max(display_height, 1))
    return (
        min(max(mapped_x, 0), max(frame_width - 1, 0)),
        min(max(mapped_y, 0), max(frame_height - 1, 0)),
    )


def get_screen_size() -> tuple[int, int]:
    if sys.platform == "win32":
        work_area = _WindowsRect()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(work_area), 0):
            return work_area.right - work_area.left, work_area.bottom - work_area.top
    return 1280, 720


class _WindowsRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]
