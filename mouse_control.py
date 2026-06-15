"""Reliable left-button mouse control (Windows)."""

from __future__ import annotations
import ctypes

_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004

_user32 = ctypes.windll.user32
_pen_down = False


def is_pen_down() -> bool:
    return _pen_down


def pen_down() -> None:
    global _pen_down
    if _pen_down:
        return
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    _pen_down = True


def pen_up() -> None:
    global _pen_down
    if not _pen_down:
        return
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    _pen_down = False


def move_rel(dx: int, dy: int) -> None:
    if dx == 0 and dy == 0:
        return
    _user32.mouse_event(_MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)





def ensure_pen_down() -> None:
    """Re-assert left button if something released it externally."""
    if not _pen_down:
        pen_down()


def force_pen_up() -> None:
    pen_up()
