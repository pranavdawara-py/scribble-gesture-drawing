"""Hold-duration gesture detection. See SPEC_gestures.md (Gesture Hold Reset Rules)."""

from __future__ import annotations
import time


class GestureHold:
    """Fires once when a pose has been held continuously for hold_seconds.

    - Resets to 0 immediately if the matched pose changes mid-hold (no
      partial credit across poses).
    - After firing, resets so it must release-and-rehold before it can
      fire again (no repeat-fire while the pose is still held).
    """

    def __init__(self, hold_seconds: float = 0.7):
        self.hold_seconds = hold_seconds
        self._active = False
        self._start: float | None = None
        self._fired = False

    def update(self, matched: bool) -> bool:
        now = time.time()
        if not matched:
            self._active = False
            self._start = None
            self._fired = False
            return False

        if not self._active:
            self._active = True
            self._start = now
            return False

        if self._fired or self._start is None:
            return False

        if now - self._start >= self.hold_seconds:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        self._active = False
        self._start = None
        self._fired = False

    @property
    def holding(self) -> bool:
        return self._active and not self._fired

    def elapsed(self) -> float:
        """Seconds since hold started; 0 if not currently holding."""
        if self._active and self._start is not None:
            return time.time() - self._start
        return 0.0
