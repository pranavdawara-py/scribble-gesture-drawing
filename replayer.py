"""Replay recorded strokes as relative mouse movement. See SPEC_recording.md.

Fix for incomplete shapes / fractional truncation: relative deltas truncate
floating point remainders (int(1.4) = 1, losing 0.4 on every point), which
causes the replayed shape to be much smaller. Replay now uses a fractional
accumulator (rem_x, rem_y) — remainders carry over to the next frame so
no precision is lost across the stroke.

Key insight: mouse_event() posts to Windows input queue — it does not
wait for the target app to process each event. Firing too fast overflows
the queue and the app drops events, causing missing strokes in replay.

Fix: sleep 10ms between each move call. This gives Windows time to
dispatch each event to the target app, and ensures apps running at 60Hz
sample intermediate cursor positions instead of drawing straight lines.
10ms * 500 samples = 5s for a typical stroke — slow enough to not drop events.
"""

from __future__ import annotations

import time

from mouse_control import move_rel, pen_down, pen_up, ensure_pen_down

# Time to sleep between each mouse_event call (seconds).
# 10ms ensures Windows has time to dispatch to the target app, and more
# importantly, ensures target apps (running at 60Hz) have time to sample
# the intermediate curves instead of drawing straight lines from start to end.
INTER_EVENT_SLEEP = 0.010



def replay_stroke_from_buffer(
    points: list[tuple[float, float]],
) -> None:
    """Replay a buffered preview-mode stroke.

    points: list of (dx_px, dy_px) pre-scaled deltas.
    Skips first delta — EMA not warmed up yet, causes large jump.
    10ms sleep between events.
    """
    if len(points) < 2:
        return

    rem_x = 0.0
    rem_y = 0.0

    pen_down()
    for dx_px, dy_px in points[1:]:
        ensure_pen_down()
        total_dx = dx_px + rem_x
        total_dy = dy_px + rem_y
        
        dx = int(total_dx)
        dy = int(total_dy)
        
        rem_x = total_dx - dx
        rem_y = total_dy - dy
        
        if abs(dx) >= 1 or abs(dy) >= 1:
            move_rel(dx, dy)
            time.sleep(INTER_EVENT_SLEEP)
    pen_up()
