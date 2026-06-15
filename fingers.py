"""Finger pose detection from MediaPipe hand landmarks. See SPEC_index.md."""

from __future__ import annotations

from dataclasses import dataclass

# Landmark indices
_WRIST      = 0
_THUMB_TIP  = 4
_THUMB_MCP  = 2
_INDEX_TIP  = 8
_INDEX_PIP  = 6
_MIDDLE_TIP = 12
_MIDDLE_PIP = 10
_RING_TIP   = 16
_RING_PIP   = 14
_PINKY_TIP  = 20
_PINKY_PIP  = 18
_PINKY_MCP  = 17


@dataclass
class HandPose:
    open_palm:         bool   # all 4 fingers extended
    fist:              bool   # all 4 fingers down
    index_only:        bool   # index only            → "index" in SPEC_index.md
    index_middle:      bool   # index + middle         → "V"
    index_middle_ring: bool   # index + middle + ring  → "IMR"
    pinky_only:        bool   # pinky only             → "pinky"


def _extended(lm, tip: int, pip: int, *, margin: float = 0.02) -> bool:
    """True when finger tip is above PIP joint and farther from wrist."""
    tip_y = lm[tip].y < lm[pip].y - margin
    wrist = lm[_WRIST]
    tip_d = (lm[tip].x - wrist.x) ** 2 + (lm[tip].y - wrist.y) ** 2
    pip_d = (lm[pip].x - wrist.x) ** 2 + (lm[pip].y - wrist.y) ** 2
    return tip_y and tip_d > pip_d * 1.05


def _thumb_extended(lm, *, margin: float = 1.3) -> bool:
    """Thumb extends sideways, not vertically, so use distance from the
    pinky-side of the palm instead of the tip/PIP-height heuristic."""
    wrist = lm[_WRIST]
    pinky_mcp = lm[_PINKY_MCP]
    palm_width = ((pinky_mcp.x - wrist.x) ** 2 + (pinky_mcp.y - wrist.y) ** 2) ** 0.5
    tip = lm[_THUMB_TIP]
    mcp = lm[_THUMB_MCP]
    tip_d = ((tip.x - pinky_mcp.x) ** 2 + (tip.y - pinky_mcp.y) ** 2) ** 0.5
    mcp_d = ((mcp.x - pinky_mcp.x) ** 2 + (mcp.y - pinky_mcp.y) ** 2) ** 0.5
    return tip_d > mcp_d * margin and tip_d > palm_width * 0.6


def classify_pose(landmarks) -> HandPose:
    index  = _extended(landmarks, _INDEX_TIP,  _INDEX_PIP)
    middle = _extended(landmarks, _MIDDLE_TIP, _MIDDLE_PIP)
    ring   = _extended(landmarks, _RING_TIP,   _RING_PIP)
    pinky  = _extended(landmarks, _PINKY_TIP,  _PINKY_PIP)
    # thumb is intentionally excluded from open_palm — its sideways extension
    # is detected unreliably; 4 fingers is sufficient and more robust.

    return HandPose(
        open_palm         = index and middle and ring and pinky,
        fist              = not index and not middle and not ring and not pinky,
        index_only        = index and not middle and not ring and not pinky,
        index_middle      = index and middle and not ring and not pinky,
        index_middle_ring = index and middle and ring and not pinky,
        pinky_only        = not index and not middle and not ring and pinky,
    )



def index_tip_xy(landmarks) -> tuple[float, float]:
    return landmarks[_INDEX_TIP].x, landmarks[_INDEX_TIP].y


def pose_name(pose: HandPose | None) -> str:
    """Human-readable pose name used in HUD / websocket broadcasts."""
    if pose is None:
        return "none"
    if pose.open_palm:         return "open_palm"
    if pose.index_middle_ring: return "IMR"
    if pose.index_middle:      return "V"
    if pose.index_only:        return "index"
    if pose.pinky_only:        return "pinky"
    if pose.fist:              return "fist"
    return "other"


def both_palms(writing_pose: HandPose | None, control_pose: HandPose | None) -> bool:
    """True when both hands show open_palm simultaneously (SPEC_index.md)."""
    return bool(
        writing_pose is not None and writing_pose.open_palm
        and control_pose is not None and control_pose.open_palm
    )


# Friendly names for the website hand-pose visualizer / gesture guide
POSE_LABELS = {
    "open_palm": "Open palm",
    "fist":      "Fist",
    "index":     "Pointing",
    "V":         "Peace sign",
    "IMR":       "Three fingers",
    "pinky":     "Pinky",
    "both_palms": "Both palms open",
    "none":      "No hand detected",
    "other":     "Other",
}
