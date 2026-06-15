"""
Scribble — gesture-driven drawing controller.

3-mode state machine (IDLE / DRAW / MOVE). See specs/.

Writing hand gestures (hold 0.7s):
  IMR  → DRAW (from IDLE/MOVE)
  V    → MOVE (from IDLE/DRAW)
  palm → IDLE (from DRAW/MOVE)
  index finger → move cursor / draw / preview

In DRAW + preview_mode: pinky/fist commits stroke, both_palms discards.

"""

from __future__ import annotations

import ctypes
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import config as cfg_module
from fingers import classify_pose, index_tip_xy, pose_name, both_palms, HandPose
from gesture_hold import GestureHold
from mouse_control import force_pen_up, move_rel, pen_down, pen_up, is_pen_down
from replayer import replay_stroke_from_buffer
import ws_bridge

CAMERA_INDEX  = 0
MIRROR        = True
HOLD_SECS     = 0.7
WS_HOST       = "127.0.0.1"
WS_PORT       = 8765
STROKE_COLOR  = (0, 255, 80)
STROKE_THICK  = 2
HAND_MARGIN   = 0.18
VIDEO_EXT     = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}


class Mode(Enum):
    IDLE = auto()
    DRAW = auto()
    MOVE = auto()





class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return self._x, self._y  # type: ignore

    def reset(self):
        self._x = self._y = None


def _hand_mask(shape, landmarks):
    if not landmarks:
        return None
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lm in landmarks:
        pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)
        cx, cy = pts.mean(axis=0)
        expanded = pts + (pts - np.array([cx, cy])) * HAND_MARGIN
        cv2.fillConvexPoly(mask, cv2.convexHull(expanded.astype(np.int32)), 255)
    return mask


def _apply_bg(frame, bg_image, landmarks):
    if bg_image is None:
        return frame
    h, w = frame.shape[:2]
    bg = cv2.resize(bg_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask = _hand_mask(frame.shape, landmarks)
    if mask is None:
        return bg
    return np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0, frame, bg)


def _txt(frame, text, pos, color, scale=0.6, thick=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


MODE_CLR = {"IDLE": (180,180,180), "DRAW": (80,220,80), "MOVE": (80,180,255)}


def _hud(frame, mode, preview, wp, cp, stroke):
    h, w = frame.shape[:2]
    c = MODE_CLR.get(mode.name, (255,255,255))
    _txt(frame, f"Mode:{mode.name} W:{pose_name(wp)} C:{pose_name(cp)}", (10,30), c)
    if preview:
        _txt(frame, "Preview ON", (10, h-15), (255,200,80), thick=2)
    if preview and stroke:
        hint = "Pinky/Fist = draw   |   Both palms = cancel"
        (tw,_),_ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = max(10,(w-tw)//2); y = h-40
        ov = frame.copy()
        cv2.rectangle(ov,(x-10,y-20),(x+tw+10,y+10),(0,0,0),-1)
        frame[:] = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)
        _txt(frame, hint, (x,y), (255,255,255), scale=0.55)


def _draw_overlay(frame, stroke_deltas, start_x, start_y):
    """Draw preview stroke on camera.
    
    stroke_deltas: list of (dx_px, dy_px) — sensitivity-scaled screen pixel deltas.
    start_x/y: camera pixel coords of finger at stroke start.
    
    To show on camera: we add the exact deltas that are drawn on the screen
    so the preview shows the true 1:1 physical pixel size of the stroke on the monitor.
    """
    if len(stroke_deltas) < 2:
        return
    cx, cy = start_x, start_y
    pts = [(cx, cy)]
    for dx, dy in stroke_deltas:
        cx = pts[-1][0] + dx
        cy = pts[-1][1] + dy
        pts.append((cx, cy))
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                  STROKE_COLOR, STROKE_THICK, cv2.LINE_AA)


def run() -> None:
    cfg = cfg_module.load()
    ws_bridge.init_config(cfg)
    ws_bridge.start_server(WS_HOST, WS_PORT)

    threading.Thread(target=lambda: (time.sleep(1.5), __import__('webbrowser').open(
        f"http://{WS_HOST}:{WS_PORT}")), daemon=True).start()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.7, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
        
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    
    # 75% weighted towards screen size (halfway between midpoint and full screen)
    scale_w = (cam_w + 3 * screen_w) / 4.0
    scale_h = (cam_h + 3 * screen_h) / 4.0
    
    cv2.namedWindow("Scribble", cv2.WINDOW_NORMAL)

    wh_imr=GestureHold(HOLD_SECS); wh_v=GestureHold(HOLD_SECS); wh_palm=GestureHold(HOLD_SECS)

    def _reset_w(): wh_imr.reset(); wh_v.reset(); wh_palm.reset()

    mode      = Mode.IDLE
    ema       = EMA(cfg["ema_alpha"])

    # last_index: EMA-smoothed normalised finger position from previous frame.
    # Used to compute deltas.
    last_index: tuple[float, float] | None = None

    # Preview stroke buffer: (dx_px, dy_px) sensitivity-scaled screen pixel deltas
    stroke_deltas:   list[tuple[float,float]] = []
    stroke_start_cx: float = 0.0   # camera px where stroke started (for overlay)
    stroke_start_cy: float = 0.0



    # Virtual background
    bg_path_loaded: str | None = None
    bg_image  = None
    bg_cap    = None
    bg_video  = False

    print(f"[Scribble] Running - screen {screen_w}x{screen_h}, cam {cam_w}x{cam_h}, scale {scale_w}x{scale_h}")
    print(f"[Scribble] Website  -> http://{WS_HOST}:{WS_PORT}")

    def _clear_stroke():
        nonlocal stroke_deltas, stroke_start_cx, stroke_start_cy
        stroke_deltas=[]; stroke_start_cx=0.0; stroke_start_cy=0.0

    def _enter_idle():
        nonlocal mode, last_index
        force_pen_up(); ema.reset(); last_index=None; _clear_stroke()
        mode=Mode.IDLE; _reset_w()

    def _enter_draw():
        nonlocal mode, last_index
        mode=Mode.DRAW; ema.reset(); last_index=None; _clear_stroke(); _reset_w()

    def _enter_move():
        nonlocal mode, last_index
        force_pen_up(); mode=Mode.MOVE; ema.reset(); last_index=None
        _clear_stroke(); _reset_w()

    try:
        while True:
            # ── Commands ──────────────────────────────────────────────────────
            for cmd in ws_bridge.pop_commands():
                t = cmd.get("type")
                if   t == "set_writing_hand":  _enter_idle(); print(f"[CMD] hand→{cfg['writing_hand']}")
                elif t == "set_ema":           ema.alpha=cfg["ema_alpha"]
                elif t == "set_sensitivity":   pass
                elif t == "set_ml_mode":
                    if mode==Mode.DRAW and stroke_deltas: _clear_stroke()
                elif t == "set_virtual_background": bg_path_loaded=None
                elif t == "clear_virtual_background":
                    if bg_cap: bg_cap.release(); bg_cap=None
                    bg_video=False; bg_path_loaded=None; bg_image=None
                    print("[CMD] bg cleared")


            # ── Virtual background ────────────────────────────────────────────
            vb = cfg.get("virtual_background")
            if vb != bg_path_loaded:
                if bg_cap: bg_cap.release(); bg_cap=None
                bg_video=False; bg_image=None; bg_path_loaded=vb
                if vb and Path(vb).exists():
                    if Path(vb).suffix.lower() in VIDEO_EXT:
                        vc=cv2.VideoCapture(vb)
                        if vc.isOpened(): bg_cap=vc; bg_video=True
                        else: vc.release()
                    else:
                        img=cv2.imread(vb)
                        if img is not None: bg_image=img; print(f"[BG] loaded:{vb}")
                        else: print(f"[BG] failed:{vb}")
            if bg_video and bg_cap:
                ok,f=bg_cap.read()
                if not ok: bg_cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,f=bg_cap.read()
                bg_image=f if ok else None

            # ── Camera ────────────────────────────────────────────────────────
            ok, frame = cap.read()
            if not ok: continue
            if MIRROR: frame=cv2.flip(frame,1)
            results=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            wp:HandPose|None=None; cp:HandPose|None=None; wx=wy=0.0
            if results.multi_hand_landmarks:
                for lm,hand in zip(results.multi_hand_landmarks,results.multi_handedness):
                    pose=classify_pose(lm.landmark)
                    if hand.classification[0].label==cfg["writing_hand"]:
                        wp=pose; wx,wy=index_tip_xy(lm.landmark)
                    else: cp=pose

            frame=_apply_bg(frame, bg_image, results.multi_hand_landmarks)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)

            preview=cfg["ml_mode"]; sens=cfg["sensitivity"]
            cam_h,cam_w=frame.shape[:2]

            # ── Mode transitions ──────────────────────────────────────────────
            if wh_palm.update(bool(wp and wp.open_palm)) and mode in (Mode.DRAW,Mode.MOVE):
                _enter_idle(); print("[W]→IDLE")
            elif wh_imr.update(bool(wp and wp.index_middle_ring)) and mode in (Mode.IDLE,Mode.MOVE):
                _enter_draw(); print("[W]→DRAW")
            elif wh_v.update(bool(wp and wp.index_middle)) and mode in (Mode.IDLE,Mode.DRAW):
                _enter_move(); print("[W]→MOVE")

            # ── Delta computation — ONE place, used by cursor and preview
            # dx_px/dy_px = sensitivity-scaled screen pixel delta.
            # Relative to last_index (previous frame's EMA position).
            # Used for: cursor movement, preview buffer.
            dx_px=dy_px=0.0

            if wp and (wp.index_only or wp.index_middle) and mode in (Mode.DRAW, Mode.MOVE):
                sx,sy = ema.update(wx,wy)
                if last_index is not None:
                    dx_px = (sx - last_index[0]) * sens * scale_w
                    dy_px = (sy - last_index[1]) * sens * scale_h

                    # ── Cursor / preview buffer ───────────────────────────────
                    if mode==Mode.MOVE:
                        if is_pen_down(): pen_up()
                        if abs(dx_px)>=1 or abs(dy_px)>=1:
                            move_rel(int(dx_px),int(dy_px))

                    elif mode==Mode.DRAW and not preview:
                        if not is_pen_down(): pen_down()
                        if abs(dx_px)>=1 or abs(dy_px)>=1:
                            move_rel(int(dx_px),int(dy_px))

                    elif mode==Mode.DRAW and preview:
                        if abs(dx_px)>=1 or abs(dy_px)>=1:
                            if not stroke_deltas:
                                # Anchor overlay at finger position from PREVIOUS frame
                                # (last_index) — the true start of the stroke
                                stroke_start_cx = last_index[0] * cam_w
                                stroke_start_cy = last_index[1] * cam_h
                            stroke_deltas.append((dx_px, dy_px))

                last_index=(sx,sy)

            # When finger leaves index pose, don't reset last_index.
            # Keep the last known position so next time index appears
            # we compute a correct delta from where we left off.
            # (last_index only resets on explicit mode transitions)

            # ── Pen state enforcement ─────────────────────────────────────────
            if mode==Mode.DRAW and not preview:
                if not is_pen_down(): pen_down()
            else:
                if is_pen_down(): pen_up()

            # ── Preview gestures ──────────────────────────────────────────────
            if mode==Mode.DRAW and preview and stroke_deltas:
                if wp and wp.pinky_only:
                    replay_stroke_from_buffer(stroke_deltas)
                    print("[Preview] Pinky→drew"); _clear_stroke()
                elif wp and wp.fist:
                    replay_stroke_from_buffer(stroke_deltas)
                    print("[Preview] Fist→drew"); _clear_stroke()
                elif both_palms(wp,cp):
                    print("[Preview] Cancelled"); _clear_stroke()



            # ── Broadcast ─────────────────────────────────────────────────────
            ws_bridge.update_state(
                mode=mode.name,
                writing_pose=pose_name(wp), control_pose=pose_name(cp),
                stroke_active=bool(stroke_deltas),
            )
            ws_bridge.broadcast_state()

            # ── Render ────────────────────────────────────────────────────────
            if mode==Mode.DRAW and preview and stroke_deltas:
                _draw_overlay(frame, stroke_deltas, stroke_start_cx, stroke_start_cy)
            _hud(frame, mode, preview, wp, cp, bool(stroke_deltas))
            cv2.imshow("Scribble", frame)
            cv2.setWindowProperty("Scribble", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    finally:
        force_pen_up(); cap.release()
        if bg_cap: bg_cap.release()
        cv2.destroyAllWindows(); print("[Scribble] Exited.")


if __name__=="__main__":
    run()
