# Scribble

A gesture-controlled drawing assistant. Webcam + MediaPipe hand tracking lets your hand control the mouse to draw on your screen (e.g., in MS Paint, skribbl.io). A local website (FastAPI + WebSocket) gives you live status, settings, a gesture guide, and a virtual background option.

> [!NOTE]
> **Recording & Replaying Features Excluded**
> The logic for recording and replaying drawing strokes has been intentionally excluded from this open-source repository. This repository demonstrates the core spatial tracking and "Preview Mode" drawing capabilities!

**Windows only** (uses `ctypes.windll.user32` for mouse control and screen dimensions).

## Setup

```bash
pip install -r requirements.txt
python main.py
```

A camera window opens (always on top) and your browser opens to `http://127.0.0.1:8765` after about 1.5 seconds — keep that tab open while the app is running.

Press `q` in the camera window to quit.

## Drawing Modes

- **Idle** — watching only, mouse untouched.
- **Draw** — index finger draws; hold three fingers (index+middle+ring) to enter, hold an open palm to leave.
- **Move** — index finger repositions the cursor without drawing; hold a peace sign to enter.

### Preview Mode
With **"Show drawing on camera first"** enabled in the web UI, your stroke is previewed locally on the camera feed instead of moving the real cursor immediately. 
- Show **pinky** or **fist** to commit the stroke and send the input to the OS.
- Show **both palms** to cancel the stroke at any time.

## Known Limitations

- Windows only.
- Single camera, single drawing-hand per session.
- Virtual background uses a convex-hull hand mask — expect a rough edge around your hand, not a precise green-screen.
