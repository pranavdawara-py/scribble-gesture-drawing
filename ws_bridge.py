"""
WebSocket bridge between the gesture engine and the website UI.
See SPEC_websocket.md for the full message contract.

The main loop calls `broadcast_state()` every frame.
The website sends JSON commands which are queued for main loop consumption.
Recognized commands are applied here (config mutations, recording
save/discard); side effects requiring state-machine access (resets,
buffer clears, replay selection, virtual background frame application)
are read back by main.py via `pop_commands()`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import config as cfg_module

# Imported at module level so FastAPI route registration resolves UploadFile
# correctly. Importing inside _run_http_server causes a 422 on every upload.
try:
    from fastapi import File, UploadFile
except ImportError:
    pass

# Imported at module level so FastAPI's route registration can resolve
# UploadFile correctly. Importing inside _run_http_server causes FastAPI
# to treat the parameter as a query param, returning 422 on every upload.
try:
    from fastapi import File, UploadFile
except ImportError:
    pass

BACKGROUNDS_DIR = Path(__file__).resolve().parent / "backgrounds"

# --- Shared state (written by main loop, read by WS handler) ---
_state: dict[str, Any] = {
    "mode": "IDLE",
    "writing_pose": "none",
    "control_pose": "none",
    "stroke_active": False,

    "writing_hand": "Right",
    "sensitivity": 0.35,
    "ema_alpha": 0.40,
    "ml_mode": False,
    "virtual_background": None,
}

# Commands sent from website -> main loop (consumed by main.py)
_command_queue: list[dict] = []
_queue_lock = threading.Lock()

# Connected websocket clients
_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None

# Shared config dict (owned by main.py, mutated here on set_* commands)
_cfg: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Public API used by main.py
# ---------------------------------------------------------------------------

def init_config(cfg: dict[str, Any]) -> None:
    """Register the live config dict; ws_bridge mutates it on set_* commands."""
    global _cfg
    _cfg = cfg
    _state.update(cfg)


def update_state(**kwargs: Any) -> None:
    """Called by main loop to update shared state."""
    _state.update(kwargs)


def pop_commands() -> list[dict]:
    """Drain and return all queued commands from the website for main.py
    to apply (state-machine side effects)."""
    with _queue_lock:
        cmds = list(_command_queue)
        _command_queue.clear()
    return cmds


def broadcast_state() -> None:
    """Push current state to all connected websocket clients (non-blocking)."""
    if _cfg is not None:
        _state.update(_cfg)
    if _loop is None or not _clients:
        return
    msg = json.dumps(_state)
    asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: str) -> None:
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(msg)   # websockets library: send(), not send_text()
        except Exception as exc:
            print(f"[WS] Send error ({type(exc).__name__}): {exc}")
            dead.add(ws)
    _clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------

# Commands whose ONLY effect is a config mutation (handled fully here).
_SIMPLE_CONFIG_COMMANDS = {
    "set_writing_hand": "writing_hand",
    "set_sensitivity": "sensitivity",
    "set_ema": "ema_alpha",
    "set_ml_mode": "ml_mode",
}


def _handle_command(cmd: dict) -> None:
    """Apply config-mutating commands immediately; queue everything else
    (and queue config commands too, so main.py can apply state-machine
    side effects like resets / buffer clears)."""
    ctype = cmd.get("type")

    if ctype in _SIMPLE_CONFIG_COMMANDS:
        key = _SIMPLE_CONFIG_COMMANDS[ctype]
        value = cmd.get("value")
        if _cfg is not None:
            if not cfg_module.update(_cfg, key, value):
                print(f"[WS] Ignored '{ctype}': invalid value {value!r}")
                return
        with _queue_lock:
            _command_queue.append(cmd)
        return

    if ctype == "set_virtual_background":
        value = cmd.get("value")
        if not isinstance(value, str) or not value:
            print(f"[WS] Ignored 'set_virtual_background': invalid value {value!r}")
            return
        if _cfg is not None:
            cfg_module.update(_cfg, "virtual_background", value)
        with _queue_lock:
            _command_queue.append(cmd)
        return

    if ctype == "clear_virtual_background":
        if _cfg is not None:
            cfg_module.update(_cfg, "virtual_background", None)
        with _queue_lock:
            _command_queue.append(cmd)
        return


    print(f"[WS] Ignored unknown command type: {ctype!r}")


# ---------------------------------------------------------------------------
# WebSocket server — runs via the `websockets` library (NOT uvicorn)
# uvicorn's wsproto/websockets backend has version-incompatibility issues
# that cause 403 on every upgrade. The websockets library serves WS directly
# on a dedicated port (HTTP_PORT + 1) without any uvicorn involvement.
# ---------------------------------------------------------------------------

async def _handle_ws_client(websocket) -> None:
    """Handle a single WebSocket client connection."""
    _clients.add(websocket)
    print(f"[WS] Browser connected ({len(_clients)} client(s))")
    try:
        async for raw in websocket:
            if not isinstance(raw, str):
                continue  # skip binary frames
            try:
                cmd = json.loads(raw)
                if not isinstance(cmd, dict) or "type" not in cmd:
                    print(f"[WS] Ignored malformed command: {raw!r}")
                    continue
                _handle_command(cmd)
            except json.JSONDecodeError:
                print(f"[WS] Ignored malformed JSON: {raw!r}")
    except Exception as exc:
        print(f"[WS] Client error: {type(exc).__name__}: {exc}")
    finally:
        _clients.discard(websocket)
        print(f"[WS] Browser disconnected ({len(_clients)} client(s) remaining)")


async def _ws_serve_forever(host: str, ws_port: int) -> None:
    """Start the websockets server and run indefinitely.
    Tries the new asyncio API (websockets >= 14.0) first, falls back to
    the legacy API for older installs.
    """
    # ── New API (websockets >= 14.0) ───────────────────────────────────
    try:
        from websockets.asyncio.server import serve
        async with serve(_handle_ws_client, host, ws_port):
            print(f"[WS] WebSocket server ready -> ws://{host}:{ws_port}")
            await asyncio.Future()   # run forever
        return
    except ImportError:
        pass  # fall through to legacy API
    except Exception as exc:
        print(f"[WS] New-API serve failed ({exc}), trying legacy API…")

    # ── Legacy API (websockets < 14.0) ─────────────────────────────────
    try:
        import websockets

        async def _legacy(ws, path):
            await _handle_ws_client(ws)

        async with websockets.serve(_legacy, host, ws_port):
            print(f"[WS] WebSocket server ready -> ws://{host}:{ws_port}")
            await asyncio.Future()
    except Exception as exc:
        print(f"[WS] Legacy-API serve also failed: {exc}")
        raise


def _run_ws_server(host: str, ws_port: int) -> None:
    """Thread target: own event loop + websockets server."""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    try:
        _loop.run_until_complete(_ws_serve_forever(host, ws_port))
    except Exception:
        import traceback
        print("[WS] WebSocket server crashed:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# HTTP server — FastAPI / uvicorn (HTTP only, no WebSocket endpoint)
# ---------------------------------------------------------------------------

def _run_http_server(host: str, http_port: int) -> None:
    """Thread target: FastAPI + uvicorn serving the website and REST endpoints."""
    try:
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse, JSONResponse
        import uvicorn
    except ImportError as e:
        print(f"[WS] HTTP server deps missing ({e}). Run: pip install fastapi 'uvicorn[standard]'")
        return

    app = FastAPI()

    website_dir = Path(__file__).resolve().parent / "website"
    static_dir = website_dir / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(website_dir / "index.html"))

    @app.post("/upload_background")
    async def upload_background(file: UploadFile = File(...)):
        BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "").suffix.lower()
        ALLOWED_IMAGE = {".jpg", ".jpeg", ".png"}
        ALLOWED_VIDEO = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
        if suffix not in ALLOWED_IMAGE | ALLOWED_VIDEO:
            suffix = ".png"
        dest_name = f"{uuid.uuid4().hex}{suffix}"
        dest_path = BACKGROUNDS_DIR / dest_name
        contents = await file.read()
        with dest_path.open("wb") as out:
            out.write(contents)
        return JSONResponse({"path": str(dest_path)})

    @app.get("/background_preview")
    async def background_preview(path: str):
        p = Path(path)
        try:
            p_resolved = p.resolve()
            p_resolved.relative_to(BACKGROUNDS_DIR.resolve())
        except (ValueError, OSError):
            return JSONResponse({"error": "invalid path"}, status_code=404)
        if not p_resolved.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(p_resolved))

    http_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(http_loop)
    server_config = uvicorn.Config(app, host=host, port=http_port, log_level="warning")
    server = uvicorn.Server(server_config)
    try:
        http_loop.run_until_complete(server.serve())
    except Exception:
        import traceback
        print("[WS] HTTP server crashed:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the HTTP server (port) and WebSocket server (port+1) as daemon threads."""
    ws_port = port + 1  # WebSocket always on HTTP_PORT + 1

    threading.Thread(
        target=_run_http_server, args=(host, port), daemon=True, name="http-server"
    ).start()

    threading.Thread(
        target=_run_ws_server, args=(host, ws_port), daemon=True, name="ws-server"
    ).start()

    print(f"[WS] Starting: http://{host}:{port} (website)  |  ws://{host}:{ws_port} (WebSocket)")