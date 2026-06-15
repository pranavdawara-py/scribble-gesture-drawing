"""
Config persistence — SPEC_config.md

Loads/saves config.json next to this file. Self-healing: any missing or
invalid key falls back to its default (see SPEC_index.md) and the
corrected file is written back immediately. Writes are atomic (temp file
+ rename).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULTS: dict[str, Any] = {
    "writing_hand": "Right",
    "sensitivity": 1.0,
    "ema_alpha": 0.40,
    "ml_mode": False,
    "virtual_background": None,
}


def _valid(key: str, value: Any) -> bool:
    if key == "writing_hand":
        return value in ("Left", "Right")
    if key == "sensitivity":
        return isinstance(value, (int, float)) 
    if key == "ema_alpha":
        return isinstance(value, (int, float)) 
    if key == "ml_mode":
        return isinstance(value, bool)
    if key == "virtual_background":
        return value is None or isinstance(value, str)
    return False


def load() -> dict[str, Any]:
    """Load config.json, self-healing any missing/invalid keys."""
    data: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, OSError):
            data = {}

    cfg: dict[str, Any] = {}
    changed = False
    for key, default in DEFAULTS.items():
        value = data.get(key, default)
        if not _valid(key, value):
            value = default
            changed = True
        if key not in data:
            changed = True
        cfg[key] = value

    # virtual_background: reset if the referenced file no longer exists
    bg = cfg["virtual_background"]
    if bg is not None and not Path(bg).exists():
        cfg["virtual_background"] = None
        changed = True

    if changed or not CONFIG_PATH.exists():
        save(cfg)

    return cfg


def save(cfg: dict[str, Any]) -> None:
    """Atomically write cfg to config.json (write temp file, then rename)."""
    tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp_path, CONFIG_PATH)


def update(cfg: dict[str, Any], key: str, value: Any) -> bool:
    """Update a single key in cfg (in place) and persist. Returns True if applied."""
    if key not in DEFAULTS or not _valid(key, value):
        return False
    cfg[key] = value
    save(cfg)
    return True
