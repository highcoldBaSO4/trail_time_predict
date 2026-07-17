from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


APP_DIRECTORY_NAME = "TrailTimePredictor"


def resource_root() -> Path:
    """Return the source directory or PyInstaller's bundled resource directory."""
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parents[1]


def user_data_directory() -> Path:
    """Return a writable per-user directory for cache and runtime logs."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path(tempfile.gettempdir())
    target = root / APP_DIRECTORY_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def weather_cache_directory(configured_path: str) -> Path:
    if getattr(sys, "frozen", False):
        target = user_data_directory() / "weather_cache"
    else:
        target = Path(__file__).resolve().parents[1] / configured_path
    target.mkdir(parents=True, exist_ok=True)
    return target
