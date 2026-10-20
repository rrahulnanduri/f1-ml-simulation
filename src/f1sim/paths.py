"""Locations of generated data: the per-user cache folder, never the repo.

The root is ``platformdirs.user_cache_dir("f1sim")``. Set the ``F1SIM_CACHE_DIR``
environment variable to use another folder; it is read on every call, so tests can
monkeypatch it. Every function creates its folder if needed and returns it.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

ENV_VAR = "F1SIM_CACHE_DIR"
APP_NAME = "f1sim"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Root of all generated data (``$F1SIM_CACHE_DIR`` or the platform user cache dir)."""
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        root = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False))
    return _ensure(root)


def tum_dir() -> Path:
    """Downloaded TUM racetrack-database CSVs, one subfolder per pinned commit."""
    return _ensure(cache_dir() / "tum")


def tracks_dir() -> Path:
    """Built track files (``<track_id>.track.json``)."""
    return _ensure(cache_dir() / "tracks")


def ghosts_dir() -> Path:
    """Ghost laps (``.npz`` + ``.json``)."""
    return _ensure(cache_dir() / "ghosts")


def fastf1_dir() -> Path:
    """FastF1's own HTTP and parsed-data cache."""
    return _ensure(cache_dir() / "fastf1")
