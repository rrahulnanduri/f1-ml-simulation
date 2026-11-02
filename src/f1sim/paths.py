"""Locations of generated data: the per-user cache folder, never the repo.

The root is ``platformdirs.user_cache_dir("f1sim")``. Set the ``F1SIM_CACHE_DIR``
environment variable to use another folder; it is read on every call, so tests can
monkeypatch it. Every function creates its folder if needed and returns it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import platformdirs

ENV_VAR = "F1SIM_CACHE_DIR"
BUNDLED_ENV_VAR = "F1SIM_BUNDLED_DIR"
APP_NAME = "f1sim"
_TRACK_ID = re.compile(r"[A-Za-z0-9_-]+")


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


def track_file(track_id: str) -> Path:
    """Where the built track ``track_id`` is (``f1sim build-track``). ValueError unless the id
    is a plain name (letters, digits, ``_``, ``-``), so it can't point outside the folder."""
    if not _TRACK_ID.fullmatch(track_id):
        raise ValueError(f"Not a track id: {track_id!r} (letters, digits, '_' and '-' only)")
    return tracks_dir() / f"{track_id}.track.json"


def find_track(track_id: str) -> Path | None:
    """The built track ``track_id`` to load: the user's own build, else the one that ships
    with the app, else None."""
    own = track_file(track_id)
    if own.exists():
        return own
    shipped = bundled_dir() / "tracks" / own.name
    return shipped if shipped.exists() else None


def ghosts_dir() -> Path:
    """Ghost laps (``.npz`` + ``.json``)."""
    return _ensure(cache_dir() / "ghosts")


def ghost_dirs() -> list[Path]:
    """Where to look for ghost laps, the user's own first, then the ones that ship."""
    return [ghosts_dir(), bundled_dir() / "ghosts"]


def bundled_dir() -> Path:
    """Tracks and real laps that ship with the app (``f1sim/data/bundled``, written by
    ``scripts/bundle_data.py``; D52). Read-only. ``$F1SIM_BUNDLED_DIR`` overrides it."""
    override = os.environ.get(BUNDLED_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "data" / "bundled"


def fastf1_dir() -> Path:
    """FastF1's own HTTP and parsed-data cache."""
    return _ensure(cache_dir() / "fastf1")


def openf1_dir() -> Path:
    """Cached OpenF1 API responses (JSON, one file per request URL)."""
    return _ensure(cache_dir() / "openf1")
