"""TUM racetrack-database: download (pinned commit, cached) and CSV parsing.

Source: https://github.com/TUMFTM/racetrack-database (LGPL-3.0). Files are downloaded at
build time into the user cache and never committed (see THIRD_PARTY.md).

``tracks/<Name>.csv``: header ``# x_m,y_m,w_tr_right_m,w_tr_left_m``, then centre-line
points about 5 m apart with the track width to the right and left of each point.
``racelines/<Name>.csv``: header ``# x_m,y_m``, a minimum-curvature racing line.
"""

from __future__ import annotations

import difflib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np

from f1sim import paths

TUM_REPO = "TUMFTM/racetrack-database"
# HEAD of the default branch ("Updated readme", 2021-09-18), from `git ls-remote`.
TUM_COMMIT = "e59595d1f3573b30d1ded6a08984935b957688e0"
TUM_LICENSE = "LGPL-3.0"

# File stems in tracks/ at TUM_COMMIT.
TUM_TRACKS: tuple[str, ...] = (
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Norisring",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "Suzuka",
    "YasMarina",
    "Zandvoort",
)
# File stems in racelines/ at TUM_COMMIT: every track has one.
TUM_RACELINES: tuple[str, ...] = TUM_TRACKS

TRACK_COLUMNS = ("x_m", "y_m", "w_tr_right_m", "w_tr_left_m")
RACELINE_COLUMNS = ("x_m", "y_m")

_RAW_BASE = "https://raw.githubusercontent.com"
_TIMEOUT_S = 30.0
_MAX_BYTES = 8 * 1024 * 1024  # the largest TUM CSV is ~0.2 MB


class TumTrack(NamedTuple):
    """Centre line and widths in metres, as stored in the CSV (closing point not repeated)."""

    x: np.ndarray
    y: np.ndarray
    w_right: np.ndarray
    w_left: np.ndarray


def tum_url(kind: str, name: str) -> str:
    """Raw-file URL of ``<kind>/<name>.csv`` at the pinned commit."""
    return f"{_RAW_BASE}/{TUM_REPO}/{TUM_COMMIT}/{kind}/{name}.csv"


def _check_name(name: str, valid: tuple[str, ...], kind: str) -> None:
    if name not in valid:
        close = difflib.get_close_matches(name, valid, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"Unknown TUM {kind} name {name!r}.{hint} Valid names: {', '.join(valid)}")


def _fetch(kind: str, name: str, valid: tuple[str, ...]) -> Path:
    _check_name(name, valid, kind)
    dest = paths.tum_dir() / TUM_COMMIT / kind / f"{name}.csv"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = tum_url(kind, name)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            payload = resp.read(_MAX_BYTES + 1)
    except OSError as e:  # URLError, HTTPError and timeouts are all OSError
        raise OSError(f"Could not download TUM {kind} {name!r} from {url}: {e}") from e
    if len(payload) > _MAX_BYTES:
        raise OSError(f"Unexpected content from {url}: over {_MAX_BYTES} bytes")
    if not payload.lstrip().startswith(b"#"):
        raise OSError(f"Unexpected content from {url}: no '#' header line")

    # Write next to the target, then rename, so a crash never leaves a partial cached file.
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


def fetch_tum_csv(name: str) -> Path:
    """Path to ``tracks/<name>.csv`` at TUM_COMMIT, downloading it into the cache if needed."""
    return _fetch("tracks", name, TUM_TRACKS)


def fetch_tum_raceline(name: str) -> Path:
    """Path to ``racelines/<name>.csv`` at TUM_COMMIT, downloading it into the cache if needed."""
    return _fetch("racelines", name, TUM_RACELINES)


def _read_columns(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    """Read a TUM CSV into an (N, len(columns)) float64 array, checking the header."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("#"):
        header = tuple(c.strip() for c in first.lstrip("#").split(","))
        if header != columns:
            raise ValueError(f"{path}: expected header '# {','.join(columns)}', got {first!r}")
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(f"{path}: expected {len(columns)} columns, got {data.shape[1]}")
    if data.shape[0] < 3:
        raise ValueError(f"{path}: need at least 3 points, got {data.shape[0]}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: contains non-finite values")
    return data


def parse_tum_csv(path: Path) -> TumTrack:
    """Parse a TUM ``tracks/`` CSV into float64 arrays (x, y, w_right, w_left), in metres."""
    data = _read_columns(path, TRACK_COLUMNS)
    if np.any(data[:, 2:] < 0):
        raise ValueError(f"{path}: negative track width")
    x, y, w_right, w_left = (np.ascontiguousarray(data[:, i]) for i in range(4))
    return TumTrack(x=x, y=y, w_right=w_right, w_left=w_left)


def parse_tum_raceline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a TUM ``racelines/`` CSV into float64 arrays (x, y), in metres."""
    data = _read_columns(path, RACELINE_COLUMNS)
    return np.ascontiguousarray(data[:, 0]), np.ascontiguousarray(data[:, 1])
