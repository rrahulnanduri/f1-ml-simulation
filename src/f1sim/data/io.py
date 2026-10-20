"""Read and write track files (docs/data-formats.md section 1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from f1sim import __version__
from f1sim.track.track import TrackData

BUILDER_VERSION: int = 1

TRACK_FORMAT = "f1sim.track"
TRACK_FORMAT_VERSION = 2
# Version 2 adds the optional limit_left/limit_right columns and the limits block
READABLE_VERSIONS = (1, 2)

# Decimals kept per column: 0.1 mm for lengths, ~1e-6 rad for heading, 1e-8 /m for curvature.
_DECIMALS = {
    "x": 4,
    "y": 4,
    "s": 4,
    "w_left": 4,
    "w_right": 4,
    "heading": 6,
    "curvature": 8,
}
_COLUMNS = tuple(_DECIMALS)


def _round_list(values, decimals: int) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float64), decimals).tolist()


def track_to_dict(data: TrackData) -> dict:
    """The JSON document for ``data`` (rounded as it will be stored)."""
    return {
        "format": TRACK_FORMAT,
        "version": TRACK_FORMAT_VERSION,
        "id": data.id,
        "name": data.name,
        "fastf1_location": data.fastf1_location,
        "source": dict(data.source),
        "resolution_m": float(data.resolution_m),
        "length_m": round(float(data.length_m), 4),
        "direction": data.direction,
        "sectors_s": _round_list(data.sectors_s, 4),
        "points": {
            **{c: _round_list(getattr(data, c), _DECIMALS[c]) for c in _COLUMNS},
            **(
                {
                    "limit_left": _round_list(data.limit_left, 4),
                    "limit_right": _round_list(data.limit_right, 4),
                }
                if data.has_limits
                else {}
            ),
        },
        "limits": data.limits,
        "fastf1_alignment": data.fastf1_alignment,
        "built_with": {"f1sim": __version__, "builder_version": BUILDER_VERSION},
    }


def _dumps(doc: dict) -> str:
    """Readable JSON: header fields indented, each points column on a single line."""
    lines = []
    for key, value in doc.items():
        if key == "points":
            cols = [
                f"    {json.dumps(c)}: {json.dumps(v, allow_nan=False)}" for c, v in value.items()
            ]
            body = "{\n" + ",\n".join(cols) + "\n  }"
        else:
            body = json.dumps(value, indent=2, allow_nan=False).replace("\n", "\n  ")
        lines.append(f"  {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write_track_json(data: TrackData, path: Path) -> None:
    """Write ``data`` to ``path`` atomically, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dumps(track_to_dict(data))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("format") != TRACK_FORMAT:
        found = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise ValueError(
            f"{path}: not an f1sim track file (format {found!r}, expected {TRACK_FORMAT!r})"
        )
    if doc.get("version") not in READABLE_VERSIONS:
        raise ValueError(
            f"{path}: unsupported track file version {doc.get('version')!r} "
            f"(this f1sim reads versions {READABLE_VERSIONS}); rebuild the track"
        )
    return doc


def read_track_builder_version(path: Path) -> int | None:
    """``built_with.builder_version`` of a track file, to spot files built by older code."""
    v = _load(path).get("built_with", {}).get("builder_version")
    return int(v) if v is not None else None


def _optional(points: dict, column: str) -> np.ndarray | None:
    values = points.get(column)
    return None if values is None else np.asarray(values, dtype=np.float64)


def read_track_json(path: Path) -> TrackData:
    """Load a track file written by :func:`write_track_json`; validates format and version."""
    doc = _load(path)
    try:
        points = doc["points"]
        arrays = {c: np.asarray(points[c], dtype=np.float64) for c in _COLUMNS}
        return TrackData(
            id=doc["id"],
            name=doc["name"],
            **arrays,
            length_m=float(doc["length_m"]),
            resolution_m=float(doc["resolution_m"]),
            sectors_s=tuple(doc.get("sectors_s") or ()),
            fastf1_location=doc.get("fastf1_location"),
            direction=doc.get("direction", ""),
            source=dict(doc.get("source") or {}),
            fastf1_alignment=doc.get("fastf1_alignment"),
            limit_left=_optional(points, "limit_left"),
            limit_right=_optional(points, "limit_right"),
            limits=doc.get("limits"),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: malformed track file ({e!r})") from e
