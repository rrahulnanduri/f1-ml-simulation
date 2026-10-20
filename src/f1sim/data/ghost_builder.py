"""Real-driver ghost laps (docs/m3-interfaces.md section M, docs/data-formats.md section 3).

A ghost is a real lap's position trace in the track frame, smoothed and resampled at 20 Hz, with
the car's throttle, brake and gear alongside:

1. Positions come from ``laps.reference_lap`` (about 3.7 Hz) and are mapped into the track frame
   with the track's stored ``fastf1_alignment``. Both sources share the raw frame, so any
   driver's lap maps with the same transform.
2. ``x(t)`` and ``y(t)`` are each fitted with ``scipy.interpolate.make_smoothing_spline``, which
   picks its smoothing by generalised cross-validation, and evaluated every ``1 / rate_hz`` s over
   ``[0, lap_time]``. Joining the raw samples with straight lines instead makes the speed jump at
   every sample: 170 g at the 99th percentile on Hamilton's 2024 Silverstone lap.
3. ``speed`` is the norm of the spline derivative. ``s`` is the projection onto the centre line
   (global for the first sample, then hinted by the previous one), unwrapped across the line so it
   increases over the lap and starts near 0.
4. Throttle is interpolated linearly from the car data, brake and gear by nearest sample. The
   speed sensor is only a cross-check: ``meta["speed_check_kmh"]`` is the median absolute
   difference between it and the spline speed.

Acceleration means the rate of change of speed, ``dv/dt`` (the quantity that showed the 170 g
steps). A smoothed ghost with ``|dv/dt| > 8 g`` anywhere, or with fewer than 95% of its samples
within 1 m of the track edges, is rejected with ``ValueError``: that means bad data or the wrong
layout. The total acceleration including the lateral part (up to about 6 g through Maggotts and
Becketts) is recorded in the meta as ``max_total_accel_g`` but not checked.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import make_smoothing_spline

from f1sim import paths
from f1sim.data import laps
from f1sim.data.alignment import Alignment, apply_alignment
from f1sim.data.laps import CarTrace
from f1sim.track.track import Track

__all__ = [
    "MAX_ACCEL_G",
    "MIN_NEAR_EDGE_RATIO",
    "EDGE_TOLERANCE_M",
    "GhostLap",
    "build_ghost",
    "ghost_from_positions",
    "ghost_path",
    "load_ghost",
    "near_edge_ratio",
    "project_along",
    "resample_channels",
    "save_ghost",
    "session_label",
    "smooth_positions",
]

G = 9.80665
MAX_ACCEL_G = 8.0
"""Reject a ghost whose smoothed ``|dv/dt|`` exceeds this anywhere."""
MIN_NEAR_EDGE_RATIO = 0.95
"""Reject a ghost with fewer than this share of samples within ``EDGE_TOLERANCE_M`` of the edges."""
EDGE_TOLERANCE_M = 1.0

GHOST_FORMAT = "f1sim.ghost"
GHOST_FORMAT_VERSION = 1
_ARRAYS = ("t", "x", "y", "s", "speed", "throttle", "brake", "gear")
_MIN_SAMPLES = 5  # make_smoothing_spline needs at least 5 points


@dataclass(frozen=True, eq=False)
class GhostLap:
    """A ghost lap (data-formats section 3): arrays sampled at ``meta["rate_hz"]``.

    ``t`` s from lap start, ``x``/``y`` m in the track frame, ``s`` m along the centre line
    (monotonic), ``speed`` m/s, ``throttle`` 0..1, ``brake`` 0/1 and ``gear`` int.
    ``meta`` holds ``driver``, ``team`` (or ``""``), ``lap_time_s``, ``year``, ``session``,
    ``lap_source``, ``track_build`` and the build statistics.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    speed: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    meta: dict


# ---------------------------------------------------------------- paths


def session_label(kind: str) -> str:
    """The session part of a ghost file name: the session kind itself ("Q" -> "Q")."""
    label = str(kind).strip().upper()
    if not label:
        raise ValueError("Empty session kind")
    return label


def ghost_path(year: int, track_id: str, kind: str, driver: str) -> Path:
    """``ghosts/<year>_<track_id>_<session>_<driver>.npz`` in ``paths.ghosts_dir()``."""
    drv = str(driver).strip().upper()
    if not drv:
        raise ValueError("Empty driver")
    return paths.ghosts_dir() / f"{int(year)}_{track_id}_{session_label(kind)}_{drv}.npz"


# ---------------------------------------------------------------- pure pieces (offline tests)


def _eval_times(lap_time: float, rate_hz: float) -> np.ndarray:
    """Every ``1 / rate_hz`` s from 0, ending exactly at ``lap_time``."""
    if not (lap_time > 0 and rate_hz > 0):
        raise ValueError(f"lap_time ({lap_time}) and rate_hz ({rate_hz}) must be positive")
    n = int(math.floor(lap_time * rate_hz + 1e-9)) + 1
    t = np.arange(n) / rate_hz
    if lap_time - t[-1] > 1e-6:
        t = np.append(t, lap_time)
    else:
        t[-1] = lap_time
    return t


def _nearest(t_src: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``values`` of the sample nearest in time to each ``t`` (ties go to the earlier one)."""
    if t_src.size == 1:
        return np.full(t.shape, values[0])
    j = np.clip(np.searchsorted(t_src, t), 1, t_src.size - 1)
    earlier = (t - t_src[j - 1]) <= (t_src[j] - t)
    return values[np.where(earlier, j - 1, j)]


def resample_channels(car: CarTrace, t: np.ndarray) -> dict[str, np.ndarray]:
    """Car data at times ``t``: throttle linear, brake and gear nearest sample, speed linear.

    Outside the car data's time range the end values are held.
    """
    if car.t.size == 0:
        raise ValueError("No car-data samples")
    return {
        "throttle": np.interp(t, car.t, car.throttle01),
        "brake": _nearest(car.t, car.brake01, t).astype(np.float64),
        "gear": _nearest(car.t, car.gear, t).astype(np.int8),
        "speed_sensor": np.interp(t, car.t, car.speed_ms),
    }


def smooth_positions(
    t_raw: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray, t: np.ndarray
) -> dict[str, np.ndarray]:
    """Fit smoothing splines (automatic lam) to ``x(t)``, ``y(t)`` and evaluate them at ``t``.

    Returns ``x``, ``y``, ``speed`` (m/s), ``accel`` (``dv/dt``, m/s^2) and ``accel_total``
    (norm of the acceleration vector, m/s^2).
    """
    t_raw = np.asarray(t_raw, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    if t_raw.size < _MIN_SAMPLES:
        raise ValueError(f"Too few position samples ({t_raw.size}) to smooth")
    if np.any(np.diff(t_raw) <= 0):
        raise ValueError("Position sample times must be strictly increasing")
    sx = make_smoothing_spline(t_raw, x_raw)
    sy = make_smoothing_spline(t_raw, y_raw)
    vx, vy = sx.derivative(1)(t), sy.derivative(1)(t)
    ax, ay = sx.derivative(2)(t), sy.derivative(2)(t)
    speed = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.where(speed > 1e-6, (vx * ax + vy * ay) / speed, np.hypot(ax, ay))
    return {
        "x": sx(t),
        "y": sy(t),
        "speed": speed,
        "accel": accel,
        "accel_total": np.hypot(ax, ay),
    }


def project_along(track: Track, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(s, n)`` of a lap's samples, ``s`` unwrapped to increase over the lap from near 0.

    The first sample is projected over the whole loop, every later one with the previous
    sample's segment as the hint. ``n`` is the signed lateral offset.
    """
    # The hint window must cover the longest step between samples (rate_hz may be low)
    step = np.hypot(np.diff(x), np.diff(y)).max(initial=0.0)
    window = math.ceil(step / track.data.resolution_m) + 5
    window = int(min(track.data.n_points // 2, max(10, window)))
    s_raw = np.empty(x.size)
    n = np.empty(x.size)
    s0, n0, idx = track.project(x[:1], y[:1])
    s_raw[0], n[0], hint = s0[0], n0[0], idx
    for k in range(1, x.size):
        sk, nk, hint = track.project(x[k : k + 1], y[k : k + 1], hint=hint, window=window)
        s_raw[k], n[k] = sk[0], nk[0]

    length = track.length
    ds = (np.diff(s_raw) + length / 2) % length - length / 2
    start = s_raw[0] - length if s_raw[0] > length / 2 else s_raw[0]
    s = start + np.concatenate([[0.0], np.cumsum(ds)])
    return np.maximum.accumulate(s), n


def near_edge_ratio(
    track: Track, s: np.ndarray, n: np.ndarray, tolerance_m: float = EDGE_TOLERANCE_M
) -> float:
    """Share of samples inside the track edges or within ``tolerance_m`` of them."""
    w_left, w_right = track.widths_at(s)
    ok = (n <= w_left + tolerance_m) & (n >= -w_right - tolerance_m)
    return float(ok.mean())


def ghost_from_positions(
    track: Track,
    t_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    lap_time_s: float,
    car: CarTrace,
    *,
    meta: dict | None = None,
    rate_hz: float = 20.0,
    validate: bool = True,
) -> GhostLap:
    """Smooth a lap's positions (already in the track frame) into a ghost and validate it.

    ``t_raw`` runs from the lap start. ``meta`` is copied into the result, which adds
    ``lap_time_s``, ``rate_hz``, ``speed_check_kmh``, ``max_accel_g``, ``p99_accel_g``,
    ``max_total_accel_g`` and ``near_edge_ratio``. Raises ``ValueError`` (when ``validate``)
    if ``|dv/dt| > MAX_ACCEL_G`` anywhere or fewer than ``MIN_NEAR_EDGE_RATIO`` of the samples
    are within ``EDGE_TOLERANCE_M`` of the edges.
    """
    t = _eval_times(float(lap_time_s), float(rate_hz))
    sm = smooth_positions(t_raw, x_raw, y_raw, t)
    s, n = project_along(track, sm["x"], sm["y"])
    ch = resample_channels(car, t)

    accel_g = np.abs(sm["accel"]) / G
    near = near_edge_ratio(track, s, n)
    out_meta = dict(meta or {})
    out_meta.update(
        lap_time_s=float(lap_time_s),
        rate_hz=float(rate_hz),
        speed_check_kmh=float(np.median(np.abs(sm["speed"] - ch["speed_sensor"])) * 3.6),
        max_accel_g=float(accel_g.max()),
        p99_accel_g=float(np.percentile(accel_g, 99)),
        max_total_accel_g=float(sm["accel_total"].max() / G),
        near_edge_ratio=near,
    )
    if validate:
        who = out_meta.get("driver", "the lap")
        if accel_g.max() > MAX_ACCEL_G:
            k = int(accel_g.argmax())
            raise ValueError(
                f"Ghost of {who}: |dv/dt| reaches {accel_g[k]:.1f} g at t = {t[k]:.2f} s "
                f"(limit {MAX_ACCEL_G:g} g); the position data looks bad"
            )
        if near < MIN_NEAR_EDGE_RATIO:
            raise ValueError(
                f"Ghost of {who}: only {near:.1%} of samples within {EDGE_TOLERANCE_M:g} m of "
                f"the track edges (need {MIN_NEAR_EDGE_RATIO:.0%}); wrong layout or alignment?"
            )
    return GhostLap(
        t=t,
        x=sm["x"],
        y=sm["y"],
        s=s,
        speed=sm["speed"],
        throttle=ch["throttle"],
        brake=ch["brake"],
        gear=ch["gear"],
        meta=out_meta,
    )


# ---------------------------------------------------------------- build from a source


def _stored_alignment(track: Track) -> Alignment:
    meta = track.data.fastf1_alignment
    if not meta:
        raise ValueError(
            f"Track {track.data.id!r} has no fastf1_alignment; rebuild it with a reference year"
        )
    return Alignment(**{k: meta[k] for k in Alignment.__dataclass_fields__})


def _track_build(track: Track) -> dict:
    from f1sim.data.io import read_track_builder_version
    from f1sim.data.track_pipeline import track_path

    try:
        builder_version = read_track_builder_version(track_path(track.data.id))
    except (OSError, ValueError):
        builder_version = None
    return {
        "track_id": track.data.id,
        "length_m": round(track.length, 4),
        "builder_version": builder_version,
        "geometry_commit": track.data.source.get("commit"),
    }


def build_ghost(
    track_id: str,
    year: int,
    *,
    driver: str | int | None = None,
    kind: str = "Q",
    source: str = "auto",
    rate_hz: float = 20.0,
) -> GhostLap:
    """The ghost of ``driver``'s fastest ``kind`` lap at ``track_id`` in ``year``.

    ``driver=None`` takes the session's fastest lap; ``meta["driver"]`` is then that driver's
    acronym. The track must be built (``load_track``) with a stored alignment.
    """
    from f1sim.data.track_pipeline import load_track

    track = load_track(track_id)
    alignment = _stored_alignment(track)
    lap, lap_source = laps.reference_lap(track_id, year, kind=kind, driver=driver, source=source)
    # Ask for the lap's own driver so both calls pick the same lap
    car = laps.reference_car_data(track_id, year, kind=kind, driver=lap.driver, source=lap_source)
    x, y = apply_alignment(alignment, lap.x_m, lap.y_m)
    meta = {
        "driver": lap.driver,
        "team": car.team or "",
        "lap_time_s": float(lap.lap_time_s),
        "year": int(year),
        "session": session_label(kind),
        "lap_source": lap_source,
        "track_build": _track_build(track),
        "track_id": track_id,
    }
    return ghost_from_positions(track, lap.t, x, y, lap.lap_time_s, car, meta=meta, rate_hz=rate_hz)


# ---------------------------------------------------------------- save / load


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def _replace_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_ghost(g: GhostLap, path: Path) -> None:
    """Write ``path`` (NPZ of the arrays) and ``path.with_suffix(".json")`` (meta), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(g, name)) for name in _ARRAYS}
    doc = {"format": GHOST_FORMAT, "version": GHOST_FORMAT_VERSION, **g.meta}
    text = json.dumps(doc, indent=2, default=_json_default, allow_nan=False) + "\n"
    _replace_atomic(path, lambda f: np.savez(f, **arrays))
    _replace_atomic(path.with_suffix(".json"), lambda f: f.write(text.encode("utf-8")))


def load_ghost(path: Path) -> GhostLap:
    """Read a ghost written by :func:`save_ghost`."""
    path = Path(path)
    side = path.with_suffix(".json")
    try:
        doc = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{side}: not valid JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("format") != GHOST_FORMAT:
        raise ValueError(f"{side}: not an f1sim ghost sidecar")
    if doc.get("version") != GHOST_FORMAT_VERSION:
        raise ValueError(
            f"{side}: unsupported ghost version {doc.get('version')!r} "
            f"(this f1sim reads version {GHOST_FORMAT_VERSION}); rebuild the ghost"
        )
    meta = {k: v for k, v in doc.items() if k not in ("format", "version")}
    with np.load(path, allow_pickle=False) as npz:
        missing = [name for name in _ARRAYS if name not in npz.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {', '.join(missing)}")
        arrays = {name: npz[name] for name in _ARRAYS}
    return GhostLap(**arrays, meta=meta)
