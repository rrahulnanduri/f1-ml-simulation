"""Replay viewer (M3-L): play back one or more recorded laps on a built track.

``run(track_id, traces, labels)`` opens a resizable arcade window. Traces are NPZ files with at
least ``t``, ``x``, ``y`` (optionally ``s`` and ``speed``): ghost laps (data-formats section 3),
a run's ``best_lap.npz`` (section 5) or anything else with those columns.

The testable logic is pure numpy/Python, without arcade objects:

- :func:`load_trace` reads a file into a :class:`Trace`.
- :func:`project_s` / :func:`unwrap_s` give a trace a monotonic arc length.
- :func:`smooth_heading` computes the heading once per trace (central differences, then a
  circular moving average over ~0.2 s), and :func:`interp_position` / :func:`interp_heading`
  interpolate linearly between samples. :class:`PreparedTrace` draws cars with
  :func:`hermite` instead (cubic, with slopes from :func:`numpy.gradient`): velocity and yaw
  rate are then continuous across samples, where linear interpolation of a 20 Hz ghost changes
  velocity in a jump at every sample (a visible tick, about 15 g between frames in fast
  corners).
- :func:`delta_at_s` is the time difference at equal ``s``.
- :class:`PlaybackClock` (pause, seek, speed steps, restart) and :class:`ReplaySession` (all
  traces on one clock) drive the window.

**Time and arc-length frame.** Every trace's ``s`` is unwrapped and shifted by whole laps so
that ``s[0]`` lies in ``[-L/2, L/2)``: the start/finish line is ``s = 0`` and a lead-in before
the line is negative. Each trace's *lap clock* is ``t - t(s = 0)`` (``t - t[0]`` if the trace
never reaches ``s = 0``). All traces play on one clock in that frame, so cars cross the line
together however long their lead-in is, and the live delta is ``t_2(s) - t_1(s)`` in lap-clock
time, evaluated at the smaller of the two cars' current ``s`` (the furthest point both cars
have reached).

**Window.** The track (asphalt, kerbs, run-off, chequered strip, white lines) is the cached
:class:`~f1sim.app.render_track.TrackRenderer`, drawn under an ``arcade.Camera2D`` synced from
:class:`~f1sim.app.camera.Camera` each frame, exactly as in :mod:`f1sim.app.drive`, into the
offscreen scene of :mod:`f1sim.app.ui.glass`. Cars are pixel-art sprites
(:mod:`f1sim.app.ui.pixelcar`): ghosts in their team livery with a light outline, the AI lap
in titanium, black and red; the followed car ("self") is drawn on top with a red label and
kept at the middle of the area the HUD leaves free. The HUD (:mod:`f1sim.app.ui.hud`) is a
ticker and headline, a timing tower ranked by lap time, the gap to the reference (the fastest
other trace; the gap card's ↗ cycles through the others), the speed, a paging card with
sectors, speed vs the reference and the gap bar, and a footer with the controls.
:class:`ReplayTiming` computes all of it without a window.

Controls: Space pause, Left/Right seek -/+2 s, Up/Down playback speed (0.25x to 8x), ``R``
restart, ``F`` cycle the followed car (then free camera), ``Tab`` next card page, mouse wheel
zoom at the cursor (pages the card when over it), click the card for the next page, click the
gap card for the next reference, right-drag pan (turns follow off), ``Esc`` back to the
menu (quit when opened on its own with ``f1sim replay``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import arcade
import numpy as np

from f1sim.app import shell
from f1sim.app.camera import Camera
from f1sim.app.drive import format_lap_time
from f1sim.app.shell import Screen
from f1sim.app.ui import theme
from f1sim.app.ui import timing as tm
from f1sim.track.track import Track

MAX_FRAME_S = 0.25
"""Longest frame time fed to the playback clock (a stall doesn't jump the replay)."""
SPEED_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""Playback speeds selectable with Up/Down."""
SEEK_STEP_S = 2.0
"""Left/Right seek step, s of trace time."""
HEADING_SMOOTH_S = 0.2
"""Width of the circular moving average applied to the heading, s."""
STILL_SPEED_MPS = 0.1
"""Below this speed a sample's direction of travel is undefined; its heading is carried over."""
REPROJECT_JUMP_M = 50.0
"""A hinted projection jumping further than this along the track is redone globally."""
PROJECT_WINDOW = 10

CAR_LENGTH_M = theme.CAR_LENGTH_M
CAR_WIDTH_M = theme.CAR_WIDTH_M
FOLLOW_PIXELS_PER_METRE = 4.0
"""Initial zoom of the follow camera."""
ZOOM_STEP = 1.15
"""Zoom factor per mouse-wheel notch."""

# ---------------------------------------------------------------------------- traces


@dataclass(frozen=True)
class Trace:
    """One recorded lap: time ``t`` (s), position ``x``, ``y`` (m, track frame), arc length
    ``s`` (m) and ``speed`` (m/s), all 1-D of the same length.

    ``s`` is an **empty array** when the file had none; :func:`with_track_s` (called by
    :func:`run`) fills it by projecting onto the track. ``lap_time_s = t[-1] - t[0]``.

    ``meta`` is the ghost sidecar (``<name>.json`` from :mod:`f1sim.data.ghost_builder`:
    ``driver``, ``team``, ``year``...) or ``None``; ``gear`` is empty when the file has none.
    """

    label: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    speed: np.ndarray
    lap_time_s: float
    meta: dict | None = None
    gear: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def is_ghost(self) -> bool:
        """A real-driver ghost lap (it has a ghost sidecar)."""
        return self.meta is not None

    @property
    def has_s(self) -> bool:
        return self.s.size == self.t.size


def _default_label(path: Path) -> str:
    if path.stem == "best_lap" and path.parent.name:
        return f"{path.parent.name} best"
    return path.stem


def _column(data, name: str, path: Path) -> np.ndarray:
    arr = np.asarray(data[name], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{path}: column '{name}' must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: column '{name}' contains non-finite values")
    return arr


GHOST_FORMAT = "f1sim.ghost"
"""``format`` of a ghost sidecar (``f1sim.data.ghost_builder.GHOST_FORMAT``; not imported, as
that module pulls in scipy and the data sources)."""


def load_ghost_meta(path: Path) -> dict | None:
    """The ghost sidecar ``path.with_suffix(".json")`` as a dict (without ``format`` and
    ``version``), or ``None`` if there is none or it isn't a ghost sidecar."""
    side = Path(path).with_suffix(".json")
    try:
        doc = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("format") != GHOST_FORMAT:
        return None
    return {k: v for k, v in doc.items() if k not in ("format", "version")}


def load_trace(path: Path, label: str | None = None) -> Trace:
    """Read a trace NPZ with at least ``t``, ``x``, ``y``; ``s``, ``speed`` and ``gear`` are
    optional, and a ghost sidecar next to it is read into ``meta``.

    A missing ``speed`` is computed from finite differences of the positions; a missing ``s``
    is left empty (see :class:`Trace`). Without ``label``, a ghost is named from its sidecar
    (``"VER ’25"``), anything else from the file name. Raises ``ValueError`` for missing
    columns, mismatched lengths, fewer than two samples or a ``t`` that isn't strictly
    increasing.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [c for c in ("t", "x", "y") if c not in data.files]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        t, x, y = (_column(data, c, path) for c in ("t", "x", "y"))
        s = _column(data, "s", path) if "s" in data.files else np.empty(0)
        speed = _column(data, "speed", path) if "speed" in data.files else None
        gear = _column(data, "gear", path) if "gear" in data.files else np.empty(0)
    n = t.size
    for name, arr in (("x", x), ("y", y), ("s", s), ("speed", speed), ("gear", gear)):
        if arr is None or (name in ("s", "gear") and arr.size == 0):
            continue  # optional column absent
        if arr.size != n:
            raise ValueError(f"{path}: column '{name}' has {arr.size} samples, expected {n}")
    if n < 2:
        raise ValueError(f"{path}: a trace needs at least 2 samples, got {n}")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{path}: 't' must be strictly increasing")
    if speed is None:
        speed = np.hypot(np.gradient(x, t), np.gradient(y, t))
    meta = load_ghost_meta(path)
    if label is None:
        label = theme.ghost_short_name(meta) or _default_label(path)
    return Trace(
        label=label,
        t=t,
        x=x,
        y=y,
        s=s,
        speed=speed,
        lap_time_s=float(t[-1] - t[0]),
        meta=meta,
        gear=gear,
    )


def unwrap_s(s: np.ndarray, length: float) -> np.ndarray:
    """Unwrap arc lengths in ``[0, length)`` (or already unwrapped) into a continuous series.

    Steps are wrapped to ``[-length/2, length/2)`` and summed; the result is then shifted by
    whole laps so that its first value lies in ``[-length/2, length/2)`` (the start line is
    ``s = 0``, a lead-in before it is negative).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return s.copy()
    steps = (np.diff(s) + 0.5 * length) % length - 0.5 * length
    out = s[0] + np.concatenate([[0.0], np.cumsum(steps)])
    shift = np.floor((out[0] + 0.5 * length) / length) * length
    return out - shift


def project_s(track: Track, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Arc length of each sample by projection: global for the first sample, then hinted from
    the previous one (redone globally after a jump of more than ``REPROJECT_JUMP_M``).
    Returns the wrapped ``s`` in ``[0, length)``; see :func:`unwrap_s`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.empty(x.size)
    if x.size == 0:
        return out
    length = track.length
    s, _, idx = track.project(x[:1], y[:1])
    out[0], hint = float(s[0]), int(idx[0])
    for k in range(1, x.size):
        s, _, idx = track.project(x[k : k + 1], y[k : k + 1], hint=hint, window=PROJECT_WINDOW)
        ds = (float(s[0]) - out[k - 1] + 0.5 * length) % length - 0.5 * length
        if abs(ds) > REPROJECT_JUMP_M:
            s, _, idx = track.project(x[k : k + 1], y[k : k + 1])
        out[k], hint = float(s[0]), int(idx[0])
    return out


def with_track_s(trace: Trace, track: Track) -> Trace:
    """``trace`` with ``s`` filled in by :func:`project_s` if it has none (else unchanged)."""
    if trace.has_s:
        return trace
    return replace(trace, s=unwrap_s(project_s(track, trace.x, trace.y), track.length))


# ------------------------------------------------------------- interpolation and heading


def _moving_average(a: np.ndarray, half: int) -> np.ndarray:
    """Centred moving average over ``2 * half + 1`` samples; the window shrinks at the ends."""
    if half <= 0 or a.size < 3:
        return a.copy()
    c = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(a.size)
    lo = np.maximum(i - half, 0)
    hi = np.minimum(i + half + 1, a.size)
    return (c[hi] - c[lo]) / (hi - lo)


def smooth_heading(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, smooth_s: float = HEADING_SMOOTH_S
) -> np.ndarray:
    """Heading of travel at each sample, in ``[-pi, pi]``, for a trace sampled at ``t``.

    The direction comes from central differences (``np.gradient``) of the positions; samples
    slower than ``STILL_SPEED_MPS`` take the nearest moving sample's direction. ``cos`` and
    ``sin`` of the heading are then smoothed with a centred moving average about ``smooth_s``
    wide (in samples, from the median sample interval) and recombined with ``atan2``, so the
    filter is unaffected by the ±pi wrap.
    """
    t = np.asarray(t, dtype=np.float64)
    vx = np.gradient(np.asarray(x, dtype=np.float64), t)
    vy = np.gradient(np.asarray(y, dtype=np.float64), t)
    v = np.hypot(vx, vy)
    moving = v > STILL_SPEED_MPS
    if not np.any(moving):
        return np.zeros(t.size)
    idx = np.arange(t.size)
    src = np.where(moving, idx, -1)
    fwd = np.maximum.accumulate(src)  # last moving sample at or before i
    first = int(np.argmax(moving))
    fwd = np.where(fwd < 0, first, fwd)
    c = vx[fwd] / v[fwd]
    s = vy[fwd] / v[fwd]
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    half = int(round(0.5 * smooth_s / dt)) if dt > 0.0 else 0
    return np.arctan2(_moving_average(s, half), _moving_average(c, half))


def interp_position(trace_t, trace_x, trace_y, t) -> tuple[np.ndarray, np.ndarray]:
    """Position at time(s) ``t``, linear between samples and clamped to the ends."""
    return np.interp(t, trace_t, trace_x), np.interp(t, trace_t, trace_y)


def hermite(trace_t, values, slopes, t) -> np.ndarray:
    """Cubic Hermite interpolation of ``values`` (with derivatives ``slopes``) sampled at
    ``trace_t``, at time(s) ``t``, clamped to the ends. Exact at the samples, and its first
    derivative is continuous across them."""
    trace_t = np.asarray(trace_t, dtype=np.float64)
    tq = np.clip(np.asarray(t, dtype=np.float64), trace_t[0], trace_t[-1])
    i = np.clip(np.searchsorted(trace_t, tq, side="right") - 1, 0, trace_t.size - 2)
    h = trace_t[i + 1] - trace_t[i]
    u = (tq - trace_t[i]) / h
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * values[i]
        + (u3 - 2 * u2 + u) * h * slopes[i]
        + (-2 * u3 + 3 * u2) * values[i + 1]
        + (u3 - u2) * h * slopes[i + 1]
    )


def interp_heading(trace_t, heading_unwrapped, t) -> np.ndarray:
    """Heading at time(s) ``t`` in ``[-pi, pi)`` from the unwrapped per-sample heading
    (``np.unwrap(smooth_heading(...))``): linear between samples, so the shorter way round."""
    h = np.interp(t, trace_t, heading_unwrapped)
    return (h + np.pi) % (2.0 * np.pi) - np.pi


def monotonic_s_t(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The samples where ``s`` reaches a new maximum: a strictly increasing ``s`` and its
    times, safe for ``np.interp`` (stalls and small backward steps are dropped)."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if s.size == 0:
        return s, t
    keep = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]])
    return s[keep], t[keep]


def time_at_s(s_mono: np.ndarray, t_mono: np.ndarray, s_query) -> np.ndarray:
    """Time at arc length(s) ``s_query`` (linear); nan outside ``[s_mono[0], s_mono[-1]]``."""
    q = np.asarray(s_query, dtype=np.float64)
    out = np.interp(q, s_mono, t_mono)
    return np.where((q < s_mono[0]) | (q > s_mono[-1]), np.nan, out)


def delta_at_s(t1, s1, t2, s2, s_query) -> np.ndarray:
    """``t_2(s) - t_1(s)`` at equal arc length: negative means trace 2 got there sooner.

    ``s1`` and ``s2`` are each trace's unwrapped arc length (non-monotonic samples are
    dropped, see :func:`monotonic_s_t`); nan where either trace doesn't cover ``s_query``.
    """
    sm1, tm1 = monotonic_s_t(s1, t1)
    sm2, tm2 = monotonic_s_t(s2, t2)
    return time_at_s(sm2, tm2, s_query) - time_at_s(sm1, tm1, s_query)


# ------------------------------------------------------------------------- playback


class PlaybackClock:
    """Playback time in ``[start, end]``, advanced by ``speed`` x the frame time.

    Pure state: :meth:`update` is called from the window's ``on_update(dt)``. Time clamps at
    both ends (playback simply holds the last frame); :meth:`restart` goes back to ``start``
    and resumes.
    """

    def __init__(self, start: float, end: float, speed: float = 1.0):
        if not end >= start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        self.start = float(start)
        self.end = float(end)
        self.time = self.start
        self.paused = False
        self._speed_i = int(np.argmin(np.abs(np.array(SPEED_STEPS) - speed)))

    @property
    def speed(self) -> float:
        return SPEED_STEPS[self._speed_i]

    @property
    def at_end(self) -> bool:
        return self.time >= self.end

    def update(self, dt: float) -> None:
        """Advance by a frame of ``dt`` seconds (ignored while paused)."""
        if not self.paused:
            self.seek_to(self.time + self.speed * min(max(float(dt), 0.0), MAX_FRAME_S))

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def seek(self, delta_s: float) -> None:
        """Move by ``delta_s`` seconds of trace time (clamped)."""
        self.seek_to(self.time + float(delta_s))

    def seek_to(self, time_s: float) -> None:
        self.time = min(max(float(time_s), self.start), self.end)

    def faster(self) -> None:
        self._speed_i = min(self._speed_i + 1, len(SPEED_STEPS) - 1)

    def slower(self) -> None:
        self._speed_i = max(self._speed_i - 1, 0)

    def restart(self) -> None:
        self.time = self.start
        self.paused = False


@dataclass(frozen=True)
class CarPose:
    """Where one trace's car is at the current playback time."""

    x: float
    y: float
    heading: float
    speed: float
    s: float
    lap_clock: float  # time since the line (clamped to the trace), s


class PreparedTrace:
    """A trace with ``s`` unwrapped into the lap frame, its lap clock and smoothed heading,
    all computed once. :meth:`pose` places the car with :func:`hermite` (position and
    heading), so it moves smoothly between samples."""

    def __init__(self, trace: Trace, length: float):
        if not trace.has_s:
            raise ValueError(f"trace '{trace.label}' has no s; call with_track_s first")
        self.trace = trace
        self.s = unwrap_s(trace.s, length)
        self.s_mono, t_mono = monotonic_s_t(self.s, trace.t)
        if self.s_mono[0] <= 0.0 <= self.s_mono[-1]:
            t_origin = float(np.interp(0.0, self.s_mono, t_mono))
        else:
            t_origin = float(trace.t[0])
        self.t = trace.t - t_origin  # lap clock at each sample
        self.t_mono = t_mono - t_origin
        self.heading = np.unwrap(smooth_heading(trace.t, trace.x, trace.y))
        self._dx = np.gradient(trace.x, self.t)
        self._dy = np.gradient(trace.y, self.t)
        self._dh = np.gradient(self.heading, self.t)

    def pose(self, lap_clock: float) -> CarPose:
        tc = min(max(lap_clock, float(self.t[0])), float(self.t[-1]))
        heading = float(hermite(self.t, self.heading, self._dh, tc))
        return CarPose(
            x=float(hermite(self.t, self.trace.x, self._dx, tc)),
            y=float(hermite(self.t, self.trace.y, self._dy, tc)),
            heading=(heading + math.pi) % (2.0 * math.pi) - math.pi,
            speed=float(np.interp(tc, self.t, self.trace.speed)),
            s=float(np.interp(tc, self.t, self.s)),
            lap_clock=tc,
        )


class ReplaySession:
    """All traces on one :class:`PlaybackClock` (lap-clock time), plus the follow target.
    Pure Python/numpy, no window."""

    def __init__(self, traces: list[Trace], length: float, follow: int = 0):
        if not traces:
            raise ValueError("need at least one trace")
        self.prepared = [PreparedTrace(tr, length) for tr in traces]
        start = min(float(p.t[0]) for p in self.prepared)
        end = max(float(p.t[-1]) for p in self.prepared)
        self.clock = PlaybackClock(start, end)
        self.follow: int | None = min(max(int(follow), 0), len(self.prepared) - 1)

    @property
    def labels(self) -> list[str]:
        return [p.trace.label for p in self.prepared]

    def update(self, dt: float) -> None:
        self.clock.update(dt)

    def poses(self) -> list[CarPose]:
        return [p.pose(self.clock.time) for p in self.prepared]

    def cycle_follow(self) -> None:
        """Next trace, and after the last one the free camera (``None``)."""
        if self.follow is None:
            self.follow = 0
        elif self.follow + 1 < len(self.prepared):
            self.follow += 1
        else:
            self.follow = None

    def delta(self, poses: list[CarPose] | None = None) -> float:
        """Live ``t_2(s) - t_1(s)`` at the smaller current ``s`` of the two cars; nan with
        fewer than two traces or where they don't overlap."""
        if len(self.prepared) < 2:
            return float("nan")
        poses = poses if poses is not None else self.poses()
        a, b = self.prepared[0], self.prepared[1]
        s_q = min(poses[0].s, poses[1].s)
        return float(time_at_s(b.s_mono, b.t_mono, s_q) - time_at_s(a.s_mono, a.t_mono, s_q))


def format_delta(d: float) -> str:
    """``+1.234 s`` / ``-0.567 s``, or ``--.---`` for nan."""
    return "--.---" if not np.isfinite(d) else f"{d:+.3f} s"


def format_clock(t: float) -> str:
    """Lap clock as ``m:ss.mmm``, with a leading ``-`` before the line (a lead-in)."""
    return "-" + format_lap_time(-t) if t < 0.0 else format_lap_time(t)


# ------------------------------------------------------------------ timing for the HUD


def default_self_index(traces: list[Trace]) -> int:
    """The "self" car: the first trace that isn't a real-driver ghost (an AI lap, a run's
    best lap...), else the first trace."""
    for i, tr in enumerate(traces):
        if not tr.is_ghost:
            return i
    return 0


def lap_time_of(p: PreparedTrace, length: float) -> float:
    """Line-to-line lap time (lap clock at ``s = length``), or the trace's duration if it
    doesn't complete the lap."""
    t_end = float(time_at_s(p.s_mono, p.t_mono, length))
    return t_end if math.isfinite(t_end) else float(p.trace.lap_time_s)


class ReplayTiming:
    """Everything the replay HUD shows, computed once from the session (pure numpy): lap
    times, sector times and session bests, speed traces, the reference, the tower rows, and
    the live gap and sectors at a playback position."""

    def __init__(self, session: ReplaySession, length: float, sectors_s, colours: list):
        self.session = session
        self.length = float(length)
        self.colours = colours
        prepared = session.prepared
        self.lap_times = [lap_time_of(p, self.length) for p in prepared]
        self.sectors = [tm.sector_times(p.t, p.s, sectors_s, self.length) for p in prepared]
        self.marks = np.concatenate(
            [[0.0], tm.sector_boundaries(sectors_s, self.length), [self.length]]
        )
        self.session_best = []
        for k in range(self.marks.size - 1):
            vals = [sec[k] for sec in self.sectors if math.isfinite(sec[k])]
            self.session_best.append(min(vals) if vals else math.nan)
        self.speed = [tm.resample_speed(p.s, p.trace.speed * 3.6, self.length) for p in prepared]

    def references(self, self_idx: int) -> list[int]:
        """The traces other than ``self_idx``, fastest first (stable on ties)."""
        others = [i for i in range(len(self.lap_times)) if i != self_idx]
        return sorted(others, key=lambda i: self.lap_times[i])

    def reference(self, self_idx: int, pick: int = 0) -> int | None:
        """The ``pick``-th fastest trace other than ``self_idx`` (wrapping; 0 = the fastest),
        or ``None`` with a single trace."""
        others = self.references(self_idx)
        if not others:
            return None
        return others[pick % len(others)]

    def tower(self, self_idx: int) -> list[tm.TowerRow]:
        entries = [
            tm.TowerEntry(p.trace.label, self.lap_times[i], self.colours[i], i == self_idx)
            for i, p in enumerate(self.session.prepared)
        ]
        return tm.tower_rows(entries)

    def gap(self, self_idx: int, ref_idx: int | None, poses: list[CarPose]) -> tuple[float, float]:
        """``(gap, s)``: self's lap clock minus the reference's at the furthest ``s`` both
        have reached (positive = self behind); nan without a reference."""
        if ref_idx is None:
            return math.nan, math.nan
        a, b = self.session.prepared[self_idx], self.session.prepared[ref_idx]
        s_q = min(poses[self_idx].s, poses[ref_idx].s)
        gap = float(time_at_s(a.s_mono, a.t_mono, s_q) - time_at_s(b.s_mono, b.t_mono, s_q))
        return gap, s_q

    def sectors_at(self, self_idx: int, s_now: float) -> tuple[list[float], list[str]]:
        """Self's sector times revealed up to ``s_now`` and their F1 colours (one lap is its
        own personal best, so each is purple or green)."""
        times, states = [], []
        for k, t in enumerate(self.sectors[self_idx]):
            shown = t if s_now >= self.marks[k + 1] - 1e-6 else math.nan
            times.append(shown)
            states.append(tm.sector_colour(shown, None, self.session_best[k]))
        return times, states


# --------------------------------------------------------------------------- window


def controls(on_back) -> list[str]:
    """The footer's controls; Esc goes back to the menu, or quits when on its own."""
    return [*CONTROLS[:-1], "Esc back" if on_back is not None else "Esc quit"]


CONTROLS = [
    "Space pause",
    "←→ seek",
    "↑↓ speed",
    "Tab pages",
    "F follow",
    "R restart",
    "Esc quit",
]


def ticker_mode(clock: PlaybackClock) -> str:
    """The ticker's last value: ``REPLAY 1×``, ``PAUSED`` or ``END``."""
    if clock.paused:
        return "PAUSED"
    if clock.at_end:
        return "END"
    return f"REPLAY {clock.speed:g}×"


class ReplayView(Screen):
    """The replay screen (an arcade View in the app window): a :class:`ReplaySession` on a
    track, with the HUD. Esc goes back (``on_back``), or quits when it's on its own."""

    def __init__(
        self,
        track: Track,
        traces: list[Trace],
        glass: bool = True,
        on_back=None,
    ):
        super().__init__(on_back)
        self.title = f"f1sim replay - {track.data.name}"
        from f1sim.app.render_track import TrackRenderer
        from f1sim.app.ui.glass import GlassRenderer
        from f1sim.app.ui.hud import Hud
        from f1sim.app.ui.pixelcar import CarLayer

        self.background_color = theme.BACKGROUND
        self.track = track
        self.self_index = default_self_index(traces)
        self.session = ReplaySession(
            [with_track_s(tr, track) for tr in traces], track.length, follow=self.self_index
        )
        prepared = self.session.prepared
        self.colours = theme.trace_colours(
            [(p.trace.meta or {}).get("team") for p in prepared],
            [p.trace.is_ghost for p in prepared],
        )
        self.timing = ReplayTiming(self.session, track.length, track.data.sectors_s, self.colours)

        self.camera = Camera(self.width, self.height)
        walls = np.vstack([track.left_wall(), track.right_wall()])
        self.camera.fit(*walls.min(axis=0), *walls.max(axis=0))
        start = self.session.poses()[self.self_index]
        self.camera.follow(start.x, start.y)
        self._zoom_to(FOLLOW_PIXELS_PER_METRE)
        self.camera2d = arcade.camera.Camera2D()
        self.track_view = TrackRenderer(track, self.ctx)  # uploaded to the GPU once

        theme.load_fonts()
        self.glass = GlassRenderer(self.window, enabled=glass)
        self.hud = Hud(self.glass, len(prepared), track.length, controls(on_back))
        self.track_title = self.track.data.id.replace("_", " ").upper()
        self.hud.headline.set([self.track_title, "Replay"])
        self.cars = CarLayer(
            [p.trace.label for p in prepared],
            [
                theme.livery((p.trace.meta or {}).get("team"), not p.trace.is_ghost)
                for p in prepared
            ],
            [p.trace.is_ghost for p in prepared],
        )
        self._ref_pick = 0
        self._static_for: tuple[int, int] | None = None
        self._ref_index: int | None = None
        self._ref_long: str | None = None
        self._update_static()
        self.hud.layout(self.width, self.height)
        self._follow_camera()

    # ------------------------------------------------------------------------ helpers

    def _zoom_to(self, pixels_per_metre: float) -> None:
        factor = pixels_per_metre / self.camera.pixels_per_metre
        self.camera.zoom_at(factor, 0.5 * self.camera.viewport_w, 0.5 * self.camera.viewport_h)

    def _follow_camera(self) -> None:
        f = self.session.follow
        if f is not None:
            if f != self.self_index:
                self.self_index = f  # the followed car is "self"; the free camera keeps it
                self._ref_pick = 0
            pose = self.session.prepared[f].pose(self.session.clock.time)
            self.camera.follow_at(pose.x, pose.y, *self.hud.stage_centre())

    def cycle_reference(self) -> None:
        """Compare against the next-fastest other car (the gap card's ↗ button)."""
        self._ref_pick += 1

    def _update_static(self) -> None:
        """HUD content that only changes with who is "self" and the chosen reference."""
        me = self.self_index
        n_refs = max(len(self.timing.references(me)), 1)
        self._ref_pick %= n_refs
        ref = self.timing.reference(me, self._ref_pick)
        prepared = self.session.prepared
        self._ref_index = ref
        name = None if ref is None else prepared[ref].trace.label
        self._ref_long = None
        if ref is not None:
            self._ref_long = theme.ghost_long_name(prepared[ref].trace.meta) or name
        self.hud.card.set_reference(name)
        self.hud.card.set_speed_traces(
            None if ref is None else self.timing.speed[ref], self.timing.speed[me]
        )
        self.hud.tower.set("The field", "Lap time", self.timing.tower(me))
        self._static_for = (me, self._ref_pick)

    # ------------------------------------------------------------------------- events

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        clock = self.session.clock
        if symbol == arcade.key.ESCAPE:
            self.close()
        elif symbol == arcade.key.SPACE:
            clock.toggle_pause()
        elif symbol == arcade.key.LEFT:
            clock.seek(-SEEK_STEP_S)
        elif symbol == arcade.key.RIGHT:
            clock.seek(SEEK_STEP_S)
        elif symbol == arcade.key.UP:
            clock.faster()
        elif symbol == arcade.key.DOWN:
            clock.slower()
        elif symbol == arcade.key.R:
            clock.restart()
        elif symbol == arcade.key.F:
            self.session.cycle_follow()
        elif symbol == arcade.key.TAB:
            self.hud.on_tab(time.perf_counter())
        self._follow_camera()

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if self.hud.on_scroll(x, y, scroll_y, time.perf_counter()):
            return  # the wheel pages the card
        if scroll_y:
            self.camera.zoom_at(ZOOM_STEP**scroll_y, x, y)  # follow re-centres on the car

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            if self.hud.on_press(x, y, time.perf_counter()) == "gap":
                self.cycle_reference()

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.session.follow = None
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self.hud.layout(width, height)
        self._follow_camera()

    def on_update(self, delta_time: float) -> None:
        self.session.update(delta_time)
        self._follow_camera()

    # --------------------------------------------------------------------------- draw

    def on_draw(self) -> None:
        if self._static_for != (self.self_index, self._ref_pick):
            self._update_static()
        now = time.perf_counter()
        poses = self.session.poses()
        self._update_hud(poses, now)
        with self.glass.scene():
            self.camera2d.position = self.camera.centre
            self.camera2d.zoom = self.camera.pixels_per_metre
            self.glass.use_camera(self.camera2d)
            self.track_view.draw(self.camera.pixels_per_metre)
            self.glass.use_pixels()
            self.track_view.draw_ticks(self.camera, self.glass)
            self.cars.draw(
                self.camera,
                [(p.x, p.y, p.heading) for p in poses],
                self.self_index,
                self.get_pixel_ratio(),
            )
        self.glass.present()
        self.hud.draw(now)

    def _update_hud(self, poses: list[CarPose], now: float) -> None:
        from f1sim.app.ui.hud import ticker_date_time

        me = self.self_index
        gap, s_q = self.timing.gap(me, self._ref_index, poses)
        self.hud.gap.set(gap, self._ref_long)
        self.hud.card.set_gap(gap, f"at {s_q:,.0f} m" if math.isfinite(s_q) else "")
        self.hud.card.set_marker(poses[me].s)
        self.hud.card.set_sectors(*self.timing.sectors_at(me, poses[me].s))
        prepared = self.session.prepared[me]
        position = f"P{tm.live_positions([p.s for p in poses])[me]}"
        if prepared.trace.gear.size == prepared.t.size:
            gear = float(np.interp(poses[me].lap_clock, prepared.t, prepared.trace.gear))
            extra = f"G{int(round(gear))} · {position}"
        else:
            extra = position
        clock = self.session.clock
        self.hud.speed.set(poses[me].speed * 3.6, extra, format_clock(clock.time))
        self.hud.ticker.set([*ticker_date_time(), self.track_title, ticker_mode(clock)], now)
        if self.session.follow is None:
            self.hud.footer.set_status("Free camera", theme.TEXT_SECONDARY)
        else:
            self.hud.footer.set_status(f"Following {prepared.trace.label}")


def run(track_id: str, traces: list[Path], labels: list[str] | None = None) -> None:
    """Open the replay viewer for ``traces`` on a built track; returns when the window closes.

    Prints a message and returns (without opening a window) if the track isn't built, a trace
    file is missing or unreadable, or ``labels`` doesn't match ``traces``.
    """
    from f1sim.data.track_pipeline import load_track

    paths = [Path(p) for p in traces]
    if not paths:
        print("No trace files given.")
        return
    if labels is not None and len(labels) != len(paths):
        print(f"Got {len(labels)} label(s) for {len(paths)} trace file(s).")
        return
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("Trace file(s) not found:\n" + "\n".join(f"    {m}" for m in missing))
        return
    try:
        track = load_track(track_id)
    except FileNotFoundError:
        print(
            f"Track '{track_id}' hasn't been built yet. Build it first with:\n"
            f"    f1sim build-track {track_id}"
        )
        return
    try:
        loaded = [load_trace(p, None if labels is None else labels[i]) for i, p in enumerate(paths)]
    except (ValueError, OSError) as exc:
        print(f"Can't read trace: {exc}")
        return
    shell.run(lambda window: ReplayView(track, loaded))
