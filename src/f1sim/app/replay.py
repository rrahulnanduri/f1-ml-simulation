"""Replay viewer (M3-L): play back one or more recorded laps on a built track.

``run(track_id, traces, labels)`` opens a resizable arcade window. Traces are NPZ files with at
least ``t``, ``x``, ``y`` (optionally ``s`` and ``speed``): ghost laps (data-formats section 3),
a run's ``best_lap.npz`` (section 5) or anything else with those columns.

The testable logic is pure numpy/Python, without arcade objects:

- :func:`load_trace` reads a file into a :class:`Trace`.
- :func:`project_s` / :func:`unwrap_s` give a trace a monotonic arc length.
- :func:`smooth_heading` computes the heading once per trace (central differences, then a
  circular moving average over ~0.2 s), and :func:`interp_position` / :func:`interp_heading`
  interpolate between samples.
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

**Window.** The track is the cached ``ShapeElementList`` from :mod:`f1sim.app.render_track`,
drawn under an ``arcade.Camera2D`` synced from :class:`~f1sim.app.camera.Camera` each frame,
exactly as in :mod:`f1sim.app.drive`. Cars are 5.6 x 2.0 m rectangles under the same camera;
labels, legend and HUD are ``arcade.Text`` objects under the default (pixel) camera.

Controls: Space pause, Left/Right seek -/+2 s, Up/Down playback speed (0.25x to 8x), ``R``
restart, ``F`` cycle the followed car (then free camera), mouse wheel zoom at the cursor,
right-drag pan (turns follow off), ``Esc`` quit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import arcade
import numpy as np

from f1sim.app.camera import Camera
from f1sim.app.drive import car_polygon, format_lap_time
from f1sim.app.render_track import build_track_shapes
from f1sim.track.track import Track

DT = 1.0 / 60.0
"""Window update and draw interval, s."""
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

CAR_LENGTH_M = 5.6
CAR_WIDTH_M = 2.0
FOLLOW_PIXELS_PER_METRE = 4.0
"""Initial zoom of the follow camera."""
ZOOM_STEP = 1.15
"""Zoom factor per mouse-wheel notch."""

CAR_COLOURS = (
    (255, 140, 0, 255),  # orange
    (0, 200, 255, 255),  # cyan
    (240, 60, 200, 255),  # magenta
    (250, 230, 40, 255),  # yellow
    (120, 255, 120, 255),  # light green
    (255, 255, 255, 255),  # white
)
CAR_NOSE_COLOUR = (20, 20, 20, 255)
BACKGROUND_COLOUR = (24, 70, 32, 255)
PANEL_COLOUR = (0, 0, 0, 150)
AHEAD_COLOUR = (80, 230, 80, 255)
BEHIND_COLOUR = (255, 80, 80, 255)


# ---------------------------------------------------------------------------- traces


@dataclass(frozen=True)
class Trace:
    """One recorded lap: time ``t`` (s), position ``x``, ``y`` (m, track frame), arc length
    ``s`` (m) and ``speed`` (m/s), all 1-D of the same length.

    ``s`` is an **empty array** when the file had none; :func:`with_track_s` (called by
    :func:`run`) fills it by projecting onto the track. ``lap_time_s = t[-1] - t[0]``.
    """

    label: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    speed: np.ndarray
    lap_time_s: float

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


def load_trace(path: Path, label: str | None = None) -> Trace:
    """Read a trace NPZ with at least ``t``, ``x``, ``y``; ``s`` and ``speed`` are optional.

    A missing ``speed`` is computed from finite differences of the positions; a missing ``s``
    is left empty (see :class:`Trace`). Raises ``ValueError`` for missing columns, mismatched
    lengths, fewer than two samples or a ``t`` that isn't strictly increasing.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [c for c in ("t", "x", "y") if c not in data.files]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        t, x, y = (_column(data, c, path) for c in ("t", "x", "y"))
        s = _column(data, "s", path) if "s" in data.files else np.empty(0)
        speed = _column(data, "speed", path) if "speed" in data.files else None
    n = t.size
    for name, arr in (("x", x), ("y", y), ("s", s), ("speed", speed)):
        if arr is None or (name == "s" and arr.size == 0):
            continue  # optional column absent
        if arr.size != n:
            raise ValueError(f"{path}: column '{name}' has {arr.size} samples, expected {n}")
    if n < 2:
        raise ValueError(f"{path}: a trace needs at least 2 samples, got {n}")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{path}: 't' must be strictly increasing")
    if speed is None:
        speed = np.hypot(np.gradient(x, t), np.gradient(y, t))
    return Trace(
        label=label if label is not None else _default_label(path),
        t=t,
        x=x,
        y=y,
        s=s,
        speed=speed,
        lap_time_s=float(t[-1] - t[0]),
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
    all computed once."""

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

    def pose(self, lap_clock: float) -> CarPose:
        tc = min(max(lap_clock, float(self.t[0])), float(self.t[-1]))
        x, y = interp_position(self.t, self.trace.x, self.trace.y, tc)
        return CarPose(
            x=float(x),
            y=float(y),
            heading=float(interp_heading(self.t, self.heading, tc)),
            speed=float(np.interp(tc, self.t, self.trace.speed)),
            s=float(np.interp(tc, self.t, self.s)),
            lap_clock=tc,
        )


class ReplaySession:
    """All traces on one :class:`PlaybackClock` (lap-clock time), plus the follow target.
    Pure Python/numpy, no window."""

    def __init__(self, traces: list[Trace], length: float):
        if not traces:
            raise ValueError("need at least one trace")
        self.prepared = [PreparedTrace(tr, length) for tr in traces]
        start = min(float(p.t[0]) for p in self.prepared)
        end = max(float(p.t[-1]) for p in self.prepared)
        self.clock = PlaybackClock(start, end)
        self.follow: int | None = 0

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


# --------------------------------------------------------------------------- window


class ReplayWindow(arcade.Window):
    """Resizable window playing a :class:`ReplaySession` on a track."""

    def __init__(self, track: Track, traces: list[Trace], width: int = 1280, height: int = 800):
        super().__init__(
            width,
            height,
            f"f1sim replay - {track.data.name}",
            resizable=True,
            update_rate=DT,
            draw_rate=DT,
        )
        self.background_color = BACKGROUND_COLOUR
        self.session = ReplaySession([with_track_s(tr, track) for tr in traces], track.length)
        n = len(self.session.prepared)
        self.colours = [CAR_COLOURS[i % len(CAR_COLOURS)] for i in range(n)]

        self.camera = Camera(self.width, self.height)
        walls = np.vstack([track.left_wall(), track.right_wall()])
        self.camera.fit(*walls.min(axis=0), *walls.max(axis=0))
        first = self.session.poses()[0]
        self.camera.follow(first.x, first.y)
        self._zoom_to(FOLLOW_PIXELS_PER_METRE)
        self.camera2d = arcade.camera.Camera2D()
        self.track_shapes = build_track_shapes(track)  # uploaded to the GPU once

        white = arcade.color.WHITE
        self.text_rows = [arcade.Text("", 0, 0, c, 14, bold=True) for c in self.colours]
        self.text_car_labels = [
            arcade.Text(lab, 0, 0, c, 11, bold=True, anchor_y="bottom")
            for lab, c in zip(self.session.labels, self.colours, strict=True)
        ]
        self.text_delta = arcade.Text("", 0, 0, white, 18, bold=True)
        self.text_clock = arcade.Text("", 0, 0, white, 18, bold=True, anchor_x="right")
        self.text_speed = arcade.Text("", 0, 0, white, 14, anchor_x="right")
        self.text_help = arcade.Text(
            "Space pause   Left/Right seek 2 s   Up/Down speed   R restart   F follow   "
            "wheel zoom   right-drag pan   Esc quit",
            10,
            10,
            arcade.color.LIGHT_GRAY,
            11,
        )
        self.text_follow = arcade.Text("", 0, 0, arcade.color.LIGHT_GRAY, 11, anchor_x="right")
        self._layout_hud()

    # ------------------------------------------------------------------------ helpers

    def _zoom_to(self, pixels_per_metre: float) -> None:
        factor = pixels_per_metre / self.camera.pixels_per_metre
        self.camera.zoom_at(factor, 0.5 * self.camera.viewport_w, 0.5 * self.camera.viewport_h)

    def _layout_hud(self) -> None:
        w, h = self.width, self.height
        for i, text in enumerate(self.text_rows):
            text.position = (34, h - 30 - 24 * i)
        self.text_delta.position = (14, h - 40 - 24 * len(self.text_rows))
        self.text_clock.position = (w - 14, h - 32)
        self.text_speed.position = (w - 14, h - 56)
        self.text_follow.position = (w - 14, 10)

    @staticmethod
    def _set_text(text: arcade.Text, value: str) -> None:
        if text.text != value:  # only re-layout when the string changes
            text.text = value

    def _follow_camera(self) -> None:
        f = self.session.follow
        if f is not None:
            pose = self.session.prepared[f].pose(self.session.clock.time)
            self.camera.follow(pose.x, pose.y)

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
        self._follow_camera()

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if scroll_y:
            self.camera.zoom_at(ZOOM_STEP**scroll_y, x, y)  # follow re-centres on the car

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.session.follow = None
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self._layout_hud()

    def on_update(self, delta_time: float) -> None:
        self.session.update(delta_time)
        self._follow_camera()

    def on_draw(self) -> None:
        self.clear()
        poses = self.session.poses()
        self.camera2d.position = self.camera.centre
        self.camera2d.zoom = self.camera.pixels_per_metre
        with self.camera2d.activate():
            self.track_shapes.draw()
            # Followed car last, so it's on top.
            order = [i for i in range(len(poses)) if i != self.session.follow]
            order += [] if self.session.follow is None else [self.session.follow]
            for i in order:
                self._draw_car(poses[i], self.colours[i])
        self.default_camera.use()
        self._draw_labels(poses)
        self._draw_hud(poses)

    def _draw_car(self, pose: CarPose, colour) -> None:
        body = car_polygon(pose.x, pose.y, pose.heading, CAR_LENGTH_M, CAR_WIDTH_M)
        arcade.draw_polygon_filled([tuple(p) for p in body], colour)
        c, s = np.cos(pose.heading), np.sin(pose.heading)
        nose = car_polygon(
            pose.x + 0.35 * CAR_LENGTH_M * c,
            pose.y + 0.35 * CAR_LENGTH_M * s,
            pose.heading,
            0.3 * CAR_LENGTH_M,
            0.6 * CAR_WIDTH_M,
        )
        arcade.draw_polygon_filled([tuple(p) for p in nose], CAR_NOSE_COLOUR)

    def _draw_labels(self, poses: list[CarPose]) -> None:
        """Small label next to each car, in screen pixels so it doesn't scale with zoom."""
        lift = 0.5 * CAR_LENGTH_M * self.camera.pixels_per_metre + 6.0
        for pose, text in zip(poses, self.text_car_labels, strict=True):
            sx, sy = self.camera.world_to_screen(pose.x, pose.y)
            text.position = (float(sx) + 8.0, float(sy) + lift)
            text.draw()

    def _draw_hud(self, poses: list[CarPose]) -> None:
        h = self.height
        n = len(poses)
        rows_h = 24 * n + (34 if n >= 2 else 0)
        arcade.draw_lbwh_rectangle_filled(6, h - 18 - rows_h, 470, rows_h + 12, PANEL_COLOUR)
        for i, (pose, text) in enumerate(zip(poses, self.text_rows, strict=True)):
            tr = self.session.prepared[i].trace
            arcade.draw_lbwh_rectangle_filled(14, h - 31 - 24 * i, 12, 12, self.colours[i])
            self._set_text(
                text,
                f"{tr.label[:16]:<16}  {pose.speed * 3.6:5.0f} km/h  "
                f"{format_clock(pose.lap_clock)}  ({format_lap_time(tr.lap_time_s)})",
            )
            text.draw()
        if n >= 2:
            d = self.session.delta(poses)
            self.text_delta.color = AHEAD_COLOUR if d < 0.0 else BEHIND_COLOUR
            labels = self.session.labels
            self._set_text(
                self.text_delta, f"delta {labels[1][:12]} vs {labels[0][:12]}  {format_delta(d)}"
            )
            self.text_delta.draw()
        clock = self.session.clock
        self._set_text(self.text_clock, format_clock(clock.time))
        state = "paused" if clock.paused else ("end" if clock.at_end else "playing")
        self._set_text(self.text_speed, f"{clock.speed:g}x  {state}")
        f = self.session.follow
        self._set_text(
            self.text_follow,
            "follow: off (F)" if f is None else f"follow: {self.session.labels[f]} (F)",
        )
        self.text_clock.draw()
        self.text_speed.draw()
        self.text_follow.draw()
        self.text_help.draw()


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
    ReplayWindow(track, loaded)
    arcade.run()
