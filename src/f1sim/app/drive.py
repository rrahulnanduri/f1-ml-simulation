"""Minimal keyboard drive viewer (M2): feel the car model on a built track.

``run(track_id)`` opens a resizable arcade window. The game logic lives in :class:`DriveSession`
(no arcade, testable offline); :class:`DriveView` (a screen of the app window,
:mod:`f1sim.app.shell`) only maps input to it and draws.

Controls: Up = throttle, Down = brake (both ramped over ``PEDAL_RAMP_S``), Left/Right = steer
(ramped over ``STEER_RAMP_S``), ``R`` reset to the start line, ``F`` toggle the follow camera,
``Tab`` next card page, mouse wheel = zoom at the cursor (pages the card when over it), click
the card = next page, right-drag = pan (turns follow off), ``Esc`` = back to the menu (quit
when opened on its own with ``f1sim drive``).

Rendering: the track is cached once in world metres
(:class:`~f1sim.app.render_track.TrackRenderer`: asphalt, kerbs, run-off, chequered strip,
white lines), drawn under an arcade ``Camera2D`` whose position and zoom are copied from
:class:`~f1sim.app.camera.Camera` every frame, into the offscreen scene of
:mod:`f1sim.app.ui.glass`. The car is a pixel-art sprite (:mod:`f1sim.app.ui.pixelcar`) in
titanium, black and red, kept at the middle of the area the HUD leaves free. The HUD
(:mod:`f1sim.app.ui.hud`) uses ``arcade.Text`` objects created once.

**Smooth motion.** Physics runs in fixed ``DT`` steps, but frames come at their own rate
(vsync, or a 120 Hz tick where the driver ignores vsync), never exactly in step with the
physics. A frame can therefore land after zero or two physics steps, and drawing the raw
state makes the car and the track stutter. The session keeps the previous step's pose, and
the window draws (and follows) :meth:`DriveSession.render_pose`, interpolated by the leftover
accumulator time. The ghost is placed at :meth:`DriveSession.render_lap_time` the same way.

**Ghost.** With ``ghost_path`` (``f1sim drive <track> --ghost lap.npz``) a ghost car (in its
team livery, with a light outline) drives its lap from the moment the lap clock starts,
restarting with every lap; the gap readout, the speed page and the sector colours compare the
live lap against it.
:class:`DriveSession` records the live lap in a :class:`~f1sim.app.ui.timing.LapTelemetry`
(sector splits, speed along the lap).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import arcade
import numpy as np

from f1sim.app import shell
from f1sim.app.camera import Camera
from f1sim.app.shell import Screen
from f1sim.app.ui import theme
from f1sim.app.ui import timing as tm
from f1sim.sim.car_model import CarParams, CarState, StepInfo, step
from f1sim.track.track import Track

DT = 1.0 / 60.0
"""Fixed physics step, s."""
MAX_FRAME_S = 0.25
"""Longest frame time fed to the accumulator (avoids a spiral of death after a stall)."""
PEDAL_RAMP_S = 0.15
"""Time for throttle or brake to go from 0 to full (and back)."""
STEER_RAMP_S = 0.15
"""Time for the steering command to go from centre to full lock (and back)."""
OFF_TRACK_MARGIN_M = 1.0
"""Off track when ``|n| > w_side - OFF_TRACK_MARGIN_M`` (car half width, as in progress.py)."""
OFF_TRACK_RESET_S = 1.0
"""How long "OFF TRACK" shows (with the car stopped) before the reset to the start line."""
LAP_MIN_FRACTION = 0.98
"""A forward crossing of s = 0 counts as a lap after this share of the length."""
REPROJECT_JUMP_M = 50.0
"""A hinted projection jumping further than this along the track is redone globally."""
PROJECT_WINDOW = 10

FOLLOW_PIXELS_PER_METRE = 4.0
"""Initial zoom of the follow camera."""
ZOOM_STEP = 1.15
"""Zoom factor per mouse-wheel notch."""


def controls(on_back) -> list[str]:
    """The footer's controls; Esc goes back to the menu, or quits when on its own."""
    return [*CONTROLS[:-1], "Esc back" if on_back is not None else "Esc quit"]


CONTROLS = [
    "↑ throttle",
    "↓ brake",
    "←→ steer",
    "Tab pages",
    "F follow",
    "R reset",
    "Esc quit",
]


@dataclass
class Controls:
    """Which driving keys are held."""

    throttle: bool = False
    brake: bool = False
    left: bool = False
    right: bool = False


def _ramp(value: float, target: float, ramp_s: float, dt: float) -> float:
    """Move ``value`` towards ``target`` at a rate of 1 per ``ramp_s`` seconds."""
    max_change = dt / ramp_s
    return value + float(np.clip(target - value, -max_change, max_change))


def car_polygon(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners ``(4, 2)`` of a ``length x width`` rectangle centred on ``(x, y)``, rotated by
    ``yaw``: rear-right, front-right, front-left, rear-left (counter-clockwise)."""
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def format_lap_time(t: float | None) -> str:
    """``m:ss.mmm``, or ``-:--.---`` for ``None``/nan."""
    if t is None or not np.isfinite(t):
        return "-:--.---"
    minutes, seconds = divmod(float(t), 60.0)
    return f"{int(minutes)}:{seconds:06.3f}"


class DriveSession:
    """One keyboard-driven car on a track: ramped inputs, fixed-step physics, off-track reset
    and lap timing. Pure Python/numpy, no window."""

    def __init__(self, track: Track, params: CarParams | None = None):
        self.track = track
        self.params = params or CarParams()
        self._accumulator = 0.0
        self.telemetry = tm.LapTelemetry(track.length, track.data.sectors_s)
        self.best_lap: float | None = None
        self.last_lap: float | None = None
        self.laps = 0
        self.reset()

    # -------------------------------------------------------------------------- state

    def reset(self) -> None:
        """Car at ``s = 0`` heading along the track, standing still; current lap discarded."""
        p = self.track.point_at(0.0)
        yaw = float(self.track.heading_at(0.0))
        self.state = CarState.at(1, float(p[0]), float(p[1]), yaw)
        self._prev_pose = (float(p[0]), float(p[1]), yaw)
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.info = StepInfo(np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool))
        s, n, idx = self.track.project(self.state.x, self.state.y)
        self.s, self.n, self.idx = float(s[0]), float(n[0]), int(idx[0])
        self.lap_time = 0.0
        self.lap_progress = 0.0  # distance along the track since the current lap started
        self.clock_running = False  # the lap clock starts when the car first moves
        self.off_track_timer = 0.0  # > 0 while "OFF TRACK" is showing
        self.telemetry.discard_lap()

    @property
    def x(self) -> float:
        return float(self.state.x[0])

    @property
    def y(self) -> float:
        return float(self.state.y[0])

    @property
    def yaw(self) -> float:
        return float(self.state.yaw[0])

    @property
    def speed_kmh(self) -> float:
        return float(self.state.v[0]) * 3.6

    @property
    def sliding(self) -> bool:
        return bool(self.info.sliding[0])

    @property
    def off_track(self) -> bool:
        return self.off_track_timer > 0.0

    @property
    def alpha(self) -> float:
        """Share of the next physics step already elapsed, in ``[0, 1)``."""
        return min(max(self._accumulator / DT, 0.0), 1.0)

    def render_pose(self) -> tuple[float, float, float]:
        """``(x, y, yaw)`` to draw: between the last two physics steps by :attr:`alpha`
        (``yaw`` the shorter way round)."""
        a = self.alpha
        x0, y0, yaw0 = self._prev_pose
        dyaw = _wrap(self.yaw - yaw0, 2.0 * math.pi)
        return x0 + a * (self.x - x0), y0 + a * (self.y - y0), yaw0 + a * dyaw

    def render_lap_time(self) -> float:
        """The lap clock at the drawn moment (0 until the clock starts)."""
        if not self.clock_running:
            return 0.0
        return self.lap_time + (0.0 if self.off_track else self.alpha * DT)

    # ------------------------------------------------------------------------- update

    def update(self, controls: Controls, frame_dt: float) -> int:
        """Advance by a frame of ``frame_dt`` seconds in fixed ``DT`` steps; returns the
        number of physics steps taken."""
        self._accumulator += min(max(frame_dt, 0.0), MAX_FRAME_S)
        steps = 0
        while self._accumulator >= DT - 1e-9:  # tolerance: frame times summing to DT count
            self._accumulator -= DT
            self.fixed_step(controls)
            steps += 1
        return steps

    def fixed_step(self, controls: Controls) -> None:
        """One physics step of ``DT``."""
        self._prev_pose = (self.x, self.y, self.yaw)
        if self.off_track:
            self.off_track_timer -= DT
            if self.off_track_timer <= 1e-9:
                self.reset()  # also resets the previous pose: no sweep back to the line
            return
        self.throttle = _ramp(self.throttle, float(controls.throttle), PEDAL_RAMP_S, DT)
        self.brake = _ramp(self.brake, float(controls.brake), PEDAL_RAMP_S, DT)
        steer_target = float(controls.left) - float(controls.right)  # + = left
        self.steer = _ramp(self.steer, steer_target, STEER_RAMP_S, DT)
        self.info = step(self.state, self.steer, self.throttle - self.brake, self.params, DT)
        if not self.clock_running and self.state.v[0] > 0.0:
            self.clock_running = True
        self.update_progress(DT)

    def update_progress(self, dt: float) -> None:
        """Project the car, time laps and check the off-track rule after a step of ``dt``."""
        track, length = self.track, self.track.length
        s, n, idx = track.project(self.state.x, self.state.y, hint=self.idx, window=PROJECT_WINDOW)
        ds = _wrap(float(s[0]) - self.s, length)
        if abs(ds) > REPROJECT_JUMP_M:
            s, n, idx = track.project(self.state.x, self.state.y)
            ds = _wrap(float(s[0]) - self.s, length)
        s_prev = self.s
        self.s, self.n, self.idx = float(s[0]), float(n[0]), int(idx[0])

        if self.clock_running:
            if ds > 0.0 and s_prev + ds >= length:  # forward crossing of s = 0
                frac = (length - s_prev) / ds
                progress_at_line = self.lap_progress + (length - s_prev)
                if progress_at_line >= LAP_MIN_FRACTION * length:
                    lap = self.lap_time + frac * dt
                    self.telemetry.complete_lap(lap)
                    self.last_lap = lap
                    self.best_lap = lap if self.best_lap is None else min(self.best_lap, lap)
                    self.laps += 1
                    self.lap_time = -frac * dt  # the rest of this step is the new lap
                    self.lap_progress = -(length - s_prev)
            self.lap_time += dt
        self.lap_progress += ds
        if self.clock_running:
            self.telemetry.record(self.lap_progress, self.lap_time, self.speed_kmh)

        w_left, w_right = track.widths_at(self.s)
        w_side = float(w_left) if self.n >= 0.0 else float(w_right)
        if abs(self.n) > w_side - OFF_TRACK_MARGIN_M:
            self.off_track_timer = OFF_TRACK_RESET_S
            self.state.v[:] = 0.0
            self.info = StepInfo(np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool))


def _wrap(ds: float, length: float) -> float:
    """Wrap a difference of arc lengths to ``[-length/2, length/2)``."""
    return (ds + 0.5 * length) % length - 0.5 * length


@dataclass
class DriveGhost:
    """A reference lap for the drive viewer, prepared once: its lap-clock poses, lap time,
    sector times and speed along the lap. Pure numpy."""

    prepared: object  # f1sim.app.replay.PreparedTrace
    label: str
    long_label: str
    colour: tuple
    lap_time: float
    sectors: list[float]
    speed: tuple[np.ndarray, np.ndarray]
    team: str | None = None

    @classmethod
    def load(cls, path, track: Track) -> DriveGhost:
        """Read a trace NPZ (ghost sidecar optional) and prepare it on ``track``."""
        from f1sim.app import replay  # replay imports this module: import lazily

        trace = replay.with_track_s(replay.load_trace(path), track)
        prepared = replay.PreparedTrace(trace, track.length)
        meta = trace.meta
        colour = theme.team_colour((meta or {}).get("team")) or theme.MUTED_PALETTE[0]
        return cls(
            prepared=prepared,
            label=trace.label,
            long_label=theme.ghost_long_name(meta) or trace.label,
            colour=colour,
            lap_time=replay.lap_time_of(prepared, track.length),
            sectors=tm.sector_times(prepared.t, prepared.s, track.data.sectors_s, track.length),
            speed=tm.resample_speed(prepared.s, trace.speed * 3.6, track.length),
            team=(meta or {}).get("team"),
        )

    def gap(self, lap_time: float, progress: float) -> float:
        """Live lap clock minus the ghost's at the same distance (positive = behind)."""
        p = self.prepared
        return float(lap_time - tm.time_at(p.s_mono, p.t_mono, progress))


def drive_sector_display(
    session: DriveSession, ghost: DriveGhost | None
) -> tuple[list[float], list[str]]:
    """Sector times to show and their F1 colours: purple = best of the session (the ghost
    and all your earlier laps), green = your personal best, yellow = slower."""
    times, pb_before = session.telemetry.displayed()
    states = []
    for k, t in enumerate(times):
        cands = [
            v for v in (pb_before[k], ghost.sectors[k] if ghost else math.nan) if math.isfinite(v)
        ]
        states.append(tm.sector_colour(t, pb_before[k], min(cands) if cands else None))
    return times, states


def drive_tower(session: DriveSession, ghost: DriveGhost | None) -> list[tm.TowerRow]:
    """The ghost and "You" (best lap) ranked, then your last lap and the running lap."""
    entries = [tm.TowerEntry("You", session.best_lap, theme.SELF, True)]
    if ghost is not None:
        entries.insert(0, tm.TowerEntry(ghost.label, ghost.lap_time, ghost.colour))
    rows = tm.tower_rows(entries)
    muted = theme.white(0.28)
    rows.append(tm.TowerRow("", "Last lap", tm.format_lap(session.last_lap), muted, False))
    running = session.lap_time if session.clock_running else None
    rows.append(tm.TowerRow("", "This lap", tm.format_lap(running), muted, False))
    return rows


class DriveView(Screen):
    """The drive screen (an arcade View in the app window): one :class:`DriveSession`, with
    the HUD. Esc goes back (``on_back``), or quits when the screen is on its own."""

    def __init__(self, track: Track, ghost_path=None, glass: bool = True, on_back=None):
        # Read the ghost before building anything, so a bad file shows nothing.
        ghost = DriveGhost.load(ghost_path, track) if ghost_path is not None else None
        super().__init__(on_back)
        self.title = f"f1sim drive - {track.data.name}"
        from f1sim.app.render_track import TrackRenderer
        from f1sim.app.ui.glass import GlassRenderer
        from f1sim.app.ui.hud import Hud
        from f1sim.app.ui.pixelcar import CarLayer

        self.background_color = theme.BACKGROUND
        self.track = track
        self.session = DriveSession(track)
        self.controls = Controls()
        self.follow = True
        self.ghost = ghost

        self.camera = Camera(self.width, self.height)
        walls = np.vstack([track.left_wall(), track.right_wall()])
        self.track_bbox = (*walls.min(axis=0), *walls.max(axis=0))
        self.camera.fit(*self.track_bbox)
        self.camera.follow(self.session.x, self.session.y)
        self._zoom_to(FOLLOW_PIXELS_PER_METRE)
        self.camera2d = arcade.camera.Camera2D()
        self.track_view = TrackRenderer(track, self.ctx)  # uploaded to the GPU once

        theme.load_fonts()
        self.glass = GlassRenderer(self.window, enabled=glass)
        self.hud = Hud(self.glass, 4, track.length, controls(on_back))
        self.track_title = self.track.data.id.replace("_", " ").upper()
        self.hud.headline.set([self.track_title, "Drive"])
        ghost_name = None if self.ghost is None else self.ghost.label
        self.hud.card.set_reference(ghost_name)
        labels, liveries, outlined = ["You"], [theme.SELF_LIVERY], [False]
        if self.ghost is not None:
            labels.insert(0, self.ghost.label)
            liveries.insert(0, theme.livery(self.ghost.team))
            outlined.insert(0, True)
        self.cars = CarLayer(labels, liveries, outlined)
        self._speed_version = -1
        self.hud.layout(self.width, self.height)
        self._follow_camera()

    # ------------------------------------------------------------------------ helpers

    def _zoom_to(self, pixels_per_metre: float) -> None:
        factor = pixels_per_metre / self.camera.pixels_per_metre
        self.camera.zoom_at(factor, 0.5 * self.camera.viewport_w, 0.5 * self.camera.viewport_h)

    def _follow_camera(self) -> None:
        if self.follow:
            x, y, _ = self.session.render_pose()
            self.camera.follow_at(x, y, *self.hud.stage_centre())

    def ghost_pose(self):
        """The ghost at the live lap clock (waiting on the line until the clock starts)."""
        return self.ghost.prepared.pose(self.session.render_lap_time())

    def live_gap(self) -> float:
        s = self.session
        if self.ghost is None or not s.clock_running:
            return math.nan
        if not 0.0 <= s.lap_progress <= self.track.length:
            return math.nan
        return self.ghost.gap(s.lap_time, s.lap_progress)

    # ------------------------------------------------------------------------- events

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._set_key(symbol, True)
        if symbol == arcade.key.ESCAPE:
            self.close()
        elif symbol == arcade.key.R:
            self.session.reset()
        elif symbol == arcade.key.F:
            self.follow = not self.follow
        elif symbol == arcade.key.TAB:
            self.hud.on_tab(time.perf_counter())

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._set_key(symbol, False)

    def _set_key(self, symbol: int, down: bool) -> None:
        if symbol == arcade.key.UP:
            self.controls.throttle = down
        elif symbol == arcade.key.DOWN:
            self.controls.brake = down
        elif symbol == arcade.key.LEFT:
            self.controls.left = down
        elif symbol == arcade.key.RIGHT:
            self.controls.right = down

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if self.hud.on_scroll(x, y, scroll_y, time.perf_counter()):
            return  # the wheel pages the card
        if scroll_y:
            self.camera.zoom_at(ZOOM_STEP**scroll_y, x, y)  # follow re-centres on the car

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.hud.on_press(x, y, time.perf_counter())  # one ghost: nothing to cycle

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.follow = False
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self.hud.layout(width, height)
        self._follow_camera()

    def on_update(self, delta_time: float) -> None:
        self.session.update(self.controls, delta_time)
        self._follow_camera()

    # --------------------------------------------------------------------------- draw

    def on_draw(self) -> None:
        now = time.perf_counter()
        self._update_hud(now)
        with self.glass.scene():
            self.camera2d.position = self.camera.centre
            self.camera2d.zoom = self.camera.pixels_per_metre
            self.glass.use_camera(self.camera2d)
            self.track_view.draw(self.camera.pixels_per_metre)
            self.glass.use_pixels()
            self.track_view.draw_ticks(self.camera, self.glass)
            self._draw_cars()
        self.glass.present()
        self.hud.draw(now)

    def _draw_cars(self) -> None:
        poses = [self.session.render_pose()]
        if self.ghost is not None:
            g = self.ghost_pose()
            poses.insert(0, (g.x, g.y, g.heading))
        self.cars.draw(self.camera, poses, len(poses) - 1, self.get_pixel_ratio())

    def _update_hud(self, now: float) -> None:
        from f1sim.app.ui.hud import ticker_date_time

        s, hud, ghost = self.session, self.hud, self.ghost
        hud.tower.set("Timing", f"Lap {s.laps + 1}", drive_tower(s, ghost))
        gap = self.live_gap()
        hud.gap.set(gap, None if ghost is None else ghost.long_label)
        at = f"at {s.lap_progress:,.0f} m" if math.isfinite(gap) else ""
        hud.card.set_gap(gap, at)
        hud.card.set_sectors(*drive_sector_display(s, ghost))
        # The live speed trace only changes when a new bin fills: rebuild it then.
        version = int(np.count_nonzero(np.isfinite(s.telemetry.speed)))
        if version != self._speed_version:
            self._speed_version = version
            hud.card.set_speed_traces(
                None if ghost is None else ghost.speed, (s.telemetry.grid, s.telemetry.speed)
            )
        hud.card.set_marker(s.lap_progress if s.clock_running else 0.0)
        if ghost is None:
            extra = f"Lap {s.laps + 1}"
        elif math.isfinite(gap):
            extra = "P1" if gap <= 0.0 else "P2"
        else:
            extra = "P–"
        hud.speed.set(s.speed_kmh, extra, format_lap_time(s.lap_time if s.clock_running else 0.0))
        hud.ticker.set([*ticker_date_time(), self.track_title, f"Lap {s.laps + 1}"], now)
        hud.show_banner = s.off_track
        if s.sliding:
            hud.footer.set_status("Sliding", theme.ACCENT)
        elif not self.follow:
            hud.footer.set_status("Free camera · F to follow", theme.TEXT_SECONDARY)
        elif ghost is not None:
            hud.footer.set_status(f"Ghost {ghost.label}", theme.TEXT_SECONDARY)
        else:
            hud.footer.set_status("")


def run(track_id: str, ghost: Path | None = None) -> None:
    """Open the drive viewer on a built track, optionally against a ``ghost`` lap NPZ;
    returns when the window closes."""
    from f1sim.data.track_pipeline import load_track

    if ghost is not None and not Path(ghost).is_file():
        print(f"Ghost file not found: {ghost}")
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
        DriveGhost.load(ghost, track) if ghost is not None else None
    except (ValueError, OSError) as exc:
        print(f"Can't read the ghost: {exc}")
        return
    shell.run(lambda window: DriveView(track, ghost_path=ghost))
