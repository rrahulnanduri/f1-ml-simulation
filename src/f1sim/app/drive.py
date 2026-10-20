"""Minimal keyboard drive viewer (M2): feel the car model on a built track.

``run(track_id)`` opens a resizable arcade window. The game logic lives in :class:`DriveSession`
(no arcade, testable offline); :class:`DriveWindow` only maps input to it and draws.

Controls: Up = throttle, Down = brake (both ramped over ``PEDAL_RAMP_S``), Left/Right = steer
(ramped over ``STEER_RAMP_S``), ``R`` reset to the start line, ``F`` toggle the follow camera,
``Tab`` next card page, mouse wheel = zoom at the cursor (pages the card when over it), click
the card = next page, right-drag = pan (turns follow off), ``Esc`` = quit.

Rendering: the track is one cached ``ShapeElementList`` in world metres
(:mod:`f1sim.app.render_track`), drawn under an arcade ``Camera2D`` whose position and zoom are
copied from :class:`~f1sim.app.camera.Camera` every frame, into the glass renderer's offscreen
scene (:mod:`f1sim.app.ui.glass`). The car is a 5.6 x 2.0 m solid orange capsule drawn in screen
pixels (with a minimum size). The HUD (:mod:`f1sim.app.ui.hud`) uses ``arcade.Text`` objects
created once.

**Ghost.** With ``ghost_path`` (``f1sim drive <track> --ghost lap.npz``) a ghost car (outlined
in its team colour) drives its lap from the moment the lap clock starts, restarting with every
lap; the gap readout, the speed page and the sector colours compare the live lap against it.
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

from f1sim.app.camera import Camera
from f1sim.app.render_track import boundary_lines, build_track_shapes
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

HELP_TEXT = (
    "↑ THROTTLE · ↓ BRAKE · ←→ STEER · R RESET · F FOLLOW · TAB PAGES · WHEEL ZOOM · ESC QUIT"
)


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
        if self.off_track:
            self.off_track_timer -= DT
            if self.off_track_timer <= 1e-9:
                self.reset()
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


class DriveWindow(arcade.Window):
    """Resizable window driving one :class:`DriveSession`, with the glass HUD."""

    def __init__(
        self,
        track: Track,
        width: int = 1280,
        height: int = 800,
        ghost_path=None,
        glass: bool = True,
        visible: bool = True,
    ):
        # Read the ghost before opening the window, so a bad file opens nothing.
        ghost = DriveGhost.load(ghost_path, track) if ghost_path is not None else None
        super().__init__(
            width,
            height,
            f"f1sim drive - {track.data.name}",
            resizable=True,
            update_rate=DT,
            draw_rate=DT,
            visible=visible,
        )
        from f1sim.app.ui.glass import GlassRenderer
        from f1sim.app.ui.hud import Hud

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

        self.track_shapes = build_track_shapes(track)  # uploaded to the GPU once
        self.start_line = boundary_lines(track)["start_finish"]
        self._car_length = self.session.params.length_m
        self._car_width = self.session.params.width_m

        theme.load_fonts()
        self.glass = GlassRenderer(self, enabled=glass)
        self.hud = Hud(self.glass, 4, track.length, HELP_TEXT)
        self.hud.banner.text = "OFF TRACK"
        ghost_name = None if self.ghost is None else self.ghost.label
        self.hud.card.set_reference(ghost_name)
        self.text_ghost = theme.make_text(
            (ghost_name or "").upper(), theme.LABEL_PX, theme.TEXT_LABEL, tracking_em=0.08
        )
        self._speed_version = -1
        self.hud.layout(self.width, self.height)

    # ------------------------------------------------------------------------ helpers

    def _zoom_to(self, pixels_per_metre: float) -> None:
        factor = pixels_per_metre / self.camera.pixels_per_metre
        self.camera.zoom_at(factor, 0.5 * self.camera.viewport_w, 0.5 * self.camera.viewport_h)

    def ghost_pose(self):
        """The ghost at the live lap clock (waiting on the line until the clock starts)."""
        s = self.session
        return self.ghost.prepared.pose(s.lap_time if s.clock_running else 0.0)

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
            self.hud.on_press(x, y, time.perf_counter())

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.follow = False
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self.hud.layout(width, height)

    def on_update(self, delta_time: float) -> None:
        self.session.update(self.controls, delta_time)
        if self.follow:
            self.camera.follow(self.session.x, self.session.y)

    # --------------------------------------------------------------------------- draw

    def on_draw(self) -> None:
        from f1sim.app.ui.glass import draw_start_line

        self._update_hud()
        with self.glass.scene():
            self.camera2d.position = self.camera.centre
            self.camera2d.zoom = self.camera.pixels_per_metre
            self.glass.use_camera(self.camera2d)
            self.track_shapes.draw()
            self.glass.use_pixels()
            draw_start_line(self.start_line, self.camera)
            self._draw_cars()
        self.glass.present()
        self.hud.draw(time.perf_counter())

    def _draw_cars(self) -> None:
        from f1sim.app.ui.layout import capsule_size_px

        length, width = capsule_size_px(
            self.camera.pixels_per_metre,
            self._car_length,
            self._car_width,
            theme.CAR_MIN_LENGTH_PX,
            theme.CAR_MIN_WIDTH_PX,
        )
        if self.ghost is not None:
            g = self.ghost_pose()
            gx, gy = self.camera.world_to_screen(g.x, g.y)
            gx, gy = float(gx), float(gy)
            self.glass.capsule(
                gx, gy, g.heading, length, width, self.ghost.colour, theme.CAR_STROKE_PX
            )
            self.text_ghost.position = (round(gx + 8.0), round(gy + 0.5 * length + 6.0))
            self.text_ghost.draw()
        s = self.session
        sx, sy = self.camera.world_to_screen(s.x, s.y)
        self.glass.capsule(float(sx), float(sy), s.yaw, length, width, theme.SELF)

    def _update_hud(self) -> None:
        s, hud, ghost = self.session, self.hud, self.ghost
        track_name = self.track.data.id.replace("_", " ").upper()
        hud.tower.set(f"{track_name} · LAP {s.laps + 1}", drive_tower(s, ghost))
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
            extra = f"LAP {s.laps + 1}"
        elif math.isfinite(gap):
            extra = "P1" if gap <= 0.0 else "P2"
        else:
            extra = "P–"
        hud.speed.set(s.speed_kmh, extra)
        hud.show_banner = s.off_track
        if s.sliding:
            status, colour = "SLIDING", theme.SECTOR_YELLOW
        elif not self.follow:
            status, colour = "FOLLOW OFF · F", theme.TEXT_MUTED
        else:
            status, colour = "", theme.TEXT_SECONDARY
        if hud.status.text != status:
            hud.status.text = status
            hud.status.color = colour


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
        DriveWindow(track, ghost_path=ghost)
    except (ValueError, OSError) as exc:
        print(f"Can't read the ghost: {exc}")
        return
    arcade.run()
