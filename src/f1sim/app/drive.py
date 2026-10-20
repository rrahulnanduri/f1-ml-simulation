"""Minimal keyboard drive viewer (M2): feel the car model on a built track.

``run(track_id)`` opens a resizable arcade window. The game logic lives in :class:`DriveSession`
(no arcade, testable offline); :class:`DriveWindow` only maps input to it and draws.

Controls: Up = throttle, Down = brake (both ramped over ``PEDAL_RAMP_S``), Left/Right = steer
(ramped over ``STEER_RAMP_S``), ``R`` reset to the start line, ``F`` toggle the follow camera,
mouse wheel = zoom at the cursor, right-drag = pan (turns follow off), ``Esc`` = quit.

Rendering: the track is one cached ``ShapeElementList`` in world metres
(:mod:`f1sim.app.render_track`), drawn under an arcade ``Camera2D`` whose position and zoom are
copied from :class:`~f1sim.app.camera.Camera` every frame. The car is a 5.6 x 2.0 m polygon drawn
under the same camera; the HUD uses ``arcade.Text`` objects created once, under the window's
default (pixel) camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np

from f1sim.app.camera import Camera
from f1sim.app.render_track import build_track_shapes
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

CAR_COLOUR = (255, 140, 0, 255)
CAR_NOSE_COLOUR = (20, 20, 20, 255)
BACKGROUND_COLOUR = (24, 70, 32, 255)


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
                    self.last_lap = lap
                    self.best_lap = lap if self.best_lap is None else min(self.best_lap, lap)
                    self.laps += 1
                    self.lap_time = -frac * dt  # the rest of this step is the new lap
                    self.lap_progress = -(length - s_prev)
            self.lap_time += dt
        self.lap_progress += ds

        w_left, w_right = track.widths_at(self.s)
        w_side = float(w_left) if self.n >= 0.0 else float(w_right)
        if abs(self.n) > w_side - OFF_TRACK_MARGIN_M:
            self.off_track_timer = OFF_TRACK_RESET_S
            self.state.v[:] = 0.0
            self.info = StepInfo(np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool))


def _wrap(ds: float, length: float) -> float:
    """Wrap a difference of arc lengths to ``[-length/2, length/2)``."""
    return (ds + 0.5 * length) % length - 0.5 * length


class DriveWindow(arcade.Window):
    """Resizable window driving one :class:`DriveSession`."""

    def __init__(self, track: Track, width: int = 1280, height: int = 800):
        super().__init__(
            width,
            height,
            f"f1sim drive - {track.data.name}",
            resizable=True,
            update_rate=DT,
            draw_rate=DT,
        )
        self.background_color = BACKGROUND_COLOUR
        self.session = DriveSession(track)
        self.controls = Controls()
        self.follow = True

        self.camera = Camera(self.width, self.height)
        walls = np.vstack([track.left_wall(), track.right_wall()])
        self.track_bbox = (*walls.min(axis=0), *walls.max(axis=0))
        self.camera.fit(*self.track_bbox)
        self.camera.follow(self.session.x, self.session.y)
        self._zoom_to(FOLLOW_PIXELS_PER_METRE)
        self.camera2d = arcade.camera.Camera2D()

        self.track_shapes = build_track_shapes(track)  # uploaded to the GPU once
        self._car_length = self.session.params.length_m
        self._car_width = self.session.params.width_m

        white = arcade.color.WHITE
        self.text_lap = arcade.Text("", 0, 0, white, 18, bold=True)
        self.text_last = arcade.Text("", 0, 0, white, 14)
        self.text_best = arcade.Text("", 0, 0, white, 14)
        self.text_speed = arcade.Text("", 0, 0, white, 22, bold=True, anchor_x="right")
        self.text_sliding = arcade.Text(
            "SLIDING", 0, 0, arcade.color.YELLOW, 16, bold=True, anchor_x="right"
        )
        self.text_off = arcade.Text(
            "OFF TRACK", 0, 0, arcade.color.RED, 40, bold=True, anchor_x="center", anchor_y="center"
        )
        self.text_help = arcade.Text(
            "Up/Down throttle/brake   Left/Right steer   R reset   F follow   "
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
        self.text_lap.position = (14, h - 32)
        self.text_last.position = (14, h - 58)
        self.text_best.position = (14, h - 80)
        self.text_speed.position = (w - 14, h - 36)
        self.text_sliding.position = (w - 14, h - 62)
        self.text_follow.position = (w - 14, 10)
        self.text_off.position = (w / 2, h / 2)

    @staticmethod
    def _set_text(text: arcade.Text, value: str) -> None:
        if text.text != value:  # only re-layout when the string changes
            text.text = value

    # ------------------------------------------------------------------------- events

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._set_key(symbol, True)
        if symbol == arcade.key.ESCAPE:
            self.close()
        elif symbol == arcade.key.R:
            self.session.reset()
        elif symbol == arcade.key.F:
            self.follow = not self.follow

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
        if scroll_y:
            self.camera.zoom_at(ZOOM_STEP**scroll_y, x, y)  # follow re-centres on the car

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.follow = False
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self._layout_hud()

    def on_update(self, delta_time: float) -> None:
        self.session.update(self.controls, delta_time)
        if self.follow:
            self.camera.follow(self.session.x, self.session.y)

    def on_draw(self) -> None:
        self.clear()
        self.camera2d.position = self.camera.centre
        self.camera2d.zoom = self.camera.pixels_per_metre
        with self.camera2d.activate():
            self.track_shapes.draw()
            self._draw_car()
        self.default_camera.use()
        self._draw_hud()

    def _draw_car(self) -> None:
        s = self.session
        body = car_polygon(s.x, s.y, s.yaw, self._car_length, self._car_width)
        arcade.draw_polygon_filled([tuple(p) for p in body], CAR_COLOUR)
        nose = car_polygon(
            s.x + 0.35 * self._car_length * np.cos(s.yaw),
            s.y + 0.35 * self._car_length * np.sin(s.yaw),
            s.yaw,
            0.3 * self._car_length,
            0.6 * self._car_width,
        )
        arcade.draw_polygon_filled([tuple(p) for p in nose], CAR_NOSE_COLOUR)

    def _draw_hud(self) -> None:
        s = self.session
        self._set_text(self.text_lap, f"Lap {format_lap_time(s.lap_time)}")
        self._set_text(self.text_last, f"Last {format_lap_time(s.last_lap)}")
        self._set_text(self.text_best, f"Best {format_lap_time(s.best_lap)}")
        self._set_text(self.text_speed, f"{s.speed_kmh:5.0f} km/h")
        self._set_text(self.text_follow, "follow: on" if self.follow else "follow: off (F)")
        self.text_lap.draw()
        self.text_last.draw()
        self.text_best.draw()
        self.text_speed.draw()
        self.text_follow.draw()
        self.text_help.draw()
        if s.sliding:
            self.text_sliding.draw()
        if s.off_track:
            self.text_off.draw()


def run(track_id: str) -> None:
    """Open the drive viewer on a built track; returns when the window closes."""
    from f1sim.data.track_pipeline import load_track

    try:
        track = load_track(track_id)
    except FileNotFoundError:
        print(
            f"Track '{track_id}' hasn't been built yet. Build it first with:\n"
            f"    f1sim build-track {track_id}"
        )
        return
    DriveWindow(track)
    arcade.run()
