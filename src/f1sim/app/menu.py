"""Main menu: a top-down F1 car lit by the cursor, next to the menu.

``f1sim menu`` opens it. The layout follows the "interactive" hero card: a near-black card,
a soft spotlight beam fading in from the top-left, a headline, paragraph and menu on the left,
and the hero on the right third.

**The hero.** An F1 car seen from above, nose up. Best first:

0. **Driving** (:mod:`f1sim.app.ui.hero_drive`): the photographed car laps Marina Bay at a
   real pole lap's speed, forever, overtaking slower cars in other dark finishes. It stays
   fixed on screen (``DRIVE_CAR_HEIGHT`` of the window high, at ``DRIVE_X``, ``DRIVE_Y``)
   while the circuit, drawn as track-limit lines and kerbs, moves and turns underneath it;
   every car's front wheels steer. Behind the menu column the track is drawn faintly. Red
   and blue lamps flash by the track. One light model (:mod:`f1sim.app.ui.drive_shaders`)
   lights the floor, the track and every car: photographed diffuse light, live reflections,
   the cars' shadows cast away from the cursor. Each car is drawn in three layers: the
   soft ground shadow, the front wheels (turned by the steering angle) and the body.
1. **Light stage** (:mod:`f1sim.app.ui.light_stage`): path-traced photos of the blacked-out
   SF-26, one per softbox position on a grid; the cursor blends the four photos around it.
   Photographic: real soft shadows, softbox reflections in the clear coat, occlusion.
2. **Relit maps** (:mod:`f1sim.app.ui.hero_maps`): the same model's colour, normal and
   material renders, or the generic car drawn in code, shaded per pixel by a point light
   above the cursor, physically based (GGX highlights from each material's roughness and
   metal, a clear coat on the paint, ambient occlusion in the crevices).

The car doesn't move on screen; the cursor is the light. It also lights the floor around it.
Away from the light the car is only just visible. The light fades in after the spotlight
beam, and out when the cursor leaves the window.

The menu is the home screen of the app window (:mod:`f1sim.app.shell`). The picker lists
every circuit. Under it a strip says what the selected one needs and has the button for it
(``strip_state``): "Get track data" (the work of ``f1sim build-track``) or "Get real lap"
(``f1sim ghost``), run as a background job (:mod:`f1sim.app.jobs`) with the time shown, a
Cancel button (Esc) and, on a failure, a message and Retry. Drive and Train on a circuit
without data fetch it first and then open. The G key does what the button does; the Train
card has ``-`` / ``+`` for how long a run lasts (``TRAIN_MINUTES``). For the selected built
track:
- **Drive:** switches the window to the drive screen, with the newest ghost of the track if
  there is one.
- **Replay:** switches to the replay screen: the newest run of the track plus that ghost.
- **Train:** starts ``f1sim train`` (for the chosen time) on the track in the background and
  switches to the training screen (:mod:`f1sim.app.training`), which shows each new generation
  driving.
- **Quit.**

Esc on the drive, replay or training screen comes back here (leaving the training screen
stops the training).

Controls: mouse (hover, click), Up/Down to pick, Enter to open, Left/Right or the arrows to
change track, Esc to quit.

The testable logic (track and run discovery, the commands, the hero layout and the
screen-to-car mapping, the light's easing) is pure Python/numpy. :class:`MenuWindow` only
draws and maps input.

**Live reload.** ``scripts/dev_menu.py`` restarts the app whenever a source file changes;
the window keeps its place (:class:`f1sim.app.shell.AppWindow`).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1sim.app import jobs
from f1sim.app.ui import (
    car_top,
    drive_shaders,
    flags,
    hero_drive,
    hero_maps,
    light_stage,
    quality,
    theme,
)

CARD_BG = theme.hex_colour("#0A0A0B")
"""The card: near-black, like the original's ``bg-black/[0.96]``."""
HEADLINE_TOP = theme.hex_colour("#FAFAFA")
HEADLINE_BOTTOM = theme.hex_colour("#A3A3A3")
"""The headline's top-to-bottom gradient (neutral-50 to neutral-400), one colour per line."""
BODY_TEXT = theme.hex_colour("#D4D4D4")
SPOTLIGHT_DELAY_S = 0.75
SPOTLIGHT_FADE_S = 2.0
"""The beam fades in after this delay over this long (the original's animation)."""

HERO_VISIBLE_M = 3.3
"""The hero shows the car from the nose to this far back (just behind the driver)."""
HERO_TOP_MARGIN = 0.05
"""Gap above the nose, as a share of the window height."""
HERO_MAX_WIDTH = 1.0 / 3.0
"""The car's width (front wing tip to tip) is at most this share of the window width."""
LIGHT_TAU_S = 0.08
"""How quickly the light follows the cursor (frame-rate independent)."""
POWER_TAU_S = 0.35
"""How quickly the light fades in and out."""
LIGHT_LEVEL_RANGE = (0.25, 1.10)
LIGHT_LEVEL_STEP = 0.05
"""The mouse wheel sets the light's level: each click up or down moves it this much, from 75%
dimmer than the default (1) to 10% brighter."""
INTRO_LIGHT = (0.35, 2.0)
"""Where the light sits (metres right of the centre line, back from the top of the maps)
before the cursor first moves: over the cockpit."""
LIGHT_INTENSITY = 7.0
"""Point-light strength (diffuse white directly under it is about this / pi)."""
COAT_ROUGHNESS = 0.07
"""The clear coat's roughness: a little broader than real lacquer, so a point light's
highlight reads as a streak rather than a pinprick."""
STAGE_AMBIENT = 0.35
"""How much of the light stage's ambient photo (the car at rest) shows."""
STAGE_FADE_OUT_M = 0.9
"""The stage light fades out this far outside its grid (over the menu, say)."""
DRIVE_CAR_HEIGHT = 0.86
"""The driving car's length as a share of the window height."""
DRIVE_X = 0.72
DRIVE_Y = 0.48
"""The driving car's centre, as a share of the window width and height (a little low, so
more track shows ahead, where the slower cars come from)."""
ZOOM_IN = 2.0
ZOOM_OUT = 0.325
"""The follow camera's zoom at low and at top speed, as a multiple of the layout's own scale
(the scale the menu had before the speed zoom, with the car 86% of the window high). At
``ZOOM_IN`` the car is bigger than the window; ``ZOOM_OUT`` shows about six times as much
track."""
ZOOM_SPEED_LOW_MS = 25.0
ZOOM_SPEED_HIGH_MS = 85.0
"""Speeds (m/s) at and below which the camera is fully zoomed in, and at and above which it
is fully pulled back."""
ZOOM_TAU_S = 0.9
"""How slowly the zoom follows the speed (a smoothing time constant), so it breathes."""
TRACK_LINE_HALF_M = 0.07
"""Half the width of a drawn track-limit line, metres."""
TRACK_LINE = (1.0, 1.0, 1.0, 0.40)
KERB_RED = (0.92, 0.05, 0.25, 0.55)
KERB_WHITE = (0.92, 0.92, 0.94, 0.40)
TRACK_FADE_PX = 70.0
"""The track rises to full strength over this many pixels right of the menu column..."""
TRACK_UNDER_MENU = 0.22
"""... from this strength behind the menu column, so its lines sweep the whole page."""
DRIVE_AMBIENT = 0.3
"""The at-rest light on the driving cars (the ambient photo's level)."""
DRIVE_AMBIENT_SPEC = 0.05
"""The sheen of that broad overhead light in the clear coat (upward faces)."""
CAST_SHADOW_BLUR = 2.0
CAST_SHADOW_STRENGTH = 0.6
"""The cursor light's shadow on the floor: softer and lighter than the softbox would give, so
it doesn't repeat the car's outline beside it (seen from above, that reads as a ghost)."""
SELF_SHADOW_LIFT = 0.3
"""The photographed self-shadows are lifted this much: at full strength they read as a
shadow laid over the car, not as its shape."""
DRIVE_LIGHT = 2.0
"""The cursor light's strength, in units of the photographed softbox."""
DRIVE_EXPOSURE = 2.2
DRIVE_COAT_ROUGHNESS = 0.06
FLOOR_ALBEDO = 0.06
"""The floor (dark asphalt): lit by the same light as the cars."""
LINE_LIT = 0.55
"""The painted lines' and kerbs' reflectance (they're lit by the floor light too)."""
LAMP_GAIN = 60.0
"""The trackside lamps' strength at the peak of a flash, in softbox units (strobes: bright
enough to tint a car several metres away)."""
LAMP_RADIUS_M = 0.35
LAMP_REACH_M = 14.0
"""Lamps farther than this outside the window are left out (at most ``MAX_LAMPS``)."""
LAMP_GLOW_M = 0.18
BACKDROP_ALPHA = 0.30
"""The red and blue backdrop behind the menu and the driving cars
(``assets/hero/backdrop.webp``): it covers the window from its left edge, under the text."""
MAX_LAMPS = 8
MAX_SHADOW_CARS = 6
"""The shaders' array sizes; the quality tier lights and shadows fewer."""
FLOOR_LIGHT_UNIT = 12
"""The texture unit of the floor's light buffer (the car's maps take 0-11)."""
SHADOW_EDGE_M = 0.45
"""The ground shadow fades out over this distance from the photo's edges (the broad light's
shadow reaches them and would otherwise end in a visible rectangle)."""
WHEEL_GAIN = 1.2
"""The front tyres' colour is this much lighter (a dark rubber grey rather than the
blacked-out body's near-black): otherwise their steering is invisible against the dark."""
FLOOR_GLOW = 0.05
"""Brightness the cursor light adds to the floor (the card) at its centre."""

ITEMS = ("Drive", "Replay", "Train", "Quit")
KICKERS = {"Drive": "Race", "Replay": "Review", "Train": "Learn", "Quit": "Exit"}

MENU_SHARE = 0.40
"""The menu cards take the window's left 40%, so the driving car on the right stays clear."""
MENU_MARGIN = 16.0
CARD_GAP = 10.0
BAR_H = 52.0
PICKER_H = 84.0
FLAG_H = 18.0
FLAG_GAP = 11.0
"""The track picker's flag: its height, and the space between it and the track's name."""
PICKER_ARROWS_W = 84.0
PICKER_NAME_PAD = 10.0
"""The picker's arrows take this much of its right end; the name stops this far before them."""
ACTION_H = 44.0
ACTION_BTN_W = 150.0
"""The strip under the picker that says what the track needs (data, a real lap) and holds the
button for it; the button's width."""
TRAIN_MINUTES = (5, 10, 20, 30, 60, 120)
DEFAULT_TRAIN_MINUTES = 30
"""The training times the Train card offers (minutes), and the one it starts on."""
TIME_BTN = 22.0
TIME_TEXT_W = 46.0
FOOTER_H = 30.0
MIN_CARD = 96.0
CARD_FILL = theme.hex_colour("#0C0C0E", 0.80)
CARD_LINE = theme.white(0.13)

Rect = tuple[float, float, float, float]
"""``(left, bottom, width, height)`` in window pixels, y up."""


@dataclass(frozen=True)
class CardLayout:
    """The menu's card grid: a bar, the track picker, then ``ITEMS`` as a 2 x 2 block of
    squares, stacked from the top of the window inside its left 40%."""

    left: float
    col_w: float
    bar: Rect
    picker: Rect
    action: Rect
    cards: tuple[Rect, ...]
    footer_y: float
    square: float


def card_layout(width: float, height: float) -> CardLayout:
    """Lay the cards out for a window. The squares are as large as fit across 40% of the
    width and between the picker and the footer; the bar and picker match their block."""
    m, g = MENU_MARGIN, CARD_GAP
    across = (MENU_SHARE * width - 2.0 * m - g) / 2.0
    down = (height - 2.0 * m - BAR_H - PICKER_H - ACTION_H - 4.0 * g - FOOTER_H - g) / 2.0
    s = float(math.floor(max(MIN_CARD, min(across, down))))
    col_w = 2.0 * s + g
    top = height - m
    bar = (m, top - BAR_H, col_w, BAR_H)
    picker = (m, bar[1] - g - PICKER_H, col_w, PICKER_H)
    action = (m, picker[1] - g - ACTION_H, col_w, ACTION_H)
    row_top = action[1] - g
    cards = tuple(
        (m + (i % 2) * (s + g), row_top - s - (i // 2) * (s + g), s, s) for i in range(len(ITEMS))
    )
    return CardLayout(m, col_w, bar, picker, action, cards, m + 8.0, s)


def picker_arrow_at(picker: Rect, x: float, y: float) -> int:
    """-1 or 1 when ``(x, y)`` is on the track picker's left or right arrow, else 0."""
    left, bottom, w, h = picker
    if not (bottom <= y <= bottom + h):
        return 0
    if left + w - 84.0 <= x < left + w - 42.0:
        return -1
    if left + w - 42.0 <= x <= left + w:
        return 1
    return 0


# --------------------------------------------------------------------------- discovery


def built_tracks() -> list[tuple[str, str]]:
    """``(track_id, circuit name)`` of every built track, by id."""
    from f1sim import paths
    from f1sim.data import catalog

    # paths, not track_pipeline: importing the pipeline (numba and all) takes most of a
    # second at start-up
    return [
        (e.id, e.name)
        for e in sorted(catalog.CATALOG.values(), key=lambda e: e.id)
        if paths.track_file(e.id).exists()
    ]


def circuits() -> list[tuple[str, str]]:
    """``(track_id, circuit name)`` of every circuit in the catalogue, built or not, by id."""
    from f1sim.data import catalog

    return [(e.id, e.name) for e in sorted(catalog.CATALOG.values(), key=lambda e: e.id)]


def is_built(track_id: str) -> bool:
    from f1sim import paths

    return paths.track_file(track_id).exists()


def ghost_label(ghost: Path | None) -> str | None:
    """``RUS 2025`` for ``2025_silverstone_Q_RUS.npz``, or None without a ghost."""
    if ghost is None:
        return None
    year, _, rest = ghost.stem.partition("_")
    return f"{rest.rsplit('_', 1)[-1]} {year}"


def step_minutes(current: int, step: int) -> int:
    """The next training time (minutes) ``step`` places along ``TRAIN_MINUTES``, clamped."""
    i = TRAIN_MINUTES.index(current) if current in TRAIN_MINUTES else 0
    return TRAIN_MINUTES[min(max(i + step, 0), len(TRAIN_MINUTES) - 1)]


def action_button_rect(action: Rect) -> Rect:
    left, bottom, w, h = action
    return (left + w - 7.0 - ACTION_BTN_W, bottom + 7.0, ACTION_BTN_W, h - 14.0)


def time_buttons(card: Rect, pad: float = 12.0) -> tuple[Rect, Rect]:
    """The Train card's ``-`` and ``+`` boxes (training time), along its bottom edge."""
    left, bottom, _, _ = card
    y = bottom + 6.0
    return (
        (left + pad, y, TIME_BTN, TIME_BTN),
        (left + pad + TIME_BTN + TIME_TEXT_W, y, TIME_BTN, TIME_BTN),
    )


def inside(rect: Rect, x: float, y: float) -> bool:
    left, bottom, w, h = rect
    return left <= x <= left + w and bottom <= y <= bottom + h


def strip_state(built: bool, ghost: str | None, job) -> tuple[str, str | None, bool]:
    """What the strip under the track picker says: ``(text, button label or None, is a
    problem)``. ``job`` is the background job for this track (or None)."""
    if job is not None:
        if job.state == "running":
            return f"{job.spec.title} · {jobs.format_elapsed(job.elapsed)}", "Cancel", False
        if job.state == "failed":
            return job.message, "Retry", True
    if not built:
        return "Track data not downloaded yet", "Get track data", False
    if ghost is None:
        return "No real lap to race yet", "Get real lap", False
    return f"Real lap: {ghost}", None, False


def latest_run(track_id: str, runs_root: Path) -> Path | None:
    """The newest run folder under ``runs_root`` for ``track_id`` with a ``best_lap.npz``."""
    import tomllib

    best: tuple[float, Path] | None = None
    if not runs_root.is_dir():
        return None
    for lap in runs_root.glob("*/best_lap.npz"):
        cfg = lap.parent / "config.toml"
        try:
            run_track = tomllib.loads(cfg.read_text(encoding="utf-8")).get("track_id")
        except (OSError, ValueError):
            continue
        if run_track != track_id:
            continue
        mtime = lap.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, lap.parent)
    return None if best is None else best[1]


def latest_ghost(track_id: str, ghosts: Path) -> Path | None:
    """The newest ghost lap ``<year>_<track>_<session>_<driver>.npz`` of ``track_id``."""
    if not ghosts.is_dir():
        return None
    found = []
    for p in ghosts.glob(f"*_{track_id}_*.npz"):
        year, _, rest = p.stem.partition("_")
        if not (year.isdigit() and rest.startswith(f"{track_id}_")):
            continue
        if len(rest[len(track_id) + 1 :].split("_")) == 2:  # session and driver, nothing more
            found.append(p)
    return max(found, key=lambda p: p.stat().st_mtime, default=None)


def command(item: str, track_id: str, *, run: Path | None, ghost: Path | None) -> list[str] | None:
    """The ``f1sim`` arguments ``item`` opens for ``track_id``, or None (Quit, or Replay
    without a run)."""
    base = [sys.executable, "-m", "f1sim.cli"]
    if item == "Drive":
        return [*base, "drive", track_id, *(["--ghost", str(ghost)] if ghost else [])]
    if item == "Replay":
        if run is None:
            return None
        return [*base, "replay", track_id, str(run), *([str(ghost)] if ghost else [])]
    if item == "Train":
        return [*base, "train", "--track", track_id]
    return None


# --------------------------------------------------------------------------- the hero


@dataclass(frozen=True)
class HeroLayout:
    """Where the car maps sit in the window (window pixels, origin bottom-left)."""

    left: float
    top: float  # the maps' first row (the nose end) is drawn here
    px_per_m: float
    width_m: float
    length_m: float

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """``(left, bottom, width, height)`` of the whole map, in window pixels."""
        w = self.width_m * self.px_per_m
        h = self.length_m * self.px_per_m
        return self.left, self.top - h, w, h

    def to_car(self, x: float, y: float) -> tuple[float, float]:
        """Window pixel -> car metres (``x`` right of the centre line, ``y`` back from the
        front wing)."""
        return (
            (x - self.left) / self.px_per_m - 0.5 * self.width_m,
            (self.top - y) / self.px_per_m,
        )

    def to_window(self, cx: float, cy: float) -> tuple[float, float]:
        return self.left + (cx + 0.5 * self.width_m) * self.px_per_m, self.top - cy * self.px_per_m

    def zoomed(self, factor: float) -> HeroLayout:
        """This layout at ``factor`` times the scale, about the car's centre (which stays put
        on screen)."""
        ax, ay = self.to_window(0.0, 0.5 * self.length_m)
        ppm = self.px_per_m * factor
        return HeroLayout(
            left=ax - 0.5 * self.width_m * ppm,
            top=ay + 0.5 * self.length_m * ppm,
            px_per_m=ppm,
            width_m=self.width_m,
            length_m=self.length_m,
        )


def zoom_for_speed(speed: float) -> float:
    """The follow camera's zoom (1 = fully in) at ``speed`` m/s: pulls back smoothly between
    ``ZOOM_SPEED_LOW_MS`` and ``ZOOM_SPEED_HIGH_MS`` from ``ZOOM_IN`` to ``ZOOM_OUT`` (evenly
    in scale, so the zoom feels the same speed all the way)."""
    t = min(max((speed - ZOOM_SPEED_LOW_MS) / (ZOOM_SPEED_HIGH_MS - ZOOM_SPEED_LOW_MS), 0.0), 1.0)
    return ZOOM_IN * (ZOOM_OUT / ZOOM_IN) ** (t * t * (3.0 - 2.0 * t))


def hero_layout(width: float, height: float, size_m: tuple[float, float]) -> HeroLayout:
    """The maps (``size_m`` = width, length) in the right third, nose up: ``HERO_VISIBLE_M``
    of them fills the height below a small margin (the rest is cut by the window's bottom
    edge), unless that would make them wider than ``HERO_MAX_WIDTH`` of the window; centred on
    the right third."""
    width_m, length_m = size_m
    margin = HERO_TOP_MARGIN * height
    ppm = (height - margin) / HERO_VISIBLE_M
    ppm = min(ppm, HERO_MAX_WIDTH * width / width_m)
    centre_x = width * 5.0 / 6.0
    return HeroLayout(
        left=centre_x - 0.5 * width_m * ppm,
        top=height - margin,
        px_per_m=ppm,
        width_m=width_m,
        length_m=length_m,
    )


def drive_layout(
    width: float, height: float, size_m: tuple[float, float], rear_axle_y: float
) -> HeroLayout:
    """The driving car whole, ``DRIVE_CAR_HEIGHT`` of the window high, its centre at
    (``DRIVE_X``, ``DRIVE_Y``) of the window. ``size_m`` is the photo's (width, length)."""
    width_m, length_m = size_m
    ppm = DRIVE_CAR_HEIGHT * height / length_m
    left = DRIVE_X * width - 0.5 * width_m * ppm
    top = DRIVE_Y * height + 0.5 * length_m * ppm
    return HeroLayout(left=left, top=top, px_per_m=ppm, width_m=width_m, length_m=length_m)


def view_radius_m(width: float, height: float, layout: HeroLayout, rear_axle_y: float) -> float:
    """How far from the hero's rear axle (metres) a car can still show in the window: the
    farthest window corner, plus a car length."""
    ax, ay = layout.to_window(0.0, rear_axle_y)
    corner = max(math.hypot(x - ax, y - ay) for x in (0.0, width) for y in (0.0, height))
    return corner / layout.px_per_m + hero_drive.CAR_LENGTH_M


def hero_camera(layout: HeroLayout, rear_axle_y: float, hero_pose):
    """The follow camera: the hero's centre (world pose, ``(x, y, heading)``) and where it
    shows in the window (pixels). The camera follows the middle of the car, not its rear axle,
    so a car turning about its rear axle doesn't swing the view."""
    x, y, heading = hero_pose
    ahead = rear_axle_y - 0.5 * layout.length_m  # rear axle -> centre, along the heading
    centre = (x + ahead * math.cos(heading), y + ahead * math.sin(heading), heading)
    return centre, layout.to_window(0.0, 0.5 * layout.length_m)


def car_in_hero_frame(layout: HeroLayout, rear_axle_y: float, hero_pose, pose):
    """Where a car at ``pose`` (world) shows: its rear axle in window pixels and its rotation
    on screen (radians, anticlockwise), in the fixed top-down view that follows the hero's
    centre."""
    centre, anchor = hero_camera(layout, rear_axle_y, hero_pose)
    px = hero_drive.world_to_screen(pose[:2], centre, layout.px_per_m, anchor)
    return (float(px[0]), float(px[1])), pose[2] - hero_drive.VIEW_HEADING


def window_to_car(layout: HeroLayout, rear_axle_y: float, px, anchor, angle):
    """A window point (pixels) in the map frame (metres) of a car whose rear axle shows at
    ``anchor``, turned ``angle`` on screen."""
    vx, vy = (px[0] - anchor[0]) / layout.px_per_m, (px[1] - anchor[1]) / layout.px_per_m
    c, s = math.cos(-angle), math.sin(-angle)
    return c * vx - s * vy, rear_axle_y - (s * vx + c * vy)


def light_in_car(layout: HeroLayout, rear_axle_y: float, light_xy, anchor, angle):
    """The cursor light (hero map metres) in the map frame of a car whose rear axle shows at
    ``anchor`` (window pixels), turned ``angle`` on screen."""
    return window_to_car(layout, rear_axle_y, layout.to_window(*light_xy), anchor, angle)


def set_uniforms(program, **values) -> None:
    """Set the uniforms ``program`` uses (the compiler drops unused ones: skip those)."""
    for name, value in values.items():
        try:
            program[name] = value
        except KeyError:
            pass


def _flat(rows, n: int, width: int) -> tuple[float, ...]:
    """``rows`` of ``width`` floats, padded to ``n`` rows, flattened (an array uniform)."""
    out = [float(v) for row in rows[:n] for v in row]
    return tuple(out + [0.0] * (n * width - len(out)))


@dataclass
class Light:
    """The cursor light: plan position (car metres) easing towards a target, and a power
    easing towards on (cursor in the window: ``level``) or off (0)."""

    x: float = INTRO_LIGHT[0]
    y: float = INTRO_LIGHT[1]
    target_x: float = INTRO_LIGHT[0]
    target_y: float = INTRO_LIGHT[1]
    power: float = 0.0
    on: bool = True
    level: float = 1.0

    def scroll(self, clicks: float) -> None:
        """Brighter for wheel clicks up, dimmer down, within ``LIGHT_LEVEL_RANGE``."""
        lo, hi = LIGHT_LEVEL_RANGE
        self.level = min(max(self.level + clicks * LIGHT_LEVEL_STEP, lo), hi)

    def aim(self, x: float, y: float) -> None:
        self.target_x, self.target_y = x, y
        self.on = True

    def off(self) -> None:
        self.on = False

    def update(self, dt: float, ready: bool = True) -> None:
        """Ease by ``dt`` seconds; ``ready=False`` holds the power at 0 (the intro)."""
        dt = max(dt, 0.0)
        k = 1.0 - math.exp(-dt / LIGHT_TAU_S)
        self.x += k * (self.target_x - self.x)
        self.y += k * (self.target_y - self.y)
        goal = self.level if (self.on and ready) else 0.0
        self.power += (1.0 - math.exp(-dt / POWER_TAU_S)) * (goal - self.power)


# ------------------------------------------------------------------------------ shaders

_FS_VS = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_vert, 0.0, 1.0); }
"""

_BACKDROP_FS = """
#version 330
uniform sampler2D u_image;
uniform vec2 u_size;
uniform float u_alpha;
in vec2 v_uv;
out vec4 frag;
void main() {
    // Cover the window, keeping the image's aspect ratio: its left edge on the window's
    // left edge, centred vertically (any spare width is cut off on the right)
    vec2 p = v_uv * u_size;
    vec2 im = vec2(textureSize(u_image, 0));
    float s = max(u_size.x / im.x, u_size.y / im.y);
    vec2 uv = (p - vec2(0.0, 0.5 * (u_size.y - im.y * s))) / (im * s);
    frag = vec4(texture(u_image, uv).rgb * u_alpha, u_alpha);  // premultiplied
}
"""

_SPOT_FS = """
#version 330
uniform vec2 u_size;
uniform float u_appear;
uniform vec3 u_bg;
uniform vec2 u_light_px;     // the cursor light on the floor, window pixels
uniform float u_light_r_px;  // its reach, pixels
uniform float u_light_power;
uniform float u_glow;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec2 p = v_uv * u_size;
    float L = length(u_size);
    // A soft beam from beyond the top-left corner, angled down to the right
    vec2 o = vec2(-0.08 * u_size.x, 1.18 * u_size.y);
    vec2 dir = normalize(vec2(0.95, -1.0));
    vec2 d = p - o;
    float along = dot(d, dir);
    float across = abs(d.x * dir.y - d.y * dir.x);
    float width = 0.05 * L + 0.16 * max(along, 0.0);
    float beam = exp(-2.2 * (across / width) * (across / width))
        * smoothstep(0.0, 0.25 * L, along) * exp(-along / (0.85 * L));
    // The cursor light's pool on the floor
    float r = length(p - u_light_px) / u_light_r_px;
    float pool = u_glow * u_light_power / (1.0 + 9.0 * r * r) * (1.0 - smoothstep(0.6, 1.0, r));
    // Very faint grain so the gradients don't band
    float n = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    vec3 c = u_bg + vec3(0.17 * beam * u_appear) + vec3(pool) + n / 255.0;
    frag = vec4(c, 1.0);
}
"""

_HERO_VS = """
#version 330
uniform vec4 u_rect;  // left, bottom, width, height in window pixels
uniform vec2 u_size;  // window size in pixels
in vec2 in_vert;      // 0..1
in vec2 in_uv;
out vec2 v_uv;
void main() {
    vec2 p = u_rect.xy + in_vert * u_rect.zw;
    v_uv = in_uv;
    gl_Position = vec4(p / u_size * 2.0 - 1.0, 0.0, 1.0);
}
"""

_STAGE_FS = """
#version 330
uniform sampler2DArray u_stage;  // one photo per softbox position
uniform sampler2D u_ambient;     // the car at rest; alpha = coverage
uniform vec4 u_layers;           // the four photos around the light...
uniform vec4 u_weights;          // ...and their bilinear weights
uniform float u_power;
uniform float u_ambient_level;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec4 amb = texture(u_ambient, v_uv);
    if (amb.a <= 0.0) discard;
    vec3 lit = u_weights.x * texture(u_stage, vec3(v_uv, u_layers.x)).rgb
        + u_weights.y * texture(u_stage, vec3(v_uv, u_layers.y)).rgb
        + u_weights.z * texture(u_stage, vec3(v_uv, u_layers.z)).rgb
        + u_weights.w * texture(u_stage, vec3(v_uv, u_layers.w)).rgb;
    // Photos are display-encoded: add the light in (approximately) linear space
    vec3 c = sqrt(amb.rgb * amb.rgb * u_ambient_level + lit * lit * u_power);
    frag = vec4(c * amb.a, amb.a);  // premultiplied
}
"""

_HERO_FS = """
#version 330
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_surface;   // r roughness, g metallic, b height / u_height_scale, a AO
uniform sampler2D u_coat;      // r clear-coat weight
uniform vec2 u_size_m;         // map width and length, metres
uniform float u_height_scale;
uniform vec3 u_light;          // light position, map metres (x right of centre, y down, z up)
uniform float u_power;
uniform float u_intensity;
uniform float u_core;
uniform float u_reach;
uniform float u_ambient;
uniform float u_coat_rough;
in vec2 v_uv;
out vec4 frag;
const float PI = 3.14159265;

float ggx(float ndh, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float smith(float ndl, float ndv, float rough) {
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
}

void main() {
    vec4 alb = texture(u_albedo, v_uv);
    if (alb.a <= 0.0) discard;
    vec3 base = alb.rgb * alb.rgb;  // sRGB-ish to linear
    vec3 n = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    vec4 s = texture(u_surface, v_uv);
    float rough = clamp(s.r, 0.04, 1.0);
    float metal = s.g;
    float h = s.b * u_height_scale;
    float ao = s.a;
    float coat = texture(u_coat, v_uv).r;
    // This pixel in map metres (row 0, the nose end, is at the top: v = 1)
    vec2 pos = vec2((v_uv.x - 0.5) * u_size_m.x, (1.0 - v_uv.y) * u_size_m.y);
    // Vectors in the normal map's frame: x right, y towards the nose, z up (to the viewer)
    vec3 Lv = vec3(u_light.x - pos.x, -(u_light.y - pos.y), u_light.z - h);
    vec3 L = normalize(Lv);
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);
    float ndl = max(dot(n, L), 0.0);
    float ndv = max(n.z, 1e-3);
    float ndh = max(dot(n, H), 0.0);
    float vdh = max(dot(V, H), 0.0);
    float fres = pow(1.0 - vdh, 5.0);
    vec3 F0 = mix(vec3(0.04), base, metal);
    vec3 F = F0 + (1.0 - F0) * fres;
    vec3 spec = ggx(ndh, rough) * smith(ndl, ndv, rough) * F / (4.0 * ndl * ndv + 1e-3);
    float Fc = 0.04 + 0.96 * fres;
    float specC = coat * ggx(ndh, u_coat_rough) * smith(ndl, ndv, u_coat_rough) * Fc
        / (4.0 * ndl * ndv + 1e-3);
    vec3 diff = (1.0 - metal) * base / PI * (1.0 - coat * Fc);
    float r = length(u_light.xy - pos);
    float att = u_power * u_intensity / (1.0 + (r / u_core) * (r / u_core))
        * (1.0 - smoothstep(0.0, u_reach, r));
    vec3 c = (diff + spec + vec3(specC)) * ndl * att;
    // A dim, soft ambient so the unlit car is just visible (occluded in the crevices)
    c += ao * (base * (1.0 - metal) + F0 * 0.5) * u_ambient;
    c = c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14);  // filmic curve
    c = sqrt(max(c, 0.0));  // back to display
    frag = vec4(c * alb.a, alb.a);  // premultiplied
}
"""


# ------------------------------------------------------------------------------ window


def _rgb(c: theme.Colour) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


@dataclass
class Button:
    item: str
    rect: tuple[float, float, float, float]  # left, bottom, width, height


def _texture_bytes(maps: hero_maps.HeroMaps) -> dict[str, bytes]:
    """The maps as RGBA8 texture data, rows flipped (GL's first row is the bottom)."""

    def rgba(a):
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[-1] < 4:
            pad = np.full((*a.shape[:2], 4 - a.shape[-1]), 255, np.uint8)
            a = np.concatenate([a, pad], axis=-1)
        return np.ascontiguousarray(np.flipud(a)).tobytes()

    return {
        "albedo": rgba(maps.albedo),
        "normal": rgba(maps.normal),
        "surface": rgba(maps.surface),
        "coat": rgba(maps.coat),
    }


def _make_view_class():
    import arcade
    from arcade.gl import BufferDescription, geometry

    from f1sim.app.shell import Screen

    class MenuView(Screen):
        """The main menu screen; see the module docstring."""

        def __init__(self, runs_root: Path = Path("runs")):
            super().__init__(on_back=None)  # Esc on the menu quits
            self.title = "f1sim"
            theme.load_fonts()
            self.background_color = CARD_BG
            self.runs_root = runs_root
            self.tracks = circuits() or [("silverstone", "Silverstone Circuit")]
            ids = [t[0] for t in self.tracks]
            self.track_i = ids.index("silverstone") if "silverstone" in ids else 0
            self.focus = 0
            self.job = None  # the background download (jobs.Job) and what it is for
            self.job_kind = ""
            self.job_track = ""
            self.job_then: str | None = None
            self.built = False
            self.ghost_name: str | None = None
            self.train_minutes = DEFAULT_TRAIN_MINUTES
            self.hover: int | None = None
            self.light = Light()
            self.t0 = time.perf_counter()
            self.clock = 0.0  # seconds of animation (the lamps' flashes)
            self.zoom = ZOOM_IN  # the follow camera's zoom now (see zoom_for_speed)
            self.status = ""
            self.status_t = -10.0
            circuit = hero_drive.load_circuit()
            self.dstage = hero_drive.load_stage() if circuit is not None else None
            self.drive = None
            self.traffic = None
            if self.dstage is not None:
                self.drive = hero_drive.Drive(circuit, wheelbase_m=self.dstage.wheelbase)
                self.traffic = hero_drive.Traffic(self.drive)
            self.stage = light_stage.load() if self.drive is None else None
            no_photos = self.stage is None and self.drive is None
            self.maps = hero_maps.load() if no_photos else None
            self._init_gl()
            self._init_text()
            self._refresh_track()
            self.on_resize(self.width, self.height)

        # ---------------------------------------------------------------- setup

        def _init_gl(self) -> None:
            ctx = self.ctx
            self._fs_quad = geometry.quad_2d_fs()
            self._p_spot = ctx.program(vertex_shader=_FS_VS, fragment_shader=_SPOT_FS)
            unit = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1], dtype="f4")
            self._unit_quad = ctx.geometry(
                [BufferDescription(ctx.buffer(data=unit.tobytes()), "2f 2f", ["in_vert", "in_uv"])],
                mode=ctx.TRIANGLE_STRIP,
            )
            if self.drive is not None:
                self._init_drive()
                return
            if self.stage is not None:
                self._init_stage()
                return
            self._p_hero = ctx.program(vertex_shader=_HERO_VS, fragment_shader=_HERO_FS)
            h, w = self.maps.albedo.shape[:2]
            self._tex = {}
            for unit_i, (name, data) in enumerate(_texture_bytes(self.maps).items()):
                tex = ctx.texture(
                    (w, h),
                    components=4,
                    data=data,
                    filter=(ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR),
                    wrap_x=ctx.CLAMP_TO_EDGE,
                    wrap_y=ctx.CLAMP_TO_EDGE,
                )
                tex.build_mipmaps()
                self._tex[name] = (tex, unit_i)
            p = self._p_hero
            p["u_albedo"], p["u_normal"], p["u_surface"], p["u_coat"] = 0, 1, 2, 3
            p["u_size_m"] = self.maps.size_m
            p["u_height_scale"] = self.maps.height_scale_m
            p["u_intensity"] = LIGHT_INTENSITY
            p["u_core"] = car_top.LIGHT_CORE_M
            p["u_reach"] = car_top.LIGHT_REACH_M
            p["u_ambient"] = car_top.AMBIENT
            p["u_coat_rough"] = COAT_ROUGHNESS

        def _init_drive(self) -> None:
            ctx = self.ctx
            st, c = self.dstage, self.drive.circuit
            # Track: kerb blocks and the two track-limit lines, world metres, per-vertex colour
            tris = [c.kerb]
            cols = [np.where(c.kerb_red[:, None], [KERB_RED], [KERB_WHITE])]
            for edge in (c.left, c.right):
                q = hero_drive.edge_quads(edge, TRACK_LINE_HALF_M)
                tris.append(q)
                cols.append(np.tile([TRACK_LINE], (q.shape[0], 1)))
            pos = np.concatenate(tris).reshape(-1, 2)
            col = np.repeat(np.concatenate(cols), 3, axis=0)
            data = np.hstack([pos, col]).astype("f4")
            self._track = ctx.geometry(
                [BufferDescription(ctx.buffer(data=data.tobytes()), "2f 4f", ["in_pos", "in_col"])],
                mode=ctx.TRIANGLES,
            )
            ds = drive_shaders
            self._p_track = ctx.program(vertex_shader=ds.TRACK_VS, fragment_shader=ds.TRACK_FS)
            self._p_floor = ctx.program(vertex_shader=_FS_VS, fragment_shader=ds.FLOOR_FS)
            self._p_floor_light = ctx.program(
                vertex_shader=_FS_VS, fragment_shader=ds.FLOOR_LIGHT_FS
            )
            self._floor_buf = None  # (size, texture, framebuffer): see _floor_buffer
            self._p_drive = ctx.program(vertex_shader=ds.CAR_VS, fragment_shader=ds.CAR_FS)
            mip = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
            clamp = ctx.CLAMP_TO_EDGE

            def array(images):  # (n, h, w) grey
                n, h, w = images.shape
                t = ctx.texture_array(
                    (w, h, n),
                    components=1,
                    data=np.ascontiguousarray(images[:, ::-1]).tobytes(),
                    filter=mip,
                    wrap_x=clamp,
                    wrap_y=clamp,
                )
                t.build_mipmaps()
                return t

            def single(image):
                h, w, _ = image.shape
                t = ctx.texture(
                    (w, h),
                    components=4,
                    data=np.ascontiguousarray(image[::-1]).tobytes(),
                    filter=mip,
                    wrap_x=clamp,
                    wrap_y=clamp,
                )
                t.build_mipmaps()
                return t

            self._p_backdrop = None
            backdrop = hero_drive.ASSETS / "backdrop.webp"
            if backdrop.exists():
                from PIL import Image

                image = np.asarray(Image.open(backdrop).convert("RGBA"))
                self._backdrop_tex = single(image)
                self._p_backdrop = ctx.program(vertex_shader=_FS_VS, fragment_shader=_BACKDROP_FS)

            # Texture units: the softbox photos, then each layer's ambient photo and maps
            self._drive_tex = [array(st.body["light"]), array(st.wheels["light"])]
            names = []
            for layer, maps in (("body", st.body), ("wheel", st.wheels)):
                for m, short in zip(
                    hero_drive.STAGE_MAPS, ("amb", "alb", "nrm", "srf", "ao"), strict=True
                ):
                    self._drive_tex.append(single(maps[m]))
                    names.append(f"u_{layer}_{short}")
            units = {"u_body_light": 0, "u_wheel_light": 1}
            units.update({n: i + 2 for i, n in enumerate(names)})
            self._units = units
            light = {
                "u_white": st.white_floor,
                "u_h0": st.light_z,
                "u_r0": 0.5 * st.softbox_m,
                "u_exposure": DRIVE_EXPOSURE,
            }
            (x0, y0), (x1, y1) = sorted(st.wheel_centres)
            row0, rows = st.wheel_rows
            set_uniforms(
                self._p_drive,
                **units,
                **light,
                u_ambient_level=DRIVE_AMBIENT,
                u_ambient_gain=st.ambient_gain,
                u_amb_spec=DRIVE_AMBIENT_SPEC,
                u_vis_lift=SELF_SHADOW_LIFT,
                u_height_scale=st.height_scale,
                u_coat_rough=DRIVE_COAT_ROUGHNESS,
                u_wheel_gain=WHEEL_GAIN,
                u_lamp_r=LAMP_RADIUS_M,
                u_size_m=st.size_m,
                u_axle_y=st.rear_axle_y,
                u_wheels_m=(x0, y0, x1, y1),
                u_wheel_r=st.wheel_radius + 0.08,
                u_strip=(row0 / st.px_per_m, rows / st.px_per_m),
            )
            for p in (self._p_floor, self._p_track):
                set_uniforms(p, **light, u_floor_light=FLOOR_LIGHT_UNIT)
            set_uniforms(self._p_track, u_floor_albedo=FLOOR_ALBEDO, u_line_lit=LINE_LIT)
            for p in (self._p_floor_light,):
                set_uniforms(
                    p,
                    **light,
                    u_cover=units["u_body_alb"],
                    u_height=units["u_body_srf"],
                    u_contact=units["u_body_amb"],
                    u_contact_edge=SHADOW_EDGE_M,
                    u_car_size_m=st.size_m,
                    u_axle_y=st.rear_axle_y,
                    u_height_scale=st.height_scale,
                    u_tex_ppm=st.px_per_m,
                    u_shadow_blur=CAST_SHADOW_BLUR,
                    u_shadow_strength=CAST_SHADOW_STRENGTH,
                    u_floor_albedo=FLOOR_ALBEDO,
                    u_line_lit=LINE_LIT,
                )
            self.lamps = hero_drive.trackside_lamps(
                c, room=(self.traffic.room_left, self.traffic.room_right)
            )
            self._lamp_xy = np.array([(lp.x, lp.y) for lp in self.lamps])

        def _init_stage(self) -> None:
            ctx = self.ctx
            st = self.stage
            self._p_stage = ctx.program(vertex_shader=_HERO_VS, fragment_shader=_STAGE_FS)
            n, h, w, _ = st.images.shape
            lin = (ctx.LINEAR, ctx.LINEAR)
            # GL's first row is the bottom: flip every photo
            self._stage_tex = ctx.texture_array(
                (w, h, n),
                components=3,
                data=np.ascontiguousarray(st.images[:, ::-1]).tobytes(),
                filter=lin,
                wrap_x=ctx.CLAMP_TO_EDGE,
                wrap_y=ctx.CLAMP_TO_EDGE,
            )
            self._ambient_tex = ctx.texture(
                (w, h),
                components=4,
                data=np.ascontiguousarray(st.ambient[::-1]).tobytes(),
                filter=lin,
                wrap_x=ctx.CLAMP_TO_EDGE,
                wrap_y=ctx.CLAMP_TO_EDGE,
            )
            p = self._p_stage
            p["u_stage"], p["u_ambient"] = 0, 1
            p["u_ambient_level"] = STAGE_AMBIENT

        def _init_text(self) -> None:
            self.t_kicker = theme.micro("Neuroevolution", theme.TEXT_SECONDARY, anchor_x="right")
            self.t_logo = theme.make_text("F1SIM", 17, theme.TEXT_PRIMARY, face="display")
            self.t_track_label = theme.micro("Track", theme.TEXT_SECONDARY)
            self.t_track = theme.make_text("", 22, theme.TEXT_PRIMARY, face="display")
            self._track_font = self.t_track.font_size
            self._track_name = ""
            self._flag = None
            self._notes = [""] * len(ITEMS)
            self.t_arrows = [
                theme.make_text("‹", 26, theme.TEXT_SECONDARY, face="medium", anchor_x="center"),
                theme.make_text("›", 26, theme.TEXT_SECONDARY, face="medium", anchor_x="center"),
            ]
            self.t_kickers = [
                theme.micro(KICKERS[i], theme.TEXT_SECONDARY, anchor_x="left") for i in ITEMS
            ]
            self.t_items = [
                theme.make_text(i, 22, theme.TEXT_PRIMARY, face="display") for i in ITEMS
            ]
            self.t_notes = [
                theme.make_text("", 12, theme.TEXT_TERTIARY, face="medium") for _ in ITEMS
            ]
            self.t_action = theme.make_text("", 13, theme.TEXT_SECONDARY, face="medium")
            self.t_action_btn = theme.make_text(
                "", 12, theme.TEXT_PRIMARY, face="display", anchor_x="center"
            )
            self.t_minutes = theme.make_text(
                "", 12, theme.TEXT_PRIMARY, face="medium", anchor_x="center"
            )
            self.t_footer = theme.micro(
                "↑↓ select   ·   ←→ track   ·   enter open   ·   g data   ·   esc quit",
                theme.TEXT_TERTIARY
            )
            self.t_status = theme.make_text("", 12, theme.TEXT_SECONDARY, face="medium")

        def _refresh_track(self) -> None:
            from f1sim import paths

            tid, name = self.tracks[self.track_i]
            self._track_name = name.upper()
            self._flag = flags.flag_for(tid)
            self._fit_track()
            self.run = latest_run(tid, self.runs_root)
            self.ghost = latest_ghost(tid, paths.ghosts_dir())
            self.built = is_built(tid)
            self.ghost_name = ghost_label(self.ghost)
            notes = {
                "Drive": f"vs ghost {self.ghost.stem.split('_')[-1]}" if self.ghost else "solo",
                "Replay": self.run.name if self.run else "no runs yet",
                "Train": "",  # the time control is drawn here instead
                "Quit": "",
            }
            self._notes = [notes[item] for item in ITEMS]
            for i in range(len(ITEMS)):
                self._fit_note(i)

        def _fit_track(self) -> None:
            """Place the track's flag and name in the picker, the name shrunk (then cut) so it
            never reaches the arrows."""
            cards = getattr(self, "cards", None)
            if cards is None:
                return
            pl, pb, pw, ph = cards.picker
            x = pl + 18.0
            self._flag_rect = None
            if self._flag is not None:
                fw = FLAG_H * flags.ASPECT
                self._flag_rect = (x, pb + 19.0, fw, FLAG_H)
                x += fw + FLAG_GAP
            room = pl + pw - PICKER_ARROWS_W - PICKER_NAME_PAD - x
            t, name = self.t_track, self._track_name
            t.font_size = self._track_font
            t.text = name
            t.font_size = self._track_font * flags.fit_scale(t.content_width, room)
            n = len(name)
            while t.content_width > room and n > 1:  # still too wide at the smallest size
                n -= 1
                t.text = name[:n].rstrip() + "..."
            t.position = (x, pb + 20.0)

        def _fit_note(self, i: int, room: float | None = None) -> None:
            """Cut card ``i``'s note (with "...") so it stays inside its card."""
            cards = getattr(self, "cards", None)
            if room is None:
                if cards is None:
                    return
                room = cards.cards[i][2] - 36.0
            t, note = self.t_notes[i], self._notes[i]
            t.text = note
            n = len(note)
            while t.content_width > room and n > 1:
                n -= 1
                t.text = note[:n].rstrip() + "..."

        def _draw_strip(self) -> None:
            """The strip under the picker: what the track needs, and a button for it."""
            lay = self.cards
            text, label, problem = self._strip()
            self._panel(lay.action)
            al, ab, aw, ah = lay.action
            room = aw - 28.0 - (ACTION_BTN_W + 7.0 if label else 0.0)
            t = self.t_action
            t.text = text
            n = len(text)
            while t.content_width > room and n > 1:
                n -= 1
                t.text = text[:n].rstrip() + "..."
            t.color = theme.ACCENT if problem else theme.TEXT_SECONDARY
            if label:
                self._panel(action_button_rect(lay.action), theme.ACCENT, theme.ACCENT)
            self.t_action_btn.text = label.upper() if label else ""
            job = self._job_here()
            if job is not None and job.state == "running":  # a bar sweeping along the bottom
                p = (self.clock * 0.7) % 1.3
                x0, x1 = max(p - 0.3, 0.0), min(p, 1.0)
                if x1 > x0:
                    arcade.draw_lrbt_rectangle_filled(
                        al + x0 * aw, al + x1 * aw, ab, ab + 3.0, theme.ACCENT
                    )

        def _draw_train_time(self) -> None:
            """``-  30 min  +`` on the Train card: how long a training run lasts."""
            card = self.cards.cards[ITEMS.index("Train")]
            lit = ITEMS.index("Train") == self.focus
            ink = theme.TEXT_PRIMARY
            line = theme.white(0.8) if lit else CARD_LINE
            minus, plus = time_buttons(card)
            for rect in (minus, plus):
                self._panel(rect, theme.white(0.0), line)
            (ml, mb, mw, mh), (pl_, _, pw_, _) = minus, plus
            cy = mb + 0.5 * mh
            arcade.draw_line(ml + 6.0, cy, ml + mw - 6.0, cy, ink, 1.4)
            arcade.draw_line(pl_ + 6.0, cy, pl_ + pw_ - 6.0, cy, ink, 1.4)
            arcade.draw_line(pl_ + 0.5 * pw_, cy - 5.0, pl_ + 0.5 * pw_, cy + 5.0, ink, 1.4)
            self.t_minutes.text = f"{self.train_minutes} min"
            self.t_minutes.position = (ml + mw + 0.5 * TIME_TEXT_W, cy - 5.0)

        def _draw_flag(self) -> None:
            if self._flag is None or self._flag_rect is None:
                return
            left, bottom, w, h = self._flag_rect
            f = self._flag

            def box(x0, y0, x1, y1, colour) -> None:
                arcade.draw_lrbt_rectangle_filled(
                    left + x0 * w,
                    left + x1 * w,
                    bottom + y0 * h,
                    bottom + y1 * h,
                    theme.hex_colour(colour),
                )

            for a, b, colour in f.bands():
                if f.vertical:
                    box(a, 0.0, b, 1.0, colour)
                else:
                    box(0.0, 1.0 - b, 1.0, 1.0 - a, colour)
            for shape in f.shapes:
                if isinstance(shape, flags.Rect):
                    box(shape.x0, shape.y0, shape.x1, shape.y1, shape.colour)
                elif isinstance(shape, flags.Line):
                    arcade.draw_line(
                        left + shape.x0 * w,
                        bottom + shape.y0 * h,
                        left + shape.x1 * w,
                        bottom + shape.y1 * h,
                        theme.hex_colour(shape.colour),
                        shape.width * h,
                    )
                else:
                    arcade.draw_circle_filled(
                        left + shape.x * w,
                        bottom + shape.y * h,
                        shape.r * h,
                        theme.hex_colour(shape.colour),
                    )
            arcade.draw_lrbt_rectangle_outline(left, left + w, bottom, bottom + h, CARD_LINE, 1)

        # --------------------------------------------------------------- layout

        def on_resize(self, width: int, height: int) -> None:
            super().on_resize(width, height)
            w, h = float(width), float(height)
            # Text is laid out on whole pixels (a glyph between pixels is resampled: blurred)
            lay = card_layout(w, h)
            self.cards = lay
            self.left, col_w = lay.left, lay.col_w
            self.col_w = col_w
            pad = 18.0
            bl, bb, bw, bh = lay.bar
            self.t_logo.position = (bl + pad, bb + 0.5 * bh - 6.0)
            self.t_kicker.position = (bl + bw - pad, bb + 0.5 * bh - 4.0)
            pl, pb, pw, ph = lay.picker
            self.t_track_label.position = (pl + pad, pb + ph - 28.0)
            self._fit_track()
            self.t_arrows[0].position = (pl + pw - 63.0, pb + 0.5 * ph - 10.0)
            self.t_arrows[1].position = (pl + pw - 21.0, pb + 0.5 * ph - 10.0)
            self.buttons = []
            size = float(round(min(26.0, max(15.0, 0.15 * lay.square))))
            for i, rect in enumerate(lay.cards):
                self.buttons.append(Button(ITEMS[i], rect))
                cl, cb, cw, ch = rect
                self.t_kickers[i].position = (cl + pad, cb + ch - 27.0)
                self.t_items[i].font_size = theme.px_to_pt(size)
                self.t_items[i].position = (cl + pad, cb + 36.0)
                self.t_notes[i].position = (cl + pad, cb + 16.0)
                self._fit_note(i, cw - 2.0 * pad)
            al, ab, aw, ah = lay.action
            self.t_action.position = (al + 14.0, ab + 0.5 * ah - 5.0)
            bl2, bb2, bw2, bh2 = action_button_rect(lay.action)
            self.t_action_btn.position = (bl2 + 0.5 * bw2, bb2 + 0.5 * bh2 - 5.0)
            self.t_footer.position = (self.left, lay.footer_y)
            self.t_status.position = (self.left, lay.footer_y)
            if self.drive is not None:
                self.hero = drive_layout(w, h, self.dstage.size_m, self.dstage.rear_axle_y)
                self.traffic.view_m = view_radius_m(
                    w, h, self.hero.zoomed(ZOOM_OUT), self.dstage.rear_axle_y
                )
                self.track_fade = (
                    self.left + col_w + 10.0,
                    self.left + col_w + 10.0 + TRACK_FADE_PX,
                )
            else:
                size_m = (self.stage or self.maps).size_m
                self.hero = hero_layout(w, h, size_m)

        # ---------------------------------------------------------------- input

        def _button_at(self, x: float, y: float) -> int | None:
            for i, b in enumerate(self.buttons):
                left, bottom, bw, bh = b.rect
                if left <= x <= left + bw and bottom <= y <= bottom + bh:
                    return i
            return None

        def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
            self.light.aim(*self.view.to_car(x, y))
            self.hover = self._button_at(x, y)
            if self.hover is not None:
                self.focus = self.hover

        def on_mouse_enter(self, x: int, y: int) -> None:
            self.light.aim(*self.view.to_car(x, y))

        def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
            self.light.scroll(scroll_y)

        def on_mouse_leave(self, x: int, y: int) -> None:
            self.light.off()
            self.hover = None

        def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
            if self._strip_button() is not None and inside(
                action_button_rect(self.cards.action), x, y
            ):
                self._strip_action()
                return
            minus, plus = time_buttons(self.cards.cards[ITEMS.index("Train")])
            for rect, step in ((minus, -1), (plus, 1)):
                if inside(rect, x, y):
                    self.train_minutes = step_minutes(self.train_minutes, step)
                    return
            i = self._button_at(x, y)
            if i is not None:
                self.focus = i
                self.open_item(ITEMS[i])
                return
            step = picker_arrow_at(self.cards.picker, x, y)
            if step:
                self.change_track(step)

        def on_key_press(self, symbol: int, modifiers: int) -> None:

            k = arcade.key
            if symbol == k.ESCAPE:
                if self.job is not None and self.job.running:
                    self.job.cancel()
                else:
                    self.close()
            elif symbol == k.G:
                self._strip_action()
            elif symbol in (k.MINUS, k.NUM_SUBTRACT):
                self.train_minutes = step_minutes(self.train_minutes, -1)
            elif symbol in (k.EQUAL, k.PLUS, k.NUM_ADD):
                self.train_minutes = step_minutes(self.train_minutes, 1)
            elif symbol in (k.UP, k.W):
                self.focus = (self.focus - 1) % len(ITEMS)
            elif symbol in (k.DOWN, k.S):
                self.focus = (self.focus + 1) % len(ITEMS)
            elif symbol in (k.LEFT, k.A):
                self.change_track(-1)
            elif symbol in (k.RIGHT, k.D):
                self.change_track(1)
            elif symbol in (k.ENTER, k.RETURN, k.SPACE):
                self.open_item(ITEMS[self.focus])

        def change_track(self, step: int) -> None:
            self.track_i = (self.track_i + step) % len(self.tracks)
            self._refresh_track()

        def open_item(self, item: str) -> None:
            if item == "Quit":
                self.close()
                return
            tid, name = self.tracks[self.track_i]
            if item in ("Drive", "Train") and not self.built:
                self._start_job("track", tid, then=item)  # then it opens by itself
                return
            self._open_screen(item, tid, name)

        # ------------------------------------------------------------- downloads

        def _job_here(self):
            """The background job if it is for the track on screen."""
            tid = self.tracks[self.track_i][0]
            return self.job if self.job is not None and self.job_track == tid else None

        def _strip(self) -> tuple[str, str | None, bool]:
            return strip_state(self.built, self.ghost_name, self._job_here())

        def _strip_button(self) -> str | None:
            return self._strip()[1]

        def _start_job(self, kind: str, tid: str, then: str | None = None) -> None:
            if self.job is not None and self.job.running:
                self._say("Wait for the download to finish (Esc cancels it)")
                return
            if kind == "track":
                spec = jobs.JobSpec("Getting track data", lambda: jobs.get_track_data(tid))
            else:
                spec = jobs.JobSpec("Getting real lap", lambda: jobs.get_real_lap(tid))
            self.job, self.job_kind, self.job_track, self.job_then = jobs.Job(spec), kind, tid, then

        def _strip_action(self) -> None:
            """The strip's button (and the G key): cancel, retry, or get what is missing."""
            tid = self.tracks[self.track_i][0]
            job = self._job_here()
            if job is not None and job.state == "running":
                job.cancel()
            elif job is not None and job.state == "failed":
                self._start_job(self.job_kind, tid, self.job_then)
            elif not self.built:
                self._start_job("track", tid)
            elif self.ghost is None:
                self._start_job("real", tid)

        def _poll_job(self) -> None:
            job = self.job
            if job is not None and job.state == "failed" and not getattr(job, "reported", False):
                job.reported = True
                self._say(f"{job.spec.title} failed: {job.message}")  # the strip has no room
            if job is None or job.state in ("running", "failed"):
                return
            tid, then, self.job, self.job_then = self.job_track, self.job_then, None, None
            if job.state != "done":
                return
            self._say(job.message)
            if tid == self.tracks[self.track_i][0]:
                self._refresh_track()
                if then is not None:
                    self.open_item(then)

        def _open_screen(self, item: str, tid: str, name: str) -> None:
            """Switch the window to the drive, replay or training screen; Esc there comes
            back here."""
            from f1sim.data.track_pipeline import load_track

            if item == "Replay" and self.run is None:
                self._say(f"No runs on {name} yet: train first")
                return
            try:
                track = load_track(tid)
            except FileNotFoundError:
                self._say(f"{name} needs its track data first")
                return
            try:
                if item == "Drive":
                    from f1sim.app.drive import DriveView

                    screen = DriveView(track, ghost_path=self.ghost, on_back=self.come_back)
                elif item == "Train":
                    from f1sim.app.training import TrainingProcess, TrainingView, new_run_dir

                    process = TrainingProcess(
                        tid, new_run_dir(tid, self.runs_root), minutes=self.train_minutes
                    )
                    screen = TrainingView(
                        track, process.run_dir, process=process, on_back=self.come_back
                    )
                else:
                    from f1sim.app.replay import ReplayView, load_trace

                    traces = [load_trace(self.run / "best_lap.npz")]
                    if self.ghost is not None:
                        traces.append(load_trace(self.ghost))
                    screen = ReplayView(track, traces, on_back=self.come_back)
            except (ValueError, OSError) as exc:
                self._say(f"Couldn't open {item.lower()}: {exc}")
                return
            self.light.off()
            self.window.show_view(screen)

        def come_back(self) -> None:
            """Back to the menu (this same view: its textures stay loaded)."""
            self._refresh_track()  # a new run or ghost may have appeared meanwhile
            self.window.show_view(self)

        def _say(self, text: str) -> None:
            self.status = text
            self.t_status.text = text
            self.status_t = time.perf_counter()

        @property
        def view(self) -> HeroLayout:
            """The hero's layout as drawn now: the follow camera's zoom applied while driving."""
            return self.hero.zoomed(self.zoom) if self.drive is not None else self.hero

        # ---------------------------------------------------------------- frame

        def on_update(self, delta_time: float) -> None:
            now = time.perf_counter() - self.t0
            self.light.update(delta_time, ready=now > SPOTLIGHT_DELAY_S + 0.5)
            self.clock += delta_time
            self._poll_job()
            if self.traffic is not None:
                self.traffic.update(min(delta_time, 0.1))
                k = 1.0 - math.exp(-min(delta_time, 0.1) / ZOOM_TAU_S)
                self.zoom += k * (zoom_for_speed(self.drive.speed) - self.zoom)

        def _draw_stage(self) -> None:
            ctx = self.ctx
            st = self.stage
            light = self.light
            cells, outside = light_stage.blend_weights(st.xs, st.ys, light.x, light.y)
            fade = 1.0 - min(outside / STAGE_FADE_OUT_M, 1.0)
            fade = fade * fade * (3.0 - 2.0 * fade)
            p = self._p_stage
            p["u_rect"] = self.hero.rect
            p["u_size"] = (float(self.width), float(self.height))
            p["u_layers"] = tuple(float(iy * len(st.xs) + ix) for iy, ix, _ in cells)
            p["u_weights"] = tuple(float(wt) for _, _, wt in cells)
            p["u_power"] = light.power * fade
            self._stage_tex.use(0)
            self._ambient_tex.use(1)
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            self._unit_quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT

        def _drive_scene(self):
            """This frame's cars on screen (``(anchor, angle, drive, finish)``, the hero
            last), the cursor light (window pixels, height) and the lit lamp heads (window
            pixels and height, colour)."""
            st, lay = self.dstage, self.view
            pose = self.drive.pose()
            reach = 0.5 * hero_drive.CAR_LENGTH_M * lay.px_per_m + lay.px_per_m
            cars = []
            for car, finish in self.traffic.cars():
                anchor, angle = car_in_hero_frame(lay, st.rear_axle_y, pose, car.pose())
                if (
                    -reach < anchor[0] < self.width + reach
                    and -reach < anchor[1] < self.height + reach
                ):
                    cars.append((anchor, angle, car, finish))
            hero_anchor, hero_angle = car_in_hero_frame(lay, st.rear_axle_y, pose, pose)
            cars.append((hero_anchor, hero_angle, self.drive, hero_drive.HERO_FINISH))
            centre, centre_anchor = hero_camera(lay, st.rear_axle_y, pose)
            lx, ly = lay.to_window(self.light.x, self.light.y)
            light = (lx, ly, st.light_z)
            px = hero_drive.world_to_screen(self._lamp_xy, centre, lay.px_per_m, centre_anchor)
            margin = LAMP_REACH_M * lay.px_per_m
            lamps = []
            for lamp, (x, y) in zip(self.lamps, px, strict=True):
                if not (-margin < x < self.width + margin and -margin < y < self.height + margin):
                    continue
                b = hero_drive.lamp_brightness(lamp, self.clock) * LAMP_GAIN
                if b > 1e-3:
                    rgb = hero_drive.LAMP_COLOURS[lamp.colour]
                    d = math.hypot(x - centre_anchor[0], y - centre_anchor[1])
                    head = (float(x), float(y), hero_drive.LAMP_HEIGHT_M)
                    lamps.append((d, head, tuple(b * v for v in rgb)))
            lamps.sort(key=lambda item: item[0])
            return cars, light, [(p, col) for _, p, col in lamps[:MAX_LAMPS]]

        def _floor_uniforms(self, program, cars, light, lamps) -> None:
            shadow_cars = cars[-quality.tier().shadow_cars :]  # the hero is last
            set_uniforms(
                program,
                u_ppm=self.view.px_per_m,
                u_light_px=light,
                u_power=self.light.power * DRIVE_LIGHT,
                u_n_lamps=len(lamps),
                u_lamp_px=_flat([p for p, _ in lamps], MAX_LAMPS, 3),
                u_lamp_col=_flat([c for _, c in lamps], MAX_LAMPS, 3),
                u_n_cars=len(shadow_cars),
                u_car_anchor=_flat([a for a, *_ in shadow_cars], MAX_SHADOW_CARS, 2),
                u_car_rot=_flat(
                    [(math.cos(g), math.sin(g)) for _, g, *_ in shadow_cars], MAX_SHADOW_CARS, 2
                ),
            )

        def _floor_buffer(self):
            """The floor's light buffer: ``quality.tier().floor_scale`` of the window's
            logical size (the light and shadows are soft, so a small buffer upsamples
            cleanly, and its cost doesn't grow with the display's pixel ratio)."""
            scale = quality.tier().floor_scale
            size = (max(int(self.width * scale), 1), max(int(self.height * scale), 1))
            if self._floor_buf is None or self._floor_buf[0] != size:
                ctx = self.ctx
                tex = ctx.texture(
                    size,
                    components=4,
                    dtype="f2",  # the lamps' light runs well above 1
                    filter=(ctx.LINEAR, ctx.LINEAR),
                    wrap_x=ctx.CLAMP_TO_EDGE,
                    wrap_y=ctx.CLAMP_TO_EDGE,
                )
                self._floor_buf = (size, tex, ctx.framebuffer(color_attachments=[tex]))
            return self._floor_buf

        def _draw_floor(self, now: float) -> None:
            """The card's floor, lit by the cursor light and the lamps, with the cars'
            shadows; the spotlight beam on top."""
            self._scene = self._drive_scene()
            size, tex, fbo = self._floor_buffer()
            p = self._p_floor_light
            self._floor_uniforms(p, *self._scene)
            set_uniforms(p, u_size=(float(self.width), float(self.height)))
            for name in ("u_body_alb", "u_body_srf", "u_body_amb"):  # for the shadows
                self._drive_tex[self._units[name]].use(self._units[name])
            with fbo.activate():
                fbo.viewport = (0, 0, *size)
                self._fs_quad.render(p)
            tex.use(FLOOR_LIGHT_UNIT)
            p = self._p_floor
            self._floor_uniforms(p, *self._scene)  # the lamp heads' glow
            set_uniforms(
                p,
                u_size=(float(self.width), float(self.height)),
                u_appear=min(max((now - SPOTLIGHT_DELAY_S) / SPOTLIGHT_FADE_S, 0.0), 1.0),
                u_bg=_rgb(CARD_BG),
                u_lamp_glow_px=LAMP_GLOW_M * self.view.px_per_m,
            )
            self._fs_quad.render(p)

        def _draw_drive(self) -> None:
            ctx = self.ctx
            st, lay = self.dstage, self.view
            cars, light, lamps = self._scene
            (x, y, _), anchor = hero_camera(lay, st.rear_axle_y, self.drive.pose())
            a = math.pi / 2 - hero_drive.VIEW_HEADING
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            t = self._p_track
            set_uniforms(
                t,
                u_car=(x, y),
                u_rot=(math.cos(a), math.sin(a)),
                u_ppm=lay.px_per_m,
                u_anchor=anchor,
                u_size=(float(self.width), float(self.height)),
                u_fade=self.track_fade,
                u_under=TRACK_UNDER_MENU,
            )
            self._track.render(t)
            for unit, tex in enumerate(self._drive_tex):
                tex.use(unit)
            car_lamps = lamps[: quality.tier().car_lamps]
            for anchor, angle, car, finish in cars:
                self._draw_car(anchor, angle, car.steer, finish, light, car_lamps)
            ctx.blend_func = ctx.BLEND_DEFAULT

        def _draw_car(self, anchor, angle, steer, finish, light, lamps) -> None:
            """One car, its rear axle at ``anchor`` (window pixels), turned ``angle`` on
            screen, lit by the cursor light and the lamps in its own frame."""
            st, lay = self.dstage, self.view
            lx, ly = window_to_car(lay, st.rear_axle_y, light[:2], anchor, angle)
            cells, _ = light_stage.blend_weights(st.xs, st.ys, lx, ly)
            gx = min(max(lx, st.xs[0]), st.xs[-1])
            gy = min(max(ly, st.ys[0]), st.ys[-1])
            local = [
                (*window_to_car(lay, st.rear_axle_y, p[:2], anchor, angle), p[2]) for p, _ in lamps
            ]
            set_uniforms(
                self._p_drive,
                u_anchor=anchor,
                u_rot=(math.cos(angle), math.sin(angle)),
                u_ppm=lay.px_per_m,
                u_size=(float(self.width), float(self.height)),
                u_layers=tuple(float(iy * len(st.xs) + ix) for iy, ix, _ in cells),
                u_weights=tuple(float(wt) for _, _, wt in cells),
                u_light=(lx, ly, light[2]),
                u_light_grid=(gx, gy, light[2]),
                u_power=self.light.power * DRIVE_LIGHT,
                u_finish=(finish.albedo_scale, finish.roughness, finish.metallic, finish.coat),
                u_steer=steer,
                u_n_lamps=len(lamps),
                u_lamp_pos=_flat(local, MAX_LAMPS, 3),
                u_lamp_col=_flat([c for _, c in lamps], MAX_LAMPS, 3),
            )
            self._unit_quad.render(self._p_drive)

        def _draw_hero(self) -> None:
            if self.drive is not None:
                self._draw_drive()
                return
            if self.stage is not None:
                self._draw_stage()
                return
            ctx = self.ctx
            lay = self.hero
            light = self.light
            p = self._p_hero
            p["u_rect"] = lay.rect
            p["u_size"] = (float(self.width), float(self.height))
            p["u_light"] = (light.x, light.y, car_top.LIGHT_HEIGHT_M)
            p["u_power"] = light.power
            for tex, unit_i in self._tex.values():
                tex.use(unit_i)
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            self._unit_quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT

        def _draw_backdrop(self) -> None:
            """The faint red and blue backdrop, over the floor and under the track."""
            p = self._p_backdrop
            if p is None:
                return
            self._backdrop_tex.use(0)
            set_uniforms(
                p,
                u_image=0,
                u_size=(float(self.width), float(self.height)),
                u_alpha=BACKDROP_ALPHA,
            )
            self.ctx.blend_func = self.ctx.BLEND_PREMULTIPLIED_ALPHA
            self._fs_quad.render(p)
            self.ctx.blend_func = self.ctx.BLEND_DEFAULT

        def on_draw(self) -> None:

            now = time.perf_counter() - self.t0
            self.clear()
            ctx = self.ctx
            ctx.disable(ctx.BLEND)
            if self.drive is not None:
                self._draw_floor(now)
                ctx.enable(ctx.BLEND)
                self._draw_backdrop()
                self._draw_hero()
                self._draw_text()
                return
            p = self._p_spot
            p["u_size"] = (float(self.width), float(self.height))
            p["u_appear"] = min(max((now - SPOTLIGHT_DELAY_S) / SPOTLIGHT_FADE_S, 0.0), 1.0)
            p["u_bg"] = _rgb(CARD_BG)
            p["u_light_px"] = self.hero.to_window(self.light.x, self.light.y)
            p["u_light_r_px"] = car_top.LIGHT_REACH_M * self.hero.px_per_m
            p["u_light_power"] = self.light.power
            p["u_glow"] = FLOOR_GLOW
            self._fs_quad.render(p)
            ctx.enable(ctx.BLEND)
            self._draw_hero()
            self._draw_text()

        def _texts(self) -> list:
            return [
                self.t_logo,
                self.t_kicker,
                self.t_track_label,
                *self.t_arrows,
                self.t_track,
                self.t_action,
                self.t_action_btn,
                *self.t_kickers,
                *self.t_items,
                *self.t_notes,
                self.t_minutes,
                self.t_footer,
                self.t_status,
            ]

        @staticmethod
        def _panel(rect: Rect, fill=CARD_FILL, line=CARD_LINE) -> None:
            left, bottom, w, h = rect
            arcade.draw_lrbt_rectangle_filled(left, left + w, bottom, bottom + h, fill)
            arcade.draw_lrbt_rectangle_outline(left, left + w, bottom, bottom + h, line, 1)

        @staticmethod
        def _arrow_button(cx: float, cy: float, fill, ink) -> None:
            """A round button with a ↗ arrow, drawn from lines (the fonts have no arrow)."""
            arcade.draw_circle_filled(cx, cy, 15.0, fill)
            a = 5.0
            arcade.draw_line(cx - a, cy - a, cx + a, cy + a, ink, 1.6)
            arcade.draw_line(cx - 1.0, cy + a, cx + a, cy + a, ink, 1.6)
            arcade.draw_line(cx + a, cy + a, cx + a, cy - 1.0, ink, 1.6)

        def _draw_text(self) -> None:
            status_on = time.perf_counter() - self.status_t < 4.0
            self._panel(self.cards.bar, theme.hex_colour("#000000", 0.88))
            self._panel(self.cards.picker)
            self._draw_flag()
            self._draw_strip()
            for i, b in enumerate(self.buttons):
                left, bottom, bw, bh = b.rect
                active = i == self.focus
                disabled = b.item == "Replay" and self.run is None
                lit = active and not disabled
                self._panel(
                    b.rect, theme.ACCENT if lit else CARD_FILL, theme.ACCENT if lit else CARD_LINE
                )
                self._arrow_button(
                    left + bw - 27.0,
                    bottom + bh - 27.0,
                    theme.TEXT_PRIMARY if lit else theme.ACCENT,
                    theme.ACCENT if lit else theme.TEXT_PRIMARY,
                )
                if b.item == "Train":
                    self._draw_train_time()
                self.t_items[i].color = theme.TEXT_TERTIARY if disabled else theme.TEXT_PRIMARY
                self.t_kickers[i].color = theme.white(0.8) if lit else theme.TEXT_SECONDARY
                self.t_notes[i].color = theme.white(0.8) if lit else theme.TEXT_TERTIARY
            with theme.text_batch():
                for t in self._texts():
                    if t is self.t_status and not status_on:
                        continue
                    if t is self.t_footer and status_on:
                        continue
                    t.draw()

    return MenuView


_VIEW_CLASS = None


def MenuView(*args, **kwargs):  # noqa: N802 - a class, built on first use (arcade import)
    """The menu screen (the class is created lazily, so importing this module needs no GL)."""
    global _VIEW_CLASS
    if _VIEW_CLASS is None:
        _VIEW_CLASS = _make_view_class()
    return _VIEW_CLASS(*args, **kwargs)


def run() -> None:
    """Open the app window on the menu; returns when it closes."""
    from f1sim.app import shell

    shell.run(lambda window: MenuView())
