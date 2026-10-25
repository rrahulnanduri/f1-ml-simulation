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

The menu is the home screen of the app window (:mod:`f1sim.app.shell`). For the selected
built track:
- **Drive:** switches the window to the drive screen, with the newest ghost of the track if
  there is one.
- **Replay:** switches to the replay screen: the newest run of the track plus that ghost.
- **Train:** starts ``f1sim train`` on the track in the background and switches to the
  training screen (:mod:`f1sim.app.training`), which shows each new generation driving.
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

from f1sim.app.ui import (
    car_top,
    drive_shaders,
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
DRIVE_CAR_HEIGHT = 0.55
"""The driving car's length as a share of the window height."""
DRIVE_X = 0.72
DRIVE_Y = 0.42
"""The driving car's centre, as a share of the window width and height (a little low, so
more track shows ahead, where the slower cars come from)."""
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
BACKDROP_ALPHA = 0.44
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


def car_in_hero_frame(layout: HeroLayout, rear_axle_y: float, hero_pose, pose):
    """Where a car at ``pose`` (world) shows: its rear axle in window pixels and its rotation
    on screen (radians, anticlockwise), with the hero fixed nose up."""
    anchor = layout.to_window(0.0, rear_axle_y)
    px = hero_drive.world_to_screen(pose[:2], hero_pose, layout.px_per_m, anchor)
    return (float(px[0]), float(px[1])), pose[2] - hero_pose[2]


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
            self.tracks = built_tracks() or [("silverstone", "Silverstone Circuit")]
            ids = [t[0] for t in self.tracks]
            self.track_i = ids.index("silverstone") if "silverstone" in ids else 0
            self.focus = 0
            self.hover: int | None = None
            self.light = Light()
            self.t0 = time.perf_counter()
            self.clock = 0.0  # seconds of animation (the lamps' flashes)
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
            self.t_kicker = theme.micro("F1SIM  ·  NEUROEVOLUTION", theme.ACCENT)
            self.t_head = [
                theme.make_text("LEARN TO", 60, HEADLINE_TOP, face="headline"),
                theme.make_text("RACE.", 60, HEADLINE_BOTTOM, face="headline"),
            ]
            self.t_body = theme.make_text(
                "AI drivers teach themselves to lap real Formula 1 circuits. Pick a track, "
                "drive it yourself against a real driver's ghost, replay the fastest AI lap, "
                "or start training a new generation.",
                16,
                BODY_TEXT,
                anchor_y="top",
                multiline=True,
                width=440,
            )
            self.t_track_label = theme.micro("Track", theme.TEXT_SECONDARY)
            self.t_track = theme.make_text("", 15, theme.TEXT_PRIMARY, face="headline")
            self.t_arrows = [
                theme.make_text("‹", 22, theme.TEXT_SECONDARY, face="medium", anchor_x="center"),
                theme.make_text("›", 22, theme.TEXT_SECONDARY, face="medium", anchor_x="center"),
            ]
            self.t_items = [
                theme.make_text(i.upper(), 13, theme.TEXT_PRIMARY, face="micro", tracking_px=1.5)
                for i in ITEMS
            ]
            self.t_notes = [
                theme.make_text("", 11, theme.TEXT_TERTIARY, face="medium") for _ in ITEMS
            ]
            self.t_footer = theme.micro(
                "↑↓ select   ·   ←→ track   ·   enter open   ·   esc quit", theme.TEXT_TERTIARY
            )
            self.t_status = theme.make_text("", 12, theme.TEXT_SECONDARY, face="medium")

        def _refresh_track(self) -> None:
            from f1sim import paths

            tid, name = self.tracks[self.track_i]
            self.t_track.text = name.upper()
            self.run = latest_run(tid, self.runs_root)
            self.ghost = latest_ghost(tid, paths.ghosts_dir())
            notes = {
                "Drive": f"vs ghost {self.ghost.stem.split('_')[-1]}" if self.ghost else "solo",
                "Replay": self.run.name if self.run else "no runs yet",
                "Train": "watch it learn",
                "Quit": "",
            }
            for t, item in zip(self.t_notes, ITEMS, strict=True):
                t.text = notes[item]

        # --------------------------------------------------------------- layout

        def on_resize(self, width: int, height: int) -> None:
            super().on_resize(width, height)
            w, h = float(width), float(height)
            # Text is laid out on whole pixels (a glyph between pixels is resampled: blurred)
            self.left = float(round(max(48.0, 0.07 * w)))
            col_w = float(round(min(460.0, 0.42 * w - self.left)))
            self.col_w = col_w
            self.t_body.width = col_w
            body_h = float(round(self.t_body.content_height))
            # Column height, top of the kicker to the bottom of the last button
            total = 62.0 + 66.0 + 34.0 + body_h + 40.0 + 34.0 + 30.0 + 4 * 48.0 - 8.0
            footer_room = 70.0
            y = float(round(min(0.5 * (h + footer_room) + 0.5 * total, h - 48.0)))
            self.t_kicker.position = (self.left, y)
            y -= 62.0
            for k, t in enumerate(self.t_head):
                t.position = (self.left, y)
                y -= 66.0 if k == 0 else 34.0
            self.t_body.position = (self.left, y)
            y -= body_h + 40.0
            self.t_track_label.position = (self.left, y)
            y -= 34.0
            self.track_row_y = y
            self.t_arrows[0].position = (self.left + 8.0, y - 2.0)
            self.t_track.position = (self.left + 32.0, y)
            self.t_arrows[1].position = (self.left + col_w - 8.0, y - 2.0)
            y -= 30.0
            self.buttons = []
            for i, (t, note) in enumerate(zip(self.t_items, self.t_notes, strict=True)):
                rect = (self.left, y - 40.0, col_w, 40.0)
                self.buttons.append(Button(ITEMS[i], rect))
                t.position = (self.left + 18.0, y - 25.0)
                note.position = (self.left + col_w - 16.0, y - 25.0)
                note.anchor_x = "right"
                y -= 48.0
            self.t_footer.position = (self.left, 28.0)
            self.t_status.position = (self.left, 52.0)
            if self.drive is not None:
                self.hero = drive_layout(w, h, self.dstage.size_m, self.dstage.rear_axle_y)
                self.traffic.view_m = view_radius_m(w, h, self.hero, self.dstage.rear_axle_y)
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
            self.light.aim(*self.hero.to_car(x, y))
            self.hover = self._button_at(x, y)
            if self.hover is not None:
                self.focus = self.hover

        def on_mouse_enter(self, x: int, y: int) -> None:
            self.light.aim(*self.hero.to_car(x, y))

        def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
            self.light.scroll(scroll_y)

        def on_mouse_leave(self, x: int, y: int) -> None:
            self.light.off()
            self.hover = None

        def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
            i = self._button_at(x, y)
            if i is not None:
                self.focus = i
                self.open_item(ITEMS[i])
                return
            if abs(y - self.track_row_y) < 18:
                if x < self.left + 24:
                    self.change_track(-1)
                elif x > self.left + self.col_w - 24:
                    self.change_track(1)

        def on_key_press(self, symbol: int, modifiers: int) -> None:

            k = arcade.key
            if symbol == k.ESCAPE:
                self.close()
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
            self._open_screen(item, tid, name)

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
                self._say(f"{name} isn't built: f1sim build-track {tid}")
                return
            try:
                if item == "Drive":
                    from f1sim.app.drive import DriveView

                    screen = DriveView(track, ghost_path=self.ghost, on_back=self.come_back)
                elif item == "Train":
                    from f1sim.app.training import TrainingProcess, TrainingView, new_run_dir

                    process = TrainingProcess(tid, new_run_dir(tid, self.runs_root))
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

        # ---------------------------------------------------------------- frame

        def on_update(self, delta_time: float) -> None:
            now = time.perf_counter() - self.t0
            self.light.update(delta_time, ready=now > SPOTLIGHT_DELAY_S + 0.5)
            self.clock += delta_time
            if self.traffic is not None:
                self.traffic.update(min(delta_time, 0.1))

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
            st, lay = self.dstage, self.hero
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
            hero_anchor = lay.to_window(0.0, st.rear_axle_y)
            cars.append((hero_anchor, 0.0, self.drive, hero_drive.HERO_FINISH))
            lx, ly = lay.to_window(self.light.x, self.light.y)
            light = (lx, ly, st.light_z)
            px = hero_drive.world_to_screen(self._lamp_xy, pose, lay.px_per_m, hero_anchor)
            margin = LAMP_REACH_M * lay.px_per_m
            lamps = []
            for lamp, (x, y) in zip(self.lamps, px, strict=True):
                if not (-margin < x < self.width + margin and -margin < y < self.height + margin):
                    continue
                b = hero_drive.lamp_brightness(lamp, self.clock) * LAMP_GAIN
                if b > 1e-3:
                    rgb = hero_drive.LAMP_COLOURS[lamp.colour]
                    d = math.hypot(x - hero_anchor[0], y - hero_anchor[1])
                    head = (float(x), float(y), hero_drive.LAMP_HEIGHT_M)
                    lamps.append((d, head, tuple(b * v for v in rgb)))
            lamps.sort(key=lambda item: item[0])
            return cars, light, [(p, col) for _, p, col in lamps[:MAX_LAMPS]]

        def _floor_uniforms(self, program, cars, light, lamps) -> None:
            shadow_cars = cars[-quality.tier().shadow_cars :]  # the hero is last
            set_uniforms(
                program,
                u_ppm=self.hero.px_per_m,
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
                u_lamp_glow_px=LAMP_GLOW_M * self.hero.px_per_m,
            )
            self._fs_quad.render(p)

        def _draw_drive(self) -> None:
            ctx = self.ctx
            st, lay = self.dstage, self.hero
            cars, light, lamps = self._scene
            x, y, heading = self.drive.pose()
            a = math.pi / 2 - heading
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            t = self._p_track
            set_uniforms(
                t,
                u_car=(x, y),
                u_rot=(math.cos(a), math.sin(a)),
                u_ppm=lay.px_per_m,
                u_anchor=lay.to_window(0.0, st.rear_axle_y),
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
            st, lay = self.dstage, self.hero
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
                self.t_kicker,
                *self.t_head,
                self.t_body,
                self.t_track_label,
                *self.t_arrows,
                self.t_track,
                *self.t_items,
                *self.t_notes,
                self.t_footer,
                self.t_status,
            ]

        def _draw_text(self) -> None:
            for i, b in enumerate(self.buttons):
                left, bottom, bw, bh = b.rect
                active = i == self.focus
                disabled = b.item == "Replay" and self.run is None
                if active and not disabled:
                    arcade.draw_lrbt_rectangle_filled(
                        left, left + bw, bottom, bottom + bh, theme.ACCENT
                    )
                else:
                    arcade.draw_lrbt_rectangle_filled(
                        left, left + bw, bottom, bottom + bh, theme.white(0.03)
                    )
                    arcade.draw_lrbt_rectangle_outline(
                        left, left + bw, bottom, bottom + bh, theme.white(0.20), 1
                    )
                self.t_items[i].color = theme.TEXT_TERTIARY if disabled else theme.TEXT_PRIMARY
                self.t_notes[i].color = (
                    theme.white(0.75) if active and not disabled else theme.TEXT_TERTIARY
                )
            with theme.text_batch():
                for t in self._texts():
                    if t is not self.t_status or time.perf_counter() - self.status_t < 4.0:
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
