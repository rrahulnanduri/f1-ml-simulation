"""Main menu: a top-down F1 car lit by the cursor, next to the menu.

``f1sim menu`` opens it. The layout follows the "interactive" hero card: a near-black card,
a soft spotlight beam fading in from the top-left, a headline, paragraph and menu on the left,
and the hero on the right third.

**The hero.** A generic, unbranded F1 car seen from above (:mod:`f1sim.app.ui.car_top`), nose
up, filling the right third from the nose to just behind the driver; the rest runs off the
bottom of the window. The car doesn't move. The cursor is a light: a point light above the
cursor shades the car per pixel from its normal map (diffuse plus a glossy highlight, so
reflections slide over the bodywork as you move), and lights the floor around it faintly.
Away from the light the car is only just visible. The light fades in after the spotlight
beam, and out when the cursor leaves the window.

The menu opens the existing tools for the selected built track, each as its own process:
- **Drive:** ``f1sim drive``, with the newest ghost of the track if there is one.
- **Replay:** ``f1sim replay``, the newest run of the track plus that ghost.
- **Train:** ``f1sim train``, in a new console window.
- **Quit.**

Controls: mouse (hover, click), Up/Down to pick, Enter to open, Left/Right or the arrows to
change track, Esc to quit.

The testable logic (track and run discovery, the commands, the hero layout and the
screen-to-car mapping, the light's easing) is pure Python/numpy. :class:`MenuWindow` only
draws and maps input.

**Live reload.** ``scripts/dev_menu.py`` restarts this window whenever a source file changes.
With ``F1SIM_MENU_STATE`` set to a file, the window saves its position and size there and
restores them on start, so each restart opens where the last one was.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1sim.app.ui import car_top, theme

STATE_ENV = "F1SIM_MENU_STATE"
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
INTRO_LIGHT = (0.35, 1.9)
"""Where the light sits (car metres) before the cursor first moves: over the cockpit."""
FLOOR_GLOW = 0.05
"""Brightness the cursor light adds to the floor (the card) at its centre."""

ITEMS = ("Drive", "Replay", "Train", "Quit")


# --------------------------------------------------------------------------- discovery


def built_tracks() -> list[tuple[str, str]]:
    """``(track_id, circuit name)`` of every built track, by id."""
    from f1sim.data import catalog
    from f1sim.data.track_pipeline import track_path

    return [
        (e.id, e.name)
        for e in sorted(catalog.CATALOG.values(), key=lambda e: e.id)
        if track_path(e.id).exists()
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
    """The car in the right third, nose up: ``HERO_VISIBLE_M`` of it fills the height below a
    small margin (the rest is cut by the window's bottom edge), unless that would make it
    wider than ``HERO_MAX_WIDTH`` of the window; centred on the right third."""
    width_m, length_m = size_m
    margin = HERO_TOP_MARGIN * height
    ppm = (height - margin) / HERO_VISIBLE_M
    ppm = min(ppm, HERO_MAX_WIDTH * width / (2.0 * car_top.HALF_WIDTH_M))
    centre_x = width * 5.0 / 6.0
    return HeroLayout(
        left=centre_x - 0.5 * width_m * ppm,
        top=height - margin,
        px_per_m=ppm,
        width_m=width_m,
        length_m=length_m,
    )


@dataclass
class Light:
    """The cursor light: plan position (car metres) easing towards a target, and a power
    0..1 easing towards on (cursor in the window) or off."""

    x: float = INTRO_LIGHT[0]
    y: float = INTRO_LIGHT[1]
    target_x: float = INTRO_LIGHT[0]
    target_y: float = INTRO_LIGHT[1]
    power: float = 0.0
    on: bool = True

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
        goal = 1.0 if (self.on and ready) else 0.0
        self.power += (1.0 - math.exp(-dt / POWER_TAU_S)) * (goal - self.power)


# ------------------------------------------------------------------------------ shaders

_FS_VS = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_vert, 0.0, 1.0); }
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

_HERO_FS = """
#version 330
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_material;  // r: specular, g: gloss, b: height / u_height_scale
uniform vec2 u_size_m;         // map width and length, metres
uniform float u_half_width_m;
uniform float u_height_scale;
uniform vec3 u_light;          // light position, car metres (x right, y back, z up)
uniform float u_power;
uniform float u_core;
uniform float u_reach;
uniform float u_ambient;
uniform float u_diffuse;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec4 alb = texture(u_albedo, v_uv);
    if (alb.a <= 0.0) discard;
    vec3 n = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    vec3 m = texture(u_material, v_uv).rgb;
    // This pixel in car metres (row 0 of the maps, the nose, is at the top: v = 1)
    vec2 pos = vec2(v_uv.x * u_size_m.x - u_half_width_m, (1.0 - v_uv.y) * u_size_m.y);
    float h = m.b * u_height_scale;
    // Light vector in the normal map's frame: x right, y towards the nose, z up
    vec3 Lv = vec3(u_light.x - pos.x, -(u_light.y - pos.y), u_light.z - h);
    vec3 L = normalize(Lv);
    float ndl = max(dot(n, L), 0.0);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float spec = m.r * pow(max(dot(n, H), 0.0), 8.0 + 120.0 * m.g);
    float r = length(u_light.xy - pos);
    float att = u_power / (1.0 + (r / u_core) * (r / u_core)) * (1.0 - smoothstep(0.0, u_reach, r));
    vec3 c = alb.rgb * (u_ambient + u_diffuse * ndl * att) + vec3(spec * att);
    c = c / (1.0 + 0.2 * c);  // soft shoulder on the highlights
    frag = vec4(c * alb.a, alb.a);  // premultiplied
}
"""


# ------------------------------------------------------------------------------ window


def _rgb(c: theme.Colour) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def _load_state() -> dict | None:
    path = os.environ.get(STATE_ENV)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_state(state: dict) -> None:
    path = os.environ.get(STATE_ENV)
    if path:
        try:
            Path(path).write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            pass


@dataclass
class Button:
    item: str
    rect: tuple[float, float, float, float]  # left, bottom, width, height


HEIGHT_SCALE_M = 1.2
"""Heights are stored as ``height / HEIGHT_SCALE_M`` in the material texture's blue channel."""


def _texture_bytes(maps: car_top.CarMaps) -> dict[str, tuple[int, bytes]]:
    """The maps as RGBA8 texture data, rows flipped (GL's first row is the bottom)."""

    def rgba(a):
        a = np.clip(a, 0.0, 1.0)
        if a.shape[-1] == 3:
            a = np.concatenate([a, np.ones((*a.shape[:2], 1), np.float32)], axis=-1)
        return np.ascontiguousarray((np.flipud(a) * 255.0 + 0.5).astype(np.uint8)).tobytes()

    normal = 0.5 + 0.5 * maps.normal
    material = np.concatenate([maps.material, (maps.height / HEIGHT_SCALE_M)[..., None]], axis=-1)
    return {"albedo": rgba(maps.albedo), "normal": rgba(normal), "material": rgba(material)}


def _make_window_class():
    import arcade
    from arcade.gl import BufferDescription, geometry

    class MenuWindow(arcade.Window):
        """The main menu; see the module docstring."""

        def __init__(self, width: int = 1280, height: int = 760, runs_root: Path = Path("runs")):
            state = _load_state() or {}
            super().__init__(
                int(state.get("w", width)),
                int(state.get("h", height)),
                "f1sim",
                resizable=True,
                update_rate=1 / 120,
                draw_rate=1 / 120,
                vsync=True,
            )
            if "x" in state and "y" in state:
                self.set_location(int(state["x"]), int(state["y"]))
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
            self.status = ""
            self.status_t = -10.0
            self.maps = car_top.build_maps()
            self._init_gl()
            self._init_text()
            self._refresh_track()
            self.on_resize(self.width, self.height)

        # ---------------------------------------------------------------- setup

        def _init_gl(self) -> None:
            ctx = self.ctx
            self._fs_quad = geometry.quad_2d_fs()
            self._p_spot = ctx.program(vertex_shader=_FS_VS, fragment_shader=_SPOT_FS)
            self._p_hero = ctx.program(vertex_shader=_HERO_VS, fragment_shader=_HERO_FS)
            unit = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1], dtype="f4")
            self._unit_quad = ctx.geometry(
                [BufferDescription(ctx.buffer(data=unit.tobytes()), "2f 2f", ["in_vert", "in_uv"])],
                mode=ctx.TRIANGLE_STRIP,
            )
            h, w = self.maps.height.shape
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
            p["u_albedo"], p["u_normal"], p["u_material"] = 0, 1, 2
            p["u_size_m"] = self.maps.size_m
            p["u_half_width_m"] = car_top.HALF_WIDTH_M
            p["u_height_scale"] = HEIGHT_SCALE_M
            p["u_core"] = car_top.LIGHT_CORE_M
            p["u_reach"] = car_top.LIGHT_REACH_M
            p["u_ambient"] = car_top.AMBIENT
            p["u_diffuse"] = car_top.DIFFUSE_GAIN

        def _init_text(self) -> None:
            import arcade

            f = theme.FACES
            self.t_kicker = theme.micro("F1SIM  ·  NEUROEVOLUTION", theme.ACCENT)
            self.t_head = [
                theme.make_text("LEARN TO", 60, HEADLINE_TOP, face="headline"),
                theme.make_text("RACE.", 60, HEADLINE_BOTTOM, face="headline"),
            ]
            self.t_body = arcade.Text(
                "AI drivers teach themselves to lap real Formula 1 circuits. Pick a track, "
                "drive it yourself against a real driver's ghost, replay the fastest AI lap, "
                "or start training a new generation.",
                0,
                0,
                BODY_TEXT,
                theme.px_to_pt(16),
                font_name=f["medium"].names,
                multiline=True,
                width=440,
                anchor_y="top",
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
                "Train": "opens a console",
                "Quit": "",
            }
            for t, item in zip(self.t_notes, ITEMS, strict=True):
                t.text = notes[item]

        # --------------------------------------------------------------- layout

        def on_resize(self, width: int, height: int) -> None:
            super().on_resize(width, height)
            w, h = float(width), float(height)
            self.left = max(48.0, 0.07 * w)
            col_w = min(460.0, 0.42 * w - self.left)
            self.col_w = col_w
            self.t_body.width = int(col_w)
            body_h = float(self.t_body.content_height)
            # Column height, top of the kicker to the bottom of the last button
            total = 62.0 + 66.0 + 34.0 + body_h + 40.0 + 34.0 + 30.0 + 4 * 48.0 - 8.0
            footer_room = 70.0
            y = min(0.5 * (h + footer_room) + 0.5 * total, h - 48.0)
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
            self.hero = hero_layout(w, h, self.maps.size_m)

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
            import arcade

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
            argv = command(item, tid, run=self.run, ghost=self.ghost)
            if argv is None:
                self._say(f"No runs on {name} yet: train first")
                return
            flags = 0
            if item == "Train" and sys.platform == "win32":
                flags = subprocess.CREATE_NEW_CONSOLE
            try:
                subprocess.Popen(argv, creationflags=flags)
            except OSError as exc:
                self._say(f"Couldn't start {item.lower()}: {exc}")
                return
            self._say(f"Opening {item.lower()} on {name}")

        def _say(self, text: str) -> None:
            self.status = text
            self.t_status.text = text
            self.status_t = time.perf_counter()

        # ----------------------------------------------------------- persistence

        def on_move(self, x: int, y: int) -> None:
            self._persist()

        def _persist(self) -> None:
            x, y = self.get_location()
            _save_state({"x": x, "y": y, "w": self.width, "h": self.height})

        def on_close(self) -> None:
            self._persist()
            super().on_close()

        # ---------------------------------------------------------------- frame

        def on_update(self, delta_time: float) -> None:
            now = time.perf_counter() - self.t0
            self.light.update(delta_time, ready=now > SPOTLIGHT_DELAY_S + 0.5)

        def _draw_hero(self) -> None:
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

        def on_draw(self) -> None:
            import arcade

            now = time.perf_counter() - self.t0
            self.clear()
            ctx = self.ctx
            ctx.disable(ctx.BLEND)
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

            self.t_kicker.draw()
            for t in self.t_head:
                t.draw()
            self.t_body.draw()
            self.t_track_label.draw()
            for t in self.t_arrows:
                t.draw()
            self.t_track.draw()
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
                t = self.t_items[i]
                t.color = theme.TEXT_TERTIARY if disabled else theme.TEXT_PRIMARY
                t.draw()
                n = self.t_notes[i]
                n.color = theme.white(0.75) if active and not disabled else theme.TEXT_TERTIARY
                n.draw()
            if time.perf_counter() - self.status_t < 4.0:
                self.t_status.draw()
            self.t_footer.draw()

    return MenuWindow


_WINDOW_CLASS = None


def MenuWindow(*args, **kwargs):  # noqa: N802 - a class, built on first use (arcade import)
    """The menu window class (created lazily so importing this module needs no display)."""
    global _WINDOW_CLASS
    if _WINDOW_CLASS is None:
        _WINDOW_CLASS = _make_window_class()
    return _WINDOW_CLASS(*args, **kwargs)


def run() -> None:
    """Open the main menu; returns when it closes."""
    import arcade

    MenuWindow()
    arcade.run()
