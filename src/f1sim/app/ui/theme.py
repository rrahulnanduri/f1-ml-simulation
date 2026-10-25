"""The viewers' design language: colours, type, sizes, team colours and liveries, fonts.

A dark, single-accent look: a deep charcoal scene, solid near-black cards with a white
hairline, one red accent (``ACCENT``) for highlights, white text at 100 % / 60 %, wide bold
UPPERCASE headlines (Archivo Expanded Bold), 10 px UPPERCASE "micro" labels with 1 px tracking,
and JetBrains Mono for tabular numbers.

Everything here is constants and pure functions except :func:`load_fonts`, :func:`make_text`
and :class:`SharpText`, which touch arcade/pyglet only when called (never on import). All text
is a :class:`SharpText`: sharp on HiDPI displays.

Colours are ``(r, g, b, a)`` tuples of ints 0-255. Sizes are screen pixels at 1x scale; arcade
text sizes are points, so :func:`px_to_pt` converts (pyglet renders at 96 dpi: 1 pt = 4/3 px).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

Colour = tuple[int, int, int, int]


def hex_colour(code: str, alpha: float = 1.0) -> Colour:
    """``"#RRGGBB"`` and an opacity 0-1 -> ``(r, g, b, a)``."""
    code = code.lstrip("#")
    if len(code) != 6:
        raise ValueError(f"expected #RRGGBB, got {code!r}")
    r, g, b = (int(code[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, int(255 * min(max(alpha, 0.0), 1.0) + 0.5))


def white(alpha: float) -> Colour:
    return hex_colour("#FFFFFF", alpha)


def with_alpha(colour: Colour, alpha: float) -> Colour:
    """``colour`` with its opacity replaced by ``alpha`` (0-1)."""
    return (colour[0], colour[1], colour[2], int(255 * min(max(alpha, 0.0), 1.0) + 0.5))


# ----------------------------------------------------------------------------- accents

ACCENT = hex_colour("#EB0D3F")
"""The single accent: self/AI highlight, "behind" gaps, active page dot, categories, buttons."""
GREY = hex_colour("#3E3E3E")
"""Inactive elements (page dots, unset sectors)."""

# ------------------------------------------------------------------------------ scene

BACKGROUND = hex_colour("#131313")
VIGNETTE = 0.22
"""Darkening of the window corners by the neutral vignette (0 = none)."""
ROAD = hex_colour("#1E1F22")
WHITE_LINE = white(0.70)
WHITE_LINE_PX = 1.5
"""The white line (the track edge) is this wide on screen at every zoom."""
KERB_RED = ACCENT
KERB_WHITE = hex_colour("#F2F2F2")
RUNOFF = hex_colour("#2A2B2F")
KERB_MIN_EXTRA_M = 0.3
"""Kerbs only where the track limit is more than this beyond the white line."""
KERB_MAX_WIDTH_M = 1.5
KERB_BLOCK_M = 3.0
"""Length of one red or white kerb block along the white line (adjusted per kerb)."""
CORNER_KERB_WIDTH_M = 1.0
"""Width of the kerbs placed from corner geometry where drivers don't run past the line."""
CORNER_MAX_RADIUS_M = 300.0
"""A corner is a stretch tighter than this (smoothed centre-line curvature)..."""
CORNER_MIN_TURN_DEG = 15.0
"""... that turns at least this much in total."""
CORNER_SMOOTH_M = 15.0
"""Gaussian smoothing of the curvature before finding corners, m."""
APEX_FRACTION = 0.6
"""The apex kerb covers the part of a corner tighter than this share of its peak curvature."""
APEX_PAD_M = 5.0
EXIT_START = 0.4
"""The exit kerb starts this share of the way from the apex to the corner's end..."""
EXIT_EXTRA_M = 15.0
"""... and runs this far past the end."""
CORNER_KERB_MIN_M = 20.0
"""Shortest apex or exit kerb, m."""
KERB_MIN_PPM = 1.2
"""Below this zoom (pixels per metre) the kerbs are drawn as run-off: 3 m blocks would be
smaller than ~4 px and only flicker."""
CHEQUER_M = 1.0
"""Size of a start/finish chequer square."""
SECTOR_TICK = ACCENT
SECTOR_TICK_PX = (2.0, 10.0)
"""Width and length of a sector tick across the white line, screen pixels."""

# ------------------------------------------------------------------------------- cars

CAR_LENGTH_M = 5.6
CAR_WIDTH_M = 2.0
CAR_MIN_LENGTH_PX = 24.0
"""Zoomed out, a car sprite is at least this long (in whole multiples of its pixels)."""
SELF = ACCENT
"""Colour of the self/AI car in the tower, the charts and its label."""
CAR_LABEL = white(0.70)
CAR_OUTLINE = hex_colour("#E8EAED", 0.55)
"""The 1 px outline around ghost cars."""


@dataclass(frozen=True)
class Livery:
    """A car's paint: ``primary`` (body), ``secondary`` (wings, sidepod sides) and
    ``accent`` (endplates, stripes)."""

    primary: Colour
    secondary: Colour
    accent: Colour


SELF_LIVERY = Livery(hex_colour("#A9ADB2"), hex_colour("#0E0E10"), ACCENT)
"""The self/AI car: titanium and black with red accents."""

LIVERIES: dict[str, Livery] = {
    "red bull": Livery(hex_colour("#1E2A5E"), hex_colour("#D8192B"), hex_colour("#FFC906")),
    "mercedes": Livery(hex_colour("#C3C8CD"), hex_colour("#111214"), hex_colour("#27F4D2")),
    "ferrari": Livery(hex_colour("#D40000"), hex_colour("#1A1A1A"), hex_colour("#FFE600")),
    "mclaren": Livery(hex_colour("#FF8000"), hex_colour("#151515"), hex_colour("#47C7FC")),
    "aston martin": Livery(hex_colour("#0B5E4B"), hex_colour("#CEDC00"), hex_colour("#CEDC00")),
    "alpine": Livery(hex_colour("#0A72C2"), hex_colour("#121418"), hex_colour("#FD4BC7")),
    "williams": Livery(hex_colour("#0A2A5E"), hex_colour("#00A3E0"), hex_colour("#FFFFFF")),
    "rb": Livery(hex_colour("#E9ECF2"), hex_colour("#1634CB"), hex_colour("#D8192B")),
    "sauber": Livery(hex_colour("#1A1C1E"), hex_colour("#52E252"), hex_colour("#52E252")),
    "haas": Livery(hex_colour("#ECEDEE"), hex_colour("#1A1A1A"), hex_colour("#E6002B")),
}
NEUTRAL_LIVERY = Livery(hex_colour("#8E929A"), hex_colour("#1A1B1E"), hex_colour("#D9DBDE"))
"""A ghost without a known team."""

FIELD_SHADES = ((0.80, 0.95), (0.45, 1.00), (0.90, 0.72))
"""``(saturation, value)`` levels :func:`field_liveries` cycles through: vivid, pastel, deep."""
GOLDEN_TURN = 0.6180339887498949
"""Hue step between consecutive field cars (a golden-ratio turn: every hue lands as far as
possible from all the earlier ones)."""


def _hsv(h: float, s: float, v: float) -> Colour:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255)


def field_liveries(n: int) -> list[Livery]:
    """``n`` liveries told apart by colour (the training view's population): hues a golden
    turn apart, cycling through :data:`FIELD_SHADES` so neighbours in hue differ in shade
    too. Wings are near-black carbon, endplates and stripes the opposite hue."""
    out = []
    for i in range(int(n)):
        hue = 0.03 + i * GOLDEN_TURN
        sat, val = FIELD_SHADES[i % len(FIELD_SHADES)]
        out.append(Livery(_hsv(hue, sat, val), hex_colour("#141518"), _hsv(hue + 0.5, 0.55, 1.0)))
    return out


# ------------------------------------------------------------------------------- text

TEXT_PRIMARY = white(1.0)
TEXT_SECONDARY = white(0.60)
TEXT_TERTIARY = white(0.36)
TEXT_LABEL = TEXT_PRIMARY
"""Metadata micro labels are white; category micro labels are ``ACCENT``."""
BEHIND = ACCENT
AHEAD = hex_colour("#2FD27A")
SELF_GAP = ACCENT
OFF_TRACK = ACCENT

SECTOR_PURPLE = hex_colour("#A97CFF")
SECTOR_GREEN = AHEAD
SECTOR_YELLOW = hex_colour("#E8C547")
SECTOR_UNSET = GREY
SECTOR_COLOURS = {
    "purple": SECTOR_PURPLE,
    "green": SECTOR_GREEN,
    "yellow": SECTOR_YELLOW,
    "unset": SECTOR_UNSET,
}

SEPARATOR = white(0.08)
SELF_ROW = white(0.05)
"""Faint fill behind the self row of the timing tower."""
CHART_REF = white(0.40)
CHART_SELF = ACCENT
CHART_MARKER = white(0.45)
CHART_GRID = white(0.06)
GAP_TRACK = white(0.10)
DOT_ACTIVE = ACCENT
DOT_INACTIVE = GREY
LINK_UNDERLINE = white(0.28)

MICRO_PX = 10
MICRO_TRACKING_PX = 1.0
HEADLINE_PX = 28
HEADLINE_FADE = (1.0, 0.42, 0.22)
"""Opacity of the headline's first, second and third line."""
WORDMARK_PX = 14
NAME_PX = 12
TIME_PX = 12
GAP_PX = 30
SPEED_PX = 40
SECTOR_TIME_PX = 13
GAP_PAGE_PX = 15
CLOCK_PX = 13

# ------------------------------------------------------------------------------ cards

CARD_RADIUS = 8.0
CARD_FILL = 0.88
"""Opacity of the black card fill over the (blurred) scene."""
CARD_BORDER = white(0.20)
BLUR_SIGMA_PX = 6.0
CARD_FALLBACK = (0, 0, 0, int(255 * CARD_FILL + 0.5))
BUTTON_RADIUS = 10.0
"""The red ↗ buttons in card corners."""

NO_GLASS_ENV = "F1SIM_NO_GLASS"


def glass_disabled_by_env() -> bool:
    """``F1SIM_NO_GLASS`` set to anything but ``""``/``0`` forces the flat fallback panels."""
    return os.environ.get(NO_GLASS_ENV, "").strip() not in ("", "0")


# ------------------------------------------------------------------------------ teams

TEAM_COLOURS: dict[str, Colour] = {
    "red bull": hex_colour("#3671C6"),
    "mercedes": hex_colour("#27F4D2"),
    "ferrari": hex_colour("#E8002D"),
    "mclaren": hex_colour("#FF8000"),
    "aston martin": hex_colour("#229971"),
    "alpine": hex_colour("#0093CC"),
    "williams": hex_colour("#64C4FF"),
    "rb": hex_colour("#6692FF"),
    "sauber": hex_colour("#52E252"),
    "haas": hex_colour("#B6BABD"),
}

# Matched in order against the lower-cased team name: the specific names come first, so
# "Racing Bulls" isn't taken for Red Bull and "Red Bull Racing" isn't taken for RB.
_TEAM_ALIASES: tuple[tuple[str, str], ...] = (
    ("racing bulls", "rb"),
    ("visa cash app", "rb"),
    ("alphatauri", "rb"),
    ("alpha tauri", "rb"),
    ("toro rosso", "rb"),
    ("red bull", "red bull"),
    ("mercedes", "mercedes"),
    ("ferrari", "ferrari"),
    ("mclaren", "mclaren"),
    ("aston martin", "aston martin"),
    ("racing point", "aston martin"),
    ("alpine", "alpine"),
    ("renault", "alpine"),
    ("williams", "williams"),
    ("kick", "sauber"),
    ("sauber", "sauber"),
    ("alfa romeo", "sauber"),
    ("audi", "sauber"),
    ("haas", "haas"),
)

MUTED_PALETTE: tuple[Colour, ...] = (
    hex_colour("#8E929A"),
    hex_colour("#6E727A"),
    hex_colour("#B4B7BE"),
    hex_colour("#5C6068"),
)
"""Greys for traces without a team, other than the red "self"."""


def team_key(team: str | None) -> str | None:
    """The key of :data:`TEAM_COLOURS` / :data:`LIVERIES` for a team name, or ``None``."""
    name = (team or "").strip().lower()
    if not name:
        return None
    if name == "rb":
        return "rb"
    for alias, key in _TEAM_ALIASES:
        if alias in name:
            return key
    return None


def team_colour(team: str | None) -> Colour | None:
    """The team's colour, or ``None`` for an empty or unknown team name."""
    key = team_key(team)
    return None if key is None else TEAM_COLOURS[key]


def livery(team: str | None, is_self: bool = False) -> Livery:
    """The car paint: :data:`SELF_LIVERY` for the self/AI car, else the team's livery
    (:data:`NEUTRAL_LIVERY` for an unknown team)."""
    if is_self:
        return SELF_LIVERY
    key = team_key(team)
    return NEUTRAL_LIVERY if key is None else LIVERIES[key]


def trace_colours(teams: list[str | None], is_ghost: list[bool]) -> list[Colour]:
    """One colour per trace: its team colour; else the accent red for the first trace that
    isn't a ghost (the "self" car) and greys from :data:`MUTED_PALETTE` for the rest."""
    out: list[Colour] = []
    self_taken = False
    grey = 0
    for team, ghost in zip(teams, is_ghost, strict=True):
        colour = team_colour(team)
        if colour is None:
            if not ghost and not self_taken:
                colour, self_taken = SELF, True
            else:
                colour = MUTED_PALETTE[grey % len(MUTED_PALETTE)]
                grey += 1
        out.append(colour)
    return out


DRIVER_SURNAMES: dict[str, str] = {
    "VER": "Verstappen",
    "PER": "Pérez",
    "HAM": "Hamilton",
    "RUS": "Russell",
    "LEC": "Leclerc",
    "SAI": "Sainz",
    "NOR": "Norris",
    "PIA": "Piastri",
    "ALO": "Alonso",
    "STR": "Stroll",
    "GAS": "Gasly",
    "OCO": "Ocon",
    "ALB": "Albon",
    "SAR": "Sargeant",
    "COL": "Colapinto",
    "TSU": "Tsunoda",
    "RIC": "Ricciardo",
    "LAW": "Lawson",
    "HAD": "Hadjar",
    "BOT": "Bottas",
    "ZHO": "Zhou",
    "BOR": "Bortoleto",
    "HUL": "Hülkenberg",
    "MAG": "Magnussen",
    "BEA": "Bearman",
    "ANT": "Antonelli",
    "DOO": "Doohan",
    "VET": "Vettel",
    "RAI": "Räikkönen",
}


def ghost_short_name(meta: dict | None) -> str | None:
    """``"VER ’25"`` from a ghost sidecar's ``driver`` and ``year``; ``None`` without a
    driver."""
    if not meta:
        return None
    driver = str(meta.get("driver") or "").strip().upper()
    if not driver:
        return None
    year = meta.get("year")
    try:
        return f"{driver} ’{int(year) % 100:02d}"
    except (TypeError, ValueError):
        return driver


def ghost_long_name(meta: dict | None) -> str | None:
    """``"Verstappen 2025"`` (surname if known, else the code); ``None`` without a driver."""
    if not meta:
        return None
    driver = str(meta.get("driver") or "").strip().upper()
    if not driver:
        return None
    name = DRIVER_SURNAMES.get(driver, driver)
    year = meta.get("year")
    try:
        return f"{name} {int(year)}"
    except (TypeError, ValueError):
        return name


# ------------------------------------------------------------------------------ fonts

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = (
    "Archivo/ArchivoExpanded-Bold.ttf",
    "Archivo/ArchivoSemiExpanded-Bold.ttf",
    "Archivo/Archivo-Medium.ttf",
    "JetBrainsMono/JetBrainsMono-Regular.ttf",
)
_SANS = ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")


@dataclass(frozen=True)
class Face:
    """A type style: font family names to try in order, weight and width ("stretch").

    The static Archivo files share the typographic family "Archivo": DirectWrite (Windows)
    picks the width from ``stretch``, while FreeType/fontconfig lists the wide cuts as
    their own families ("Archivo Expanded"), which are tried first.
    """

    names: tuple[str, ...]
    weight: str = "normal"
    stretch: str | None = None


FACES: dict[str, Face] = {
    "headline": Face(("Archivo Expanded", "Archivo", *_SANS), "bold", "expanded"),
    "micro": Face(("Archivo SemiExpanded", "Archivo", *_SANS), "bold", "semiexpanded"),
    "medium": Face(("Archivo", "Archivo Medium", *_SANS), "medium"),
    "mono": Face(("JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")),
}

_fonts_loaded: bool | None = None


def load_fonts() -> bool:
    """Register the bundled TTFs with pyglet (once). Returns ``True`` if all loaded; on any
    failure logs one warning and the text falls back to system fonts."""
    global _fonts_loaded
    if _fonts_loaded is not None:
        return _fonts_loaded
    import arcade

    ok = True
    for rel in FONT_FILES:
        try:
            arcade.load_font(FONT_DIR / rel)
        except Exception as exc:  # noqa: BLE001 - any failure means "use system fonts"
            ok = False
            log.warning("Can't load font %s (%s); using system fonts", rel, exc)
            break
    _fonts_loaded = ok
    return ok


def px_to_pt(px: float) -> float:
    """Pixel size -> the point size arcade/pyglet expect (96 dpi)."""
    return px * 0.75


def pixel_ratio() -> float:
    """Framebuffer pixels per logical pixel of the current window (1 without one)."""
    try:
        import arcade

        return float(arcade.get_window().get_pixel_ratio()) or 1.0
    except Exception:  # noqa: BLE001 - no window (tests, before it opens)
        return 1.0


class SharpText:
    """An ``arcade.Text`` that stays sharp at any display scale.

    arcade's HiDPI mode ("stretch") lays everything out in logical pixels and stretches them
    onto the framebuffer, so pyglet's 96-dpi glyphs come out magnified, blurred on a 2x
    display. This keeps the logical API the layouts use (``position``, ``x``, ``y``,
    ``content_width``, ``width`` and so on, all in logical pixels) but rasters the glyphs at
    the framebuffer's DPI and draws them at whole framebuffer pixels under a framebuffer-pixel
    projection. It follows the window's pixel ratio when that changes (another monitor)."""

    def __init__(self, text, tracking_px: float = 0.0, ratio: float | None = None):
        self._t = text  # the arcade.Text, laid out in framebuffer pixels
        self._x, self._y = float(text.x), float(text.y)
        self._tracking = tracking_px
        self._wrap: float | None = None
        self._ratio = 0.0
        self._sync(ratio)

    # The pixel ratio: everything physical is logical times this

    def _sync(self, ratio: float | None = None) -> float:
        r = pixel_ratio() if ratio is None else ratio
        if r != self._ratio:
            self._ratio = r
            t = self._t
            t.label.dpi = 96 * r
            if self._tracking:
                t.tracking = self._tracking * r
            if self._wrap is not None:
                t.width = int(self._wrap * r)
            self._move()
        return r

    def _move(self) -> None:
        r = self._ratio
        self._t.position = (round(self._x * r), round(self._y * r))

    # Logical-pixel API (the subset of arcade.Text the app uses)

    @property
    def label(self):
        return self._t.label

    @property
    def text(self) -> str:
        return self._t.text

    @text.setter
    def text(self, value) -> None:
        self._t.text = value

    @property
    def color(self):
        return self._t.color

    @color.setter
    def color(self, value) -> None:
        self._t.color = value

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self.position = (value, self._y)

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        self.position = (self._x, value)

    @property
    def position(self) -> tuple[float, float]:
        return self._x, self._y

    @position.setter
    def position(self, value) -> None:
        x, y = float(value[0]), float(value[1])
        if (x, y) != (self._x, self._y):
            self._x, self._y = x, y
            self._move()

    @property
    def anchor_x(self) -> str:
        return self._t.anchor_x

    @anchor_x.setter
    def anchor_x(self, value: str) -> None:
        self._t.anchor_x = value

    @property
    def anchor_y(self) -> str:
        return self._t.anchor_y

    @anchor_y.setter
    def anchor_y(self, value: str) -> None:
        self._t.anchor_y = value

    @property
    def font_size(self) -> float:
        return self._t.font_size  # points: the DPI scales them

    @font_size.setter
    def font_size(self, value: float) -> None:
        self._t.font_size = value

    @property
    def tracking(self) -> float:
        return self._tracking

    @tracking.setter
    def tracking(self, px: float) -> None:
        self._tracking = px
        self._t.tracking = px * self._ratio

    @property
    def width(self) -> float | None:
        """The wrap width of multiline text, logical pixels."""
        return self._wrap

    @width.setter
    def width(self, px: float) -> None:
        self._wrap = float(px)
        self._t.width = int(px * self._ratio)

    @property
    def content_width(self) -> float:
        r = self._sync()
        return self._t.content_width / r

    @property
    def content_height(self) -> float:
        r = self._sync()
        return self._t.content_height / r

    def draw(self) -> None:
        self._sync()
        if _batch is not None:
            _batch.append(self._t)  # drawn when the batch ends
        else:
            _draw_in_pixels([self._t])


_batch: list | None = None
_pixel_projection: dict[tuple[int, int], object] = {}


def _draw_in_pixels(texts) -> None:
    """Draw ``arcade.Text`` objects laid out in framebuffer pixels, under one projection."""
    import arcade
    from pyglet.math import Mat4

    window = arcade.get_window()
    size = window.get_framebuffer_size()
    pixels = _pixel_projection.get(size)
    if pixels is None:
        pixels = _pixel_projection[size] = Mat4.orthogonal_projection(
            0, size[0], 0, size[1], -100, 100
        )
    projection, view = window.projection, window.view
    window.projection, window.view = pixels, Mat4()
    for t in texts:
        t.draw()
    window.projection, window.view = projection, view


@contextmanager
def text_batch():
    """Inside this block :meth:`SharpText.draw` only queues; the queue is drawn at the end
    under one projection change (one per text costs as much as drawing the text). Text then
    lands above everything else drawn in the block: batch per widget, not per screen."""
    global _batch
    if _batch is not None:  # already batching
        yield
        return
    _batch = []
    try:
        yield
    finally:
        queued, _batch = _batch, None
        if queued:
            _draw_in_pixels(queued)


def make_text(
    text: str,
    px: float,
    colour: Colour,
    *,
    face: str = "medium",
    anchor_x: str = "left",
    anchor_y: str = "baseline",
    tracking_px: float = 0.0,
    multiline: bool = False,
    width: float | None = None,
) -> SharpText:
    """A :class:`SharpText` in one of the :data:`FACES`. Create once, then only change
    ``.text`` / ``.color`` / ``.position``. ``width`` wraps ``multiline`` text."""
    import arcade

    style = FACES[face]
    t = arcade.Text(
        text,
        0,
        0,
        colour,
        px_to_pt(px),
        font_name=style.names,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        multiline=multiline,
        width=int(width) if width is not None else None,
    )
    if style.weight != "normal":
        t.label.weight = style.weight
    if style.stretch:
        t.label.set_style("stretch", style.stretch)
    sharp = SharpText(t, tracking_px)
    if width is not None:
        sharp.width = width
    return sharp


def micro(text: str, colour: Colour = TEXT_LABEL, anchor_x: str = "left"):
    """A 10 px bold UPPERCASE label with 1 px tracking (red for categories, white for
    metadata)."""
    return make_text(
        text.upper(),
        MICRO_PX,
        colour,
        face="micro",
        anchor_x=anchor_x,
        tracking_px=MICRO_TRACKING_PX,
    )
