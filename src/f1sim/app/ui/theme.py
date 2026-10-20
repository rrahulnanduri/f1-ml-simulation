"""The "pit wall noir" theme: colours, type, sizes, team colours and font loading.

Everything here is constants and pure functions except :func:`load_fonts` and
:func:`make_text`, which touch arcade/pyglet only when called (never on import).

Colours are ``(r, g, b, a)`` tuples of ints 0-255. Sizes are screen pixels at 1x scale; arcade
text sizes are points, so :func:`px_to_pt` converts (pyglet renders at 96 dpi: 1 pt = 4/3 px).
"""

from __future__ import annotations

import logging
import os
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


# ------------------------------------------------------------------------------ scene

BACKGROUND = hex_colour("#0B0C0F")
ROAD = hex_colour("#1A1C21")
WALL = white(0.30)
"""Hairline walls: 1 GL pixel at any zoom."""
START_LINE = white(1.0)
START_LINE_PX = 2.5

AMBIENT_COOL = hex_colour("#3D6BD8", 0.16)
AMBIENT_WARM = hex_colour("#FF7A2F", 0.16)
AMBIENT_COOL_CENTRE = (0.10, 0.90)
"""Centre of the cool wash as a fraction of the window (x from the left, y from the bottom)."""
AMBIENT_WARM_CENTRE = (0.90, 0.10)
AMBIENT_RADIUS = 0.45
"""Radius of each wash as a fraction of the window diagonal."""

# ------------------------------------------------------------------------------- cars

CAR_LENGTH_M = 5.6
CAR_WIDTH_M = 2.0
CAR_MIN_LENGTH_PX = 9.0
CAR_MIN_WIDTH_PX = 4.0
CAR_STROKE_PX = 1.1
SELF = hex_colour("#FF7A2F")

# ------------------------------------------------------------------------------- text

TEXT_PRIMARY = hex_colour("#EEEFF1")
TEXT_SECONDARY = hex_colour("#A7AAB1")
TEXT_MUTED = hex_colour("#8C8F96")
TEXT_LABEL = hex_colour("#9A9DA4")
TEXT_SPEED = hex_colour("#F4F4F5")
SELF_GAP = hex_colour("#FF9A5C")
BEHIND = SELF
AHEAD = hex_colour("#3DD68C")
OFF_TRACK = hex_colour("#FF5A5A")

SECTOR_PURPLE = hex_colour("#A97CFF")
SECTOR_GREEN = AHEAD
SECTOR_YELLOW = hex_colour("#E8C547")
SECTOR_UNSET = white(0.12)
SECTOR_COLOURS = {
    "purple": SECTOR_PURPLE,
    "green": SECTOR_GREEN,
    "yellow": SECTOR_YELLOW,
    "unset": SECTOR_UNSET,
}

SEPARATOR = white(0.08)
CHART_REF = white(0.32)
CHART_SELF = hex_colour("#FF7A2F", 0.80)
CHART_MARKER = white(0.40)
GAP_TRACK = white(0.10)
DOT_ACTIVE = white(0.90)
DOT_INACTIVE = white(0.28)

LABEL_PX = 10
LABEL_TRACKING_EM = 0.14
NAME_PX = 12
TIME_PX = 11
GAP_PX = 26
SPEED_PX = 44
UNIT_PX = 12
SECTOR_TIME_PX = 10
GAP_PAGE_PX = 15

# ------------------------------------------------------------------------------ glass

PANEL_RADIUS = 18.0
CARD_RADIUS = 20.0
BLUR_SIGMA_PX = 9.0
BLUR_BRIGHTEN = 1.15
TINT_TOP = 0.10
TINT_BOTTOM = 0.035
EDGE_TOP_LEFT = 0.38
EDGE_MIDDLE = 0.08
EDGE_BOTTOM_RIGHT = 0.16
SHADOW_DY_PX = 10.0
SHADOW_BLUR_PX = 12.0
SHADOW_ALPHA = 0.45
PANEL_FALLBACK = hex_colour("#16181C", 0.85)

NO_GLASS_ENV = "F1SIM_NO_GLASS"


def glass_disabled_by_env() -> bool:
    """``F1SIM_NO_GLASS`` set to anything but ``""``/``0`` forces the flat fallback panels."""
    return os.environ.get(NO_GLASS_ENV, "").strip() not in ("", "0")


def edge_alpha(g: float) -> float:
    """Opacity of the panel edge at diagonal position ``g`` (0 = top-left, 1 = bottom-right):
    ``EDGE_TOP_LEFT`` -> ``EDGE_MIDDLE`` at 0.5 -> ``EDGE_BOTTOM_RIGHT``, linear."""
    g = min(max(float(g), 0.0), 1.0)
    if g < 0.5:
        return EDGE_TOP_LEFT + (EDGE_MIDDLE - EDGE_TOP_LEFT) * (g / 0.5)
    return EDGE_MIDDLE + (EDGE_BOTTOM_RIGHT - EDGE_MIDDLE) * ((g - 0.5) / 0.5)


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
"""Greys for traces without a team, other than the orange "self"."""


def team_colour(team: str | None) -> Colour | None:
    """The team's colour, or ``None`` for an empty or unknown team name."""
    name = (team or "").strip().lower()
    if not name:
        return None
    if name == "rb":
        return TEAM_COLOURS["rb"]
    for alias, key in _TEAM_ALIASES:
        if alias in name:
            return TEAM_COLOURS[key]
    return None


def trace_colours(teams: list[str | None], is_ghost: list[bool]) -> list[Colour]:
    """One colour per trace: its team colour; else orange for the first trace that isn't a
    ghost (the "self" car) and greys from :data:`MUTED_PALETTE` for the rest."""
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
    "InterTight/InterTight-Light.ttf",
    "InterTight/InterTight-Regular.ttf",
    "JetBrainsMono/JetBrainsMono-Light.ttf",
    "JetBrainsMono/JetBrainsMono-Regular.ttf",
)
# Family names as pyglet resolves them: DirectWrite groups the static files by their
# typographic family ("Inter Tight" + weight), FreeType/GDI may list "Inter Tight Light".
# The system fonts after them are the fallback when the bundled files can't be loaded.
TEXT_FONTS = ("Inter Tight", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")
TEXT_FONTS_LIGHT = ("Inter Tight", "Inter Tight Light", *TEXT_FONTS[1:])
MONO_FONTS = ("JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")
MONO_FONTS_LIGHT = ("JetBrains Mono", "JetBrains Mono Light", *MONO_FONTS[1:])

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


def make_text(
    text: str,
    px: float,
    colour: Colour,
    *,
    mono: bool = False,
    light: bool = False,
    anchor_x: str = "left",
    anchor_y: str = "baseline",
    tracking_em: float = 0.0,
):
    """An ``arcade.Text`` in the theme's fonts. Create once, then only change ``.text`` /
    ``.color`` / ``.position``."""
    import arcade

    if mono:
        names = MONO_FONTS_LIGHT if light else MONO_FONTS
    else:
        names = TEXT_FONTS_LIGHT if light else TEXT_FONTS
    t = arcade.Text(
        text,
        0,
        0,
        colour,
        px_to_pt(px),
        font_name=names,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )
    if light:
        t.label.weight = "light"
    if tracking_em:
        t.tracking = tracking_em * px
    return t
