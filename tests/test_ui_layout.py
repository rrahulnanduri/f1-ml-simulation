"""HUD layout and theme helpers: pure functions, no window."""

import importlib
import math
import sys

import numpy as np
import pytest

from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme

SIZES = [(1280, 800), (800, 600), (2560, 1440), (1024, 768)]


@pytest.mark.parametrize("size", SIZES)
def test_cards_are_anchored_to_the_window_corners(size):
    w, h = size
    m = lay.MARGIN
    tower = lay.tower_rect(w, h, 4)
    gap = lay.gap_rect(w, h)
    card = lay.card_rect(w, h)
    speed = lay.speed_rect(w, h)
    footer = lay.footer_rect(w, h)
    assert (footer.left, footer.bottom, footer.right) == (m, m, w - m)
    assert footer.h == lay.FOOTER_H == 40
    assert (gap.right, gap.top) == (w - m, h - m)
    assert (card.right, speed.left) == (w - m, m)
    # The bottom row sits one gutter above the footer; the tower under the ticker/headline.
    assert card.bottom == speed.bottom == footer.top + lay.GUTTER
    assert tower.left == m and tower.top == h - m - lay.HEAD_H
    for r in (tower, gap, card, speed):
        assert r.w == lay.COL_W == 256


@pytest.mark.parametrize("size", SIZES)
def test_cards_never_overlap_and_live_on_the_8px_grid(size):
    w, h = size
    rects = lay.card_rects(w, h, 4)
    names = list(rects)
    for i, a in enumerate(names):
        r = rects[a]
        assert 0 <= r.left and r.right <= w and 0 <= r.bottom and r.top <= h
        for b in names[i + 1 :]:
            assert not r.overlaps(rects[b]), (a, b)
    # Offsets from the window edge they're anchored to, and sizes, are multiples of 8.
    for r in rects.values():
        assert r.w % 8 == 0 or r.w == w - 2 * lay.MARGIN
        assert r.h % 8 == 0
        assert r.left % 8 == 0 or (w - r.right) % 8 == 0
        assert r.bottom % 8 == 0 or (h - r.top) % 8 == 0


@pytest.mark.parametrize("size", SIZES)
def test_the_followed_car_is_never_under_a_card(size):
    w, h = size
    cx, cy = lay.stage_centre(w, h)
    assert cx == 0.5 * w and cy > 0.5 * h  # nudged up: the footer takes the bottom
    car = lay.Rect(cx - 60, cy - 60, 120, 120)  # a large car sprite and its label
    for name, r in lay.card_rects(w, h, 5).items():
        assert not r.overlaps(car), name
    assert not lay.banner_rect(w, h).overlaps(car)


def test_ticker_and_headline_sit_above_the_tower():
    w, h = 1280, 800
    tx, ty = lay.ticker_origin(w, h)
    assert tx == lay.MARGIN and ty + lay.MICRO_CAP == h - lay.MARGIN  # caps flush with the top
    lines = lay.headline_baselines(w, h, 2)
    assert lines[0] - lines[1] == lay.HEADLINE_LEAD
    tower = lay.tower_rect(w, h, 3)
    assert ty > lines[0] > lines[1] > tower.top
    assert lines[1] - tower.top == 16.0


def test_tower_grows_with_its_rows():
    assert lay.tower_height(4) == 32 + 4 * 24 + 8
    assert lay.tower_height(5) - lay.tower_height(4) == pytest.approx(lay.TOWER_ROW_H)
    t = lay.tower_rect(1280, 800, 4)
    ys = [lay.tower_row_baseline(t, i) for i in range(4)]
    assert ys == sorted(ys, reverse=True)
    assert t.bottom < ys[-1] < ys[0] < t.top - lay.TOWER_HEADER_H
    assert lay.tower_row_centre(t, 3) - lay.TOWER_ROW_H / 2 == t.bottom + lay.TOWER_PAD_BOTTOM
    assert lay.tower_separator_y(t, 0) == pytest.approx(lay.tower_row_centre(t, 0) - 12.0)


def test_rect_helpers():
    r = lay.Rect(10, 20, 100, 50)
    assert r.contains(10, 20) and r.contains(110, 70) and not r.contains(111, 30)
    assert r.centre == (60, 45)
    assert r.inset(5) == lay.Rect(15, 25, 90, 40)
    assert r.overlaps(lay.Rect(100, 60, 50, 50))
    assert not r.overlaps(lay.Rect(110, 20, 5, 5))  # touching edges don't overlap


def test_button_and_page_dots_live_in_the_card_corner():
    card = lay.card_rect(1280, 800)
    bx, by = lay.button_centre(card)
    assert (card.right - bx, card.top - by) == (lay.PAD + lay.BUTTON_R,) * 2
    dots = lay.page_dots(card, 3)
    assert len(dots) == 3 and all(card.contains(x, y) for x, y in dots)
    assert all(x == bx for x, _ in dots)
    assert by - lay.BUTTON_R > dots[0][1] > dots[1][1] > dots[2][1] > card.bottom
    strokes = lay.arrow_strokes(bx, by)
    tip = strokes[0][2:]
    assert all(s[2:] == tip for s in strokes)  # all three meet at the tip...
    assert tip[0] > bx and tip[1] > by  # ...top-right: ↗
    assert all(math.hypot(x - bx, y - by) < lay.BUTTON_R for s in strokes for x, y in (s[:2],))


def test_page_offsets():
    # The shown page sits at 0; the next one is a card height below, the previous above.
    assert lay.page_offset_y(1, 1.0, 96) == 0.0
    assert lay.page_offset_y(2, 1.0, 96) == -96.0
    assert lay.page_offset_y(0, 1.0, 96) == 96.0
    assert lay.page_offset_y(1, 0.5, 96) == -48.0


def test_footer_drops_links_that_dont_fit():
    footer = lay.footer_rect(1280, 800)
    x0, xs, sx = lay.footer_positions(footer, 60, [80, 60, 70, 60], 140)
    assert x0 == footer.x + 16
    assert xs[0] == x0 + 60 + 32
    assert np.diff(xs).tolist() == [80 + 24, 60 + 24, 70 + 24]
    assert sx == footer.right - 16 - 140
    # Narrower: the controls come first, so the status goes before any link does.
    narrow = lay.footer_rect(500, 600)
    _, xs, sx = lay.footer_positions(narrow, 60, [80, 60, 70, 60], 140)
    assert sx is None and len(xs) == 4
    assert xs[-1] + 60 <= narrow.right - 16
    # Narrower still: links that don't fit are dropped from the end.
    _, xs, sx = lay.footer_positions(lay.footer_rect(400, 600), 60, [80, 60, 70, 60], 140)
    assert sx is None and len(xs) == 2
    _, xs, sx = lay.footer_positions(narrow, 60, [80, 60], 0)
    assert sx is None and len(xs) == 2


def test_stacked_labels_dont_overlap():
    boxes = [lay.Rect(100, 200, 60, 12), lay.Rect(110, 205, 60, 12), lay.Rect(400, 200, 60, 12)]
    ys = lay.stack_labels(boxes, [1, 0, 2], 12.0)
    assert ys[1] == 205 and ys[2] == 200  # the first keeps its place, a lone one too
    assert ys[0] == 200 - 12  # moved down a step: clear of the first
    moved = [lay.Rect(b.x, y, b.w, b.h) for b, y in zip(boxes, ys, strict=True)]
    assert not any(moved[i].overlaps(moved[j]) for i in range(3) for j in range(i + 1, 3))


def test_rounded_rect_points_stay_inside_and_round_the_corners():
    r = lay.Rect(0, 0, 100, 40)
    pts = lay.rounded_rect_points(r, 18, arc_segments=8)
    assert pts.shape == (36, 2)
    assert pts[:, 0].min() == pytest.approx(0) and pts[:, 0].max() == pytest.approx(100)
    assert pts[:, 1].min() == pytest.approx(0) and pts[:, 1].max() == pytest.approx(40)
    assert np.min(np.hypot(pts[:, 0], pts[:, 1])) > 18 * (math.sqrt(2) - 1) * 0.99
    pts = lay.rounded_rect_points(lay.Rect(0, 0, 100, 10), 50)
    assert pts[:, 1].max() == pytest.approx(10)


def test_capsule_points():
    pts = lay.capsule_points(10.0, 20.0, math.pi / 2, 5.6, 2.0)
    # Along +y: 5.6 long, 2.0 wide, centred on (10, 20).
    assert np.ptp(pts[:, 1]) == pytest.approx(5.6)
    assert np.ptp(pts[:, 0]) == pytest.approx(2.0)
    assert pts.mean(axis=0) == pytest.approx([10.0, 20.0])


# ------------------------------------------------------------------------------ theme


def test_design_tokens():
    assert theme.hex_colour("#EB0D3F") == theme.ACCENT == (235, 13, 63, 255)
    assert theme.hex_colour("3D6BD8", 0.16) == (61, 107, 216, 41)
    assert theme.white(0.20) == theme.CARD_BORDER
    assert theme.BACKGROUND == theme.hex_colour("#131313")
    assert theme.ROAD == theme.hex_colour("#1E1F22")
    assert theme.GREY == theme.hex_colour("#3E3E3E")
    assert theme.TEXT_SECONDARY == theme.white(0.6)
    assert theme.CARD_RADIUS == 8 and theme.CARD_FILL == 0.88
    assert theme.CARD_FALLBACK == (0, 0, 0, 224)
    # The accent replaced orange everywhere.
    for c in (theme.SELF, theme.BEHIND, theme.DOT_ACTIVE, theme.CHART_SELF, theme.KERB_RED):
        assert c == theme.ACCENT
    assert theme.with_alpha(theme.ACCENT, 0.5) == (235, 13, 63, 128)
    with pytest.raises(ValueError):
        theme.hex_colour("#FFF")


@pytest.mark.parametrize(
    "team, colour",
    [
        ("Red Bull Racing", "#3671C6"),
        ("Mercedes", "#27F4D2"),
        ("Ferrari", "#E8002D"),
        ("McLaren", "#FF8000"),
        ("Aston Martin", "#229971"),
        ("Alpine", "#0093CC"),
        ("Williams", "#64C4FF"),
        ("RB", "#6692FF"),
        ("Racing Bulls", "#6692FF"),
        ("Visa Cash App RB", "#6692FF"),
        ("Kick Sauber", "#52E252"),
        ("Haas F1 Team", "#B6BABD"),
    ],
)
def test_team_colours(team, colour):
    assert theme.team_colour(team) == theme.hex_colour(colour)


def test_liveries():
    assert theme.livery("Red Bull Racing") == theme.LIVERIES["red bull"]
    assert theme.livery("Racing Bulls") == theme.LIVERIES["rb"]
    assert theme.livery("Brabham") == theme.NEUTRAL_LIVERY
    assert theme.livery(None) == theme.NEUTRAL_LIVERY
    me = theme.livery("Ferrari", is_self=True)
    assert me == theme.SELF_LIVERY
    assert me.primary == theme.hex_colour("#A9ADB2") and me.accent == theme.ACCENT
    assert set(theme.LIVERIES) == set(theme.TEAM_COLOURS)


def test_unknown_teams_and_trace_colours():
    assert theme.team_colour("") is None
    assert theme.team_colour(None) is None
    assert theme.team_colour("Brabham") is None
    colours = theme.trace_colours(
        ["Red Bull Racing", None, "", None, None], [True, True, False, False, True]
    )
    assert colours[0] == theme.TEAM_COLOURS["red bull"]
    assert colours[1] == theme.MUTED_PALETTE[0]  # a ghost without a team: grey
    assert colours[2] == theme.SELF  # the first non-ghost: the accent
    assert colours[3] == theme.MUTED_PALETTE[1]  # only one accent
    assert colours[4] == theme.MUTED_PALETTE[2]


def test_ghost_names():
    meta = {"driver": "VER", "year": 2025, "team": "Red Bull Racing"}
    assert theme.ghost_short_name(meta) == "VER ’25"
    assert theme.ghost_long_name(meta) == "Verstappen 2025"
    assert theme.ghost_long_name({"driver": "XYZ", "year": 2023}) == "XYZ 2023"
    assert theme.ghost_short_name({"driver": "ham"}) == "HAM"
    assert theme.ghost_short_name(None) is None
    assert theme.ghost_short_name({"team": "Ferrari"}) is None


def test_no_glass_env_switch(monkeypatch):
    monkeypatch.delenv(theme.NO_GLASS_ENV, raising=False)
    assert not theme.glass_disabled_by_env()
    monkeypatch.setenv(theme.NO_GLASS_ENV, "0")
    assert not theme.glass_disabled_by_env()
    monkeypatch.setenv(theme.NO_GLASS_ENV, "1")
    assert theme.glass_disabled_by_env()


def test_fonts_are_bundled_with_licences():
    for rel in theme.FONT_FILES:
        path = theme.FONT_DIR / rel
        assert path.is_file() and path.read_bytes()[:4] == b"\x00\x01\x00\x00"  # TrueType
    families = {rel.split("/")[0] for rel in theme.FONT_FILES}
    assert families == {"Archivo", "JetBrainsMono"}
    for family in families:
        text = (theme.FONT_DIR / family / "OFL.txt").read_text(encoding="utf-8")
        assert "SIL OPEN FONT LICENSE Version 1.1" in text
    # Nothing bundled that isn't used.
    bundled = {p.relative_to(theme.FONT_DIR).as_posix() for p in theme.FONT_DIR.rglob("*.ttf")}
    assert bundled == set(theme.FONT_FILES)
    assert theme.FACES["headline"].stretch == "expanded"
    assert theme.FACES["headline"].weight == "bold"


def test_importing_the_ui_opens_no_window():
    for name in [m for m in sys.modules if m.startswith("f1sim.app.ui")]:
        sys.modules.pop(name)
    for mod in ("theme", "layout", "timing", "paging", "scramble", "pixelcar", "glass", "hud"):
        importlib.import_module(f"f1sim.app.ui.{mod}")
    from arcade import window_commands

    assert window_commands._window is None
