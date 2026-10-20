"""Glass HUD layout and theme helpers: pure functions, no window."""

import importlib
import math
import sys

import numpy as np
import pytest

from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme


@pytest.mark.parametrize("size", [(1280, 800), (800, 600), (2560, 1440)])
def test_panels_are_anchored_to_the_window_corners(size):
    w, h = size
    tower = lay.tower_rect(w, h, 4)
    gap = lay.gap_rect(w, h)
    card = lay.card_rect(w, h)
    m = lay.MARGIN
    assert (tower.left, tower.top) == (m, h - m)
    assert (gap.right, gap.top) == (w - m, h - m)
    assert (card.right, card.bottom) == (w - m, m)
    assert (card.w, card.h) == (258, 82)
    assert (gap.w, gap.h) == (150, 58)
    assert tower.w == 176
    sx, sy = lay.speed_origin(w, h)
    assert sx < 30 and sy < 60


def test_tower_grows_with_its_rows():
    assert lay.tower_height(3) == pytest.approx(108.0)  # ~176 x 114 for three rows
    assert lay.tower_height(5) - lay.tower_height(4) == pytest.approx(lay.TOWER_ROW_H)
    t = lay.tower_rect(1280, 800, 4)
    # Rows sit inside the panel, top to bottom, separators between them.
    ys = [lay.tower_row_baseline(t, i) for i in range(4)]
    assert ys == sorted(ys, reverse=True)
    assert t.bottom < ys[-1] < ys[0] < t.top
    assert lay.tower_row_centre(t, 3) - lay.TOWER_ROW_H / 2 >= t.bottom
    assert lay.tower_separator_y(t, 0) == pytest.approx(lay.tower_row_centre(t, 0) - 11.0)


def test_rect_helpers():
    r = lay.Rect(10, 20, 100, 50)
    assert r.contains(10, 20) and r.contains(110, 70) and not r.contains(111, 30)
    assert r.centre == (60, 45)
    assert r.inset(5) == lay.Rect(15, 25, 90, 40)


def test_help_line_only_when_it_fits():
    assert lay.help_anchor(1280, 800, 450) is not None
    assert lay.help_anchor(700, 500, 450) is None
    x, _ = lay.help_anchor(1280, 800, 450)
    assert lay.MARGIN + lay.SPEED_BLOCK_W < x < 1280 - lay.MARGIN - lay.CARD_W


def test_page_offsets_and_dots():
    # The shown page sits at 0; the next one is a card height below, the previous above.
    assert lay.page_offset_y(1, 1.0, 82) == 0.0
    assert lay.page_offset_y(2, 1.0, 82) == -82.0
    assert lay.page_offset_y(0, 1.0, 82) == 82.0
    assert lay.page_offset_y(1, 0.5, 82) == -41.0
    card = lay.card_rect(1280, 800)
    dots = lay.page_dots(card, 3)
    assert len(dots) == 3 and all(card.contains(x, y) for x, y in dots)
    assert dots[0][1] > dots[1][1] > dots[2][1]
    assert dots[0][0] > card.right - 20


def test_rounded_rect_points_stay_inside_and_round_the_corners():
    r = lay.Rect(0, 0, 100, 40)
    pts = lay.rounded_rect_points(r, 18, arc_segments=8)
    assert pts.shape == (36, 2)
    assert pts[:, 0].min() == pytest.approx(0) and pts[:, 0].max() == pytest.approx(100)
    assert pts[:, 1].min() == pytest.approx(0) and pts[:, 1].max() == pytest.approx(40)
    # No point near a sharp corner.
    assert np.min(np.hypot(pts[:, 0], pts[:, 1])) > 18 * (math.sqrt(2) - 1) * 0.99
    # Radius clamps to half the shorter side.
    pts = lay.rounded_rect_points(lay.Rect(0, 0, 100, 10), 50)
    assert pts[:, 1].max() == pytest.approx(10)


def test_capsule_has_world_size_with_a_minimum_on_screen():
    assert lay.capsule_size_px(4.0, 5.6, 2.0, 9, 4) == pytest.approx((22.4, 8.0))
    assert lay.capsule_size_px(0.1, 5.6, 2.0, 9, 4) == (9, 4)
    pts = lay.capsule_points(10.0, 20.0, math.pi / 2, 5.6, 2.0)
    # Along +y: 5.6 long, 2.0 wide, centred on (10, 20).
    assert np.ptp(pts[:, 1]) == pytest.approx(5.6)
    assert np.ptp(pts[:, 0]) == pytest.approx(2.0)
    assert pts.mean(axis=0) == pytest.approx([10.0, 20.0])


# ------------------------------------------------------------------------------ theme


def test_hex_colours_and_edge_gradient():
    assert theme.hex_colour("#FF7A2F") == (255, 122, 47, 255)
    assert theme.hex_colour("3D6BD8", 0.16) == (61, 107, 216, 41)
    assert theme.white(0.30)[3] == 77
    with pytest.raises(ValueError):
        theme.hex_colour("#FFF")
    assert theme.edge_alpha(0.0) == pytest.approx(0.38)
    assert theme.edge_alpha(0.5) == pytest.approx(0.08)
    assert theme.edge_alpha(1.0) == pytest.approx(0.16)
    assert theme.edge_alpha(0.25) == pytest.approx(0.23)
    assert theme.edge_alpha(-3) == pytest.approx(0.38)


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


def test_unknown_teams_and_trace_colours():
    assert theme.team_colour("") is None
    assert theme.team_colour(None) is None
    assert theme.team_colour("Brabham") is None
    colours = theme.trace_colours(
        ["Red Bull Racing", None, "", None, None], [True, True, False, False, True]
    )
    assert colours[0] == theme.TEAM_COLOURS["red bull"]
    assert colours[1] == theme.MUTED_PALETTE[0]  # a ghost without a team: grey
    assert colours[2] == theme.SELF  # the first non-ghost: orange
    assert colours[3] == theme.MUTED_PALETTE[1]  # only one orange
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
    for family in ("InterTight", "JetBrainsMono"):
        text = (theme.FONT_DIR / family / "OFL.txt").read_text(encoding="utf-8")
        assert "SIL OPEN FONT LICENSE Version 1.1" in text


def test_importing_the_ui_opens_no_window():
    for name in [m for m in sys.modules if m.startswith("f1sim.app.ui")]:
        sys.modules.pop(name)
    for mod in ("theme", "layout", "timing", "paging", "glass", "hud"):
        importlib.import_module(f"f1sim.app.ui.{mod}")
    from arcade import window_commands

    assert window_commands._window is None
