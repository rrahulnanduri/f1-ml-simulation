"""HUD widgets in screen space: timing tower, gap readout, speed readout and the paging card.

Each widget creates its ``arcade.Text`` objects once (in ``__init__``) and afterwards only
changes their text, colour and position; panels are drawn by
:class:`~f1sim.app.ui.glass.GlassRenderer`. Call ``layout(win_w, win_h)`` after a resize and
``draw()`` every frame under the window's default (pixel) camera.
"""

from __future__ import annotations

import math

import arcade
import numpy as np

from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme
from f1sim.app.ui.glass import GlassRenderer
from f1sim.app.ui.paging import Pager
from f1sim.app.ui.timing import TowerRow, format_gap, format_sector

NO_REFERENCE = "No reference — pass --ghost"
SPEED_MIN_KMH = 60.0
SPEED_MAX_KMH = 350.0
GAP_PX_PER_S = 48.0


def _set(text: arcade.Text, value: str) -> None:
    if text.text != value:  # only re-layout when the string changes
        text.text = value


def _colour(text: arcade.Text, colour) -> None:
    if tuple(text.color) != tuple(colour):
        text.color = colour


def _place(text: arcade.Text, x: float, y: float) -> None:
    x, y = round(x), round(y)  # whole pixels keep the glyphs crisp
    if (text.x, text.y) != (x, y):
        text.position = (x, y)


def _label(text: str = "") -> arcade.Text:
    return theme.make_text(
        text.upper(), theme.LABEL_PX, theme.TEXT_LABEL, tracking_em=theme.LABEL_TRACKING_EM
    )


def _hairline(x0: float, x1: float, y: float, colour) -> None:
    arcade.draw_lrbt_rectangle_filled(x0, x1, y - 0.5, y + 0.5, colour)


# ------------------------------------------------------------------------ the tower


class TimingTower:
    """Top-left glass panel: a header and ranked rows ``position, name, time-or-gap``."""

    def __init__(self, glass: GlassRenderer, capacity: int):
        self.glass = glass
        self.header = _label()
        self.capacity = int(capacity)
        self.pos = [
            theme.make_text("", theme.TIME_PX, theme.TEXT_MUTED, mono=True) for _ in range(capacity)
        ]
        self.names = [
            theme.make_text("", theme.NAME_PX, theme.TEXT_SECONDARY) for _ in range(capacity)
        ]
        self.values = [
            theme.make_text("", theme.TIME_PX, theme.TEXT_SECONDARY, mono=True, anchor_x="right")
            for _ in range(capacity)
        ]
        self.rows: list[TowerRow] = []
        self.rect = lay.Rect(0, 0, lay.TOWER_W, lay.tower_height(0))
        self._win = (1.0, 1.0)

    def set(self, header: str, rows: list[TowerRow]) -> None:
        _set(self.header, header.upper())
        rows = rows[: self.capacity]
        if len(rows) != len(self.rows):
            self.rows = rows
            self.layout(*self._win)
        self.rows = rows
        for i, row in enumerate(rows):
            _set(self.pos[i], row.position)
            _set(self.names[i], row.name[:14])
            _set(self.values[i], row.value)
            _colour(self.names[i], theme.TEXT_PRIMARY if row.is_self else theme.TEXT_SECONDARY)
            _colour(self.values[i], theme.SELF_GAP if row.is_self else theme.TEXT_SECONDARY)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        r = self.rect = lay.tower_rect(win_w, win_h, len(self.rows))
        _place(self.header, r.x + lay.TOWER_PAD_X, r.top - lay.TOWER_PAD_TOP - 9)
        for i in range(self.capacity):
            base = lay.tower_row_baseline(r, i)
            _place(self.pos[i], r.x + lay.TOWER_PAD_X + 7, base)
            _place(self.names[i], r.x + lay.TOWER_PAD_X + 22, base)
            _place(self.values[i], r.right - lay.TOWER_PAD_X, base)

    def draw(self) -> None:
        r = self.rect
        self.glass.panel(r, theme.PANEL_RADIUS)
        self.header.draw()
        x0 = r.x + lay.TOWER_PAD_X
        for i, row in enumerate(self.rows):
            cy = lay.tower_row_centre(r, i)
            arcade.draw_lrbt_rectangle_filled(x0, x0 + 2.5, cy - 9, cy + 9, row.colour)
            if i + 1 < len(self.rows):
                _hairline(
                    x0, r.right - lay.TOWER_PAD_X, lay.tower_separator_y(r, i), theme.SEPARATOR
                )
            self.pos[i].draw()
            self.names[i].draw()
            self.values[i].draw()


# --------------------------------------------------------------------- gap readout


class GapReadout:
    """Top-right glass panel: the big gap to the reference and ``vs <reference>``."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.value = theme.make_text(
            "--.---", theme.GAP_PX, theme.TEXT_MUTED, mono=True, light=True
        )
        self.sub = theme.make_text("", theme.LABEL_PX, theme.TEXT_LABEL)
        self.rect = lay.Rect(0, 0, lay.GAP_W, lay.GAP_H)

    def set(self, gap: float | None, ref_label: str | None) -> None:
        if ref_label is None:
            _set(self.value, "--.---")
            _colour(self.value, theme.TEXT_MUTED)
            _set(self.sub, NO_REFERENCE)
            return
        _set(self.value, format_gap(gap))
        if gap is None or not math.isfinite(gap):
            _colour(self.value, theme.TEXT_MUTED)
        else:
            _colour(self.value, theme.BEHIND if gap > 0.0 else theme.AHEAD)
        _set(self.sub, f"vs {ref_label}")

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.gap_rect(win_w, win_h)
        _place(self.value, r.x + 14, r.y + 25)
        _place(self.sub, r.x + 15, r.y + 11)

    def draw(self) -> None:
        self.glass.panel(self.rect, theme.PANEL_RADIUS)
        self.value.draw()
        self.sub.draw()


# ------------------------------------------------------------------- speed readout


class SpeedReadout:
    """Bottom-left, no panel: big speed and ``km/h · <gear or position>``."""

    def __init__(self):
        self.value = theme.make_text("0", theme.SPEED_PX, theme.TEXT_SPEED, mono=True, light=True)
        self.unit = theme.make_text("km/h", theme.UNIT_PX, theme.TEXT_MUTED)
        self._origin = (0.0, 0.0)

    def set(self, speed_kmh: float, extra: str) -> None:
        _set(self.value, f"{max(speed_kmh, 0.0):.0f}")
        _set(self.unit, f"km/h · {extra}" if extra else "km/h")

    def layout(self, win_w: float, win_h: float) -> None:
        self._origin = lay.speed_origin(win_w, win_h)
        _place(self.value, *self._origin)

    def draw(self) -> None:
        x, y = self._origin
        _place(self.unit, x + self.value.content_width + 8, y)
        self.value.draw()
        self.unit.draw()


# --------------------------------------------------------------------- paging card


class PagedWidget:
    """Bottom-right glass card with three vertically sliding pages: sectors, speed vs the
    reference over the lap, and the gap bar. See :class:`~f1sim.app.ui.paging.Pager`."""

    N_PAGES = 3

    def __init__(self, glass: GlassRenderer, lap_length: float, page: int = 0):
        self.glass = glass
        self.length = float(lap_length)
        self.pager = Pager(self.N_PAGES, page)
        self.rect = lay.Rect(0, 0, lay.CARD_W, lay.CARD_H)
        self.titles = [_label("Sectors"), _label("Speed"), _label("Gap")]
        # Page 1
        self.sector_times = [
            theme.make_text("--.---", theme.SECTOR_TIME_PX, theme.TEXT_MUTED, mono=True)
            for _ in range(3)
        ]
        self.sector_states = ["unset"] * 3
        # Page 2
        self._ref_speed: tuple[np.ndarray, np.ndarray] | None = None
        self._self_speed: tuple[np.ndarray, np.ndarray] | None = None
        self._ref_lines: list[list[tuple[float, float]]] = []
        self._self_lines: list[list[tuple[float, float]]] = []
        self.marker_s: float | None = None
        self.no_ref = [
            theme.make_text(NO_REFERENCE, theme.LABEL_PX, theme.TEXT_MUTED) for _ in range(2)
        ]
        # Page 3
        self.gap: float | None = None
        self.gap_value = theme.make_text(
            "--.---", theme.GAP_PAGE_PX, theme.TEXT_MUTED, mono=True, light=True, anchor_x="right"
        )
        self.gap_at = theme.make_text("", theme.LABEL_PX, theme.TEXT_LABEL, anchor_x="right")
        self.has_ref = False

    # -------------------------------------------------------------------- inputs

    def contains(self, x: float, y: float) -> bool:
        return self.rect.contains(x, y)

    def click(self, now: float) -> None:
        self.pager.next(now)

    def wheel(self, notches: float, now: float) -> None:
        self.pager.wheel(notches, now)

    @property
    def page(self) -> int:
        return self.pager.page

    # ---------------------------------------------------------------------- data

    def set_reference(self, ref_name: str | None) -> None:
        """Short reference name for the page titles, or ``None`` without a reference."""
        self.has_ref = ref_name is not None
        name = (ref_name or "").upper()
        _set(self.titles[1], f"SPEED VS {name}" if name else "SPEED")
        _set(self.titles[2], f"GAP TO {name}" if name else "GAP")

    def set_sectors(self, times: list[float], states: list[str]) -> None:
        n = len(self.sector_times)
        times = (list(times) + [math.nan] * n)[:n]
        states = (list(states) + ["unset"] * n)[:n]
        self.sector_states = states
        for text, t, st in zip(self.sector_times, times, states, strict=True):
            _set(text, format_sector(t))
            _colour(text, theme.TEXT_MUTED if st == "unset" else theme.SECTOR_COLOURS[st])

    def set_speed_traces(self, ref, self_trace) -> None:
        """``(s_grid, v_kmh)`` arrays for the reference (or ``None``) and the self car; the
        polylines are rebuilt here, not every frame."""
        self._ref_speed = ref
        self._self_speed = self_trace
        self._rebuild_chart()

    def set_marker(self, s: float | None) -> None:
        self.marker_s = s

    def set_gap(self, gap: float | None, at_label: str) -> None:
        self.gap = gap
        _set(self.gap_value, format_gap(gap))
        if gap is None or not math.isfinite(gap):
            _colour(self.gap_value, theme.TEXT_MUTED)
        else:
            _colour(self.gap_value, theme.BEHIND if gap > 0.0 else theme.AHEAD)
        _set(self.gap_at, at_label)

    # -------------------------------------------------------------------- layout

    def _chart_rect(self) -> lay.Rect:
        r = self.rect
        return lay.Rect(r.x + 16, r.y + 12, r.w - 16 - 30, r.h - 12 - 32)

    def _bar_span(self) -> tuple[float, float, float]:
        r = self.rect
        return r.x + 16, r.right - 30, r.y + 22

    def _polylines(self, trace) -> list[list[tuple[float, float]]]:
        if trace is None:
            return []
        s, v = trace
        c = self._chart_rect()
        x = c.x + np.asarray(s) / self.length * c.w
        vv = np.clip(np.asarray(v, dtype=np.float64), SPEED_MIN_KMH, SPEED_MAX_KMH)
        y = c.y + (vv - SPEED_MIN_KMH) / (SPEED_MAX_KMH - SPEED_MIN_KMH) * c.h
        ok = np.isfinite(np.asarray(v, dtype=np.float64))
        lines, cur = [], []
        for xi, yi, good in zip(x, y, ok, strict=True):
            if good:
                cur.append((float(xi), float(yi)))
            elif cur:
                lines.append(cur)
                cur = []
        if cur:
            lines.append(cur)
        return [ln for ln in lines if len(ln) >= 2]

    def _rebuild_chart(self) -> None:
        self._ref_lines = self._polylines(self._ref_speed)
        self._self_lines = self._polylines(self._self_speed)

    def layout(self, win_w: float, win_h: float) -> None:
        self.rect = lay.card_rect(win_w, win_h)
        self._rebuild_chart()

    # ---------------------------------------------------------------------- draw

    def draw(self, now: float) -> None:
        r = self.rect
        self.glass.panel(r, theme.CARD_RADIUS)
        ctx = self.glass.ctx
        pos = self.pager.position(now)
        ctx.scissor = (int(r.x), int(r.y + 1), int(r.w), int(r.h - 2))
        try:
            for i, draw_page in enumerate((self._draw_sectors, self._draw_speed, self._draw_gap)):
                dy = lay.page_offset_y(i, pos, r.h)
                if abs(dy) < r.h:
                    _place(self.titles[i], r.x + 16, r.top - 22 + dy)
                    self.titles[i].draw()
                    draw_page(dy)
        finally:
            ctx.scissor = None
        for k, (cx, cy) in enumerate(lay.page_dots(r, self.N_PAGES)):
            colour = theme.DOT_ACTIVE if k == self.pager.page else theme.DOT_INACTIVE
            arcade.draw_circle_filled(cx, cy, 2.0, colour, num_segments=12)

    def _draw_sectors(self, dy: float) -> None:
        x0, x1, _ = self._bar_span()
        gap = 8.0
        w = (x1 - x0 - 2 * gap) / 3.0
        for k, (text, state) in enumerate(zip(self.sector_times, self.sector_states, strict=True)):
            bx = x0 + k * (w + gap)
            by = self.rect.y + 20 + dy
            colour = theme.SECTOR_COLOURS[state]
            arcade.draw_lrbt_rectangle_filled(bx, bx + w, by, by + 1.5, colour)
            _place(text, bx, by + 9)
            text.draw()

    def _draw_no_ref(self, k: int, dy: float) -> None:
        r = self.rect
        _place(self.no_ref[k], r.x + 16, r.y + 30 + dy)
        self.no_ref[k].draw()

    def _draw_speed(self, dy: float) -> None:
        if not self.has_ref:
            self._draw_no_ref(0, dy)
            return
        for lines, colour, width in (
            (self._ref_lines, theme.CHART_REF, 1),
            (self._self_lines, theme.CHART_SELF, 1.2),
        ):
            for line in lines:
                # Shifted copies only while the page slides
                pts = line if dy == 0.0 else [(x, y + dy) for x, y in line]
                arcade.draw_line_strip(pts, colour, width)
        if self.marker_s is not None and math.isfinite(self.marker_s):
            c = self._chart_rect()
            mx = c.x + min(max(self.marker_s / self.length, 0.0), 1.0) * c.w
            arcade.draw_lrbt_rectangle_filled(
                mx - 0.5, mx + 0.5, c.y + dy, c.top + dy, theme.CHART_MARKER
            )

    def _draw_gap(self, dy: float) -> None:
        r = self.rect
        if not self.has_ref:
            self._draw_no_ref(1, dy)
            return
        _place(self.gap_value, r.right - 30, r.top - 24 + dy)
        _place(self.gap_at, r.right - 30, r.top - 38 + dy)
        self.gap_value.draw()
        self.gap_at.draw()
        x0, x1, y = self._bar_span()
        y += dy - 4
        arcade.draw_lrbt_rectangle_filled(x0 + 1.5, x1 - 1.5, y - 1.5, y + 1.5, theme.GAP_TRACK)
        arcade.draw_circle_filled(x0 + 1.5, y, 1.5, theme.GAP_TRACK, num_segments=8)
        arcade.draw_circle_filled(x1 - 1.5, y, 1.5, theme.GAP_TRACK, num_segments=8)
        cx = 0.5 * (x0 + x1)
        g = self.gap
        if g is not None and math.isfinite(g) and g != 0.0:
            length = min(abs(g) * GAP_PX_PER_S, 0.5 * (x1 - x0) - 1.5)
            colour = theme.BEHIND if g > 0.0 else theme.AHEAD
            # Behind grows to the right of the centre tick, ahead to the left.
            a, b = (cx, cx + length) if g > 0.0 else (cx - length, cx)
            arcade.draw_lrbt_rectangle_filled(a, b, y - 1.5, y + 1.5, colour)
        arcade.draw_lrbt_rectangle_filled(cx - 0.5, cx + 0.5, y - 5, y + 5, theme.CHART_MARKER)


# ----------------------------------------------------------------------- whole HUD


class Hud:
    """All HUD widgets of a viewer window, plus a top-centre status line, a centre banner
    (e.g. OFF TRACK) and a bottom help line. Owns the paging card's input handling."""

    def __init__(
        self, glass: GlassRenderer, tower_capacity: int, lap_length: float, help_text: str
    ):
        self.glass = glass
        self.tower = TimingTower(glass, tower_capacity)
        self.gap = GapReadout(glass)
        self.speed = SpeedReadout()
        self.card = PagedWidget(glass, lap_length)
        self.status = theme.make_text(
            "", theme.TIME_PX, theme.TEXT_SECONDARY, mono=True, anchor_x="center"
        )
        self.banner = theme.make_text(
            "",
            28,
            theme.OFF_TRACK,
            light=True,
            anchor_x="center",
            anchor_y="center",
            tracking_em=0.12,
        )
        self.help = theme.make_text(help_text, theme.LABEL_PX, theme.TEXT_MUTED, anchor_x="center")
        self.show_banner = False
        self._help_at: tuple[float, float] | None = None
        self._win = (1.0, 1.0)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        self.tower.layout(win_w, win_h)
        self.gap.layout(win_w, win_h)
        self.speed.layout(win_w, win_h)
        self.card.layout(win_w, win_h)
        _place(self.status, 0.5 * win_w, win_h - lay.MARGIN - 12)
        _place(self.banner, 0.5 * win_w, 0.5 * win_h + 60)
        self._help_at = lay.help_anchor(win_w, win_h, self.help.content_width)
        if self._help_at is not None:
            _place(self.help, *self._help_at)

    def draw(self, now: float) -> None:
        self.tower.draw()
        self.gap.draw()
        self.card.draw(now)
        self.speed.draw()
        if self.status.text:
            self.status.draw()
        if self.show_banner:
            self.banner.draw()
        if self._help_at is not None:
            self.help.draw()

    # ------------------------------------------------------------------- input

    def on_scroll(self, x: float, y: float, scroll_y: float, now: float) -> bool:
        """``True`` if the wheel was over the card (it pages); else the caller zooms."""
        if self.card.contains(x, y):
            if scroll_y:
                self.card.wheel(scroll_y, now)
            return True
        return False

    def on_press(self, x: float, y: float, now: float) -> bool:
        """``True`` if the click was on the card (next page)."""
        if self.card.contains(x, y):
            self.card.click(now)
            return True
        return False

    def on_tab(self, now: float) -> None:
        self.card.click(now)
