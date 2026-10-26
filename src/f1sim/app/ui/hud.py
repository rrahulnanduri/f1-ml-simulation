"""HUD widgets in screen space, laid out as a bento grid of cards (:mod:`.layout`).

- :class:`Ticker` - the top-left micro line ``NOW: <date> · <clock> · <TRACK> · <mode>``,
  whose values decode (:mod:`.scramble`) when they change.
- :class:`Headline` - the wide bold track name, its second line faded.
- :class:`TimingTower` - the field, ranked.
- :class:`GapCard` - the gap to the reference, with a ↗ button that cycles the reference.
- :class:`SpeedCard` - the speed, gear/position and the lap clock.
- :class:`PagedWidget` - three sliding pages: sectors, speed vs the reference, gap bar.
- :class:`Footer` - the F1SIM wordmark, the controls as underlined links and a status.

Each widget creates its ``arcade.Text`` objects once (in ``__init__``) and afterwards only
changes their text, colour and position; cards are drawn by
:class:`~f1sim.app.ui.glass.GlassRenderer`. Call ``layout(win_w, win_h)`` after a resize and
``draw()`` every frame under the window's default (pixel) camera.
"""

from __future__ import annotations

import math
import time

import arcade
import numpy as np

from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme
from f1sim.app.ui.glass import GlassRenderer
from f1sim.app.ui.paging import Pager
from f1sim.app.ui.scramble import ScrambleLine
from f1sim.app.ui.timing import TowerRow, format_gap, format_sector

NO_REFERENCE = "No reference lap"
SPEED_MIN_KMH = 60.0
SPEED_MAX_KMH = 350.0
GAP_PX_PER_S = 48.0
PAD = lay.PAD


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


def _hairline(x0: float, x1: float, y: float, colour) -> None:
    arcade.draw_lrbt_rectangle_filled(x0, x1, y - 0.5, y + 0.5, colour)


def draw_button(glass: GlassRenderer, cx: float, cy: float) -> None:
    """The small red ↗ button of a card corner."""
    glass.disc(cx, cy, lay.BUTTON_R, theme.ACCENT)
    for x0, y0, x1, y1 in lay.arrow_strokes(cx, cy):
        glass.stroke(x0, y0, x1, y1, 1.5, theme.TEXT_PRIMARY)


def _in_button(card: lay.Rect, x: float, y: float) -> bool:
    cx, cy = lay.button_centre(card)
    return math.hypot(x - cx, y - cy) <= lay.BUTTON_R + 6.0


class TabularText:
    """A short number with every character in a fixed cell (digits as wide as the widest
    digit, signs as wide as the widest sign), so changing digits never shift the rest; for
    big numbers in a font with proportional figures."""

    SIGNS = "+-−–"

    def __init__(self, n_slots: int, px: float, colour, face: str, align: str = "left"):
        self.px, self.face, self.align = px, face, align
        self.chars = [
            theme.make_text("", px, colour, face=face, anchor_x="center") for _ in range(n_slots)
        ]
        self._widths: dict[str, float] = {}
        self.digit_w = max(self._measure(c) for c in "0123456789")
        self.sign_w = max(self._measure(c) for c in self.SIGNS)
        self.value = ""
        self._origin = (0.0, 0.0)

    def _measure(self, ch: str) -> float:
        w = self._widths.get(ch)
        if w is None:
            probe = theme.make_text(ch, self.px, (255, 255, 255, 255), face=self.face)
            w = self._widths[ch] = float(probe.content_width)
        return w

    def cell(self, ch: str) -> float:
        if ch.isdigit():
            return self.digit_w
        if ch in self.SIGNS:
            return self.sign_w
        return self._measure(ch)

    @property
    def width(self) -> float:
        return sum(self.cell(c) for c in self.value)

    def set(self, value: str) -> None:
        value = value[: len(self.chars)]
        if value == self.value:
            return
        self.value = value
        for i, text in enumerate(self.chars):
            _set(text, value[i] if i < len(value) else "")
        self.place(*self._origin)

    def colour(self, colour) -> None:
        for text in self.chars:
            _colour(text, colour)

    def place(self, x: float, baseline: float) -> None:
        """``x`` is the left edge (``align="left"``) or the right edge (``"right"``)."""
        self._origin = (x, baseline)
        left = x - self.width if self.align == "right" else x
        for i, ch in enumerate(self.value):
            w = self.cell(ch)
            _place(self.chars[i], left + 0.5 * w, baseline)
            left += w

    def draw(self) -> None:
        for text in self.chars[: len(self.value)]:
            text.draw()


# ------------------------------------------------------------------------ the top-left

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def ticker_date_time(t: float | None = None) -> tuple[str, str]:
    """``("21 OCT 2026", "14:32")``: the local date and clock for the ticker (English month
    abbreviations whatever the locale)."""
    lt = time.localtime(t)
    return (
        f"{lt.tm_mday:02d} {_MONTHS[lt.tm_mon - 1]} {lt.tm_year}",
        f"{lt.tm_hour:02d}:{lt.tm_min:02d}",
    )


class Ticker:
    """``NOW: 21 OCT 2026 · 14:32 · SILVERSTONE · REPLAY 1×``: micro text, each value
    decoding on its own when it changes (the text only re-lays out while that happens)."""

    def __init__(self, n_values: int = 4):
        self.prefix = theme.micro("Now:", theme.ACCENT)
        self.text = theme.micro("", theme.TEXT_PRIMARY)
        self.line = ScrambleLine(n_values)

    def set(self, values: list[str], now: float) -> None:
        self.line.set([v.upper() for v in values], now)

    def layout(self, win_w: float, win_h: float) -> None:
        x, y = lay.ticker_origin(win_w, win_h)
        _place(self.prefix, x, y)
        _place(self.text, x + self.prefix.content_width + 6.0, y)

    def draw(self, now: float) -> None:
        _set(self.text, self.line.text(now))
        self.prefix.draw()
        self.text.draw()


class Headline:
    """The track name in wide bold capitals, the second (and third) line fading."""

    MAX_W = lay.COL_W + 64.0

    def __init__(self, n_lines: int = 2):
        self.lines = [
            theme.make_text(
                "",
                theme.HEADLINE_PX,
                theme.white(theme.HEADLINE_FADE[i]),
                face="display",
            )
            for i in range(n_lines)
        ]
        self._win = (1.0, 1.0)

    def set(self, lines: list[str]) -> None:
        for text, value in zip(self.lines, list(lines) + [""] * len(self.lines), strict=False):
            value = value.upper()
            if text.text == value:
                continue
            text.text = value
            text.font_size = theme.px_to_pt(theme.HEADLINE_PX)
            if text.content_width > self.MAX_W:  # a long name: shrink it to fit
                text.font_size = theme.px_to_pt(theme.HEADLINE_PX * self.MAX_W / text.content_width)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        ys = lay.headline_baselines(win_w, win_h, len(self.lines))
        for text, y in zip(self.lines, ys, strict=True):
            _place(text, lay.MARGIN - 1.0, y)  # optical alignment of the wide capitals

    def draw(self) -> None:
        for text in self.lines:
            if text.text:
                text.draw()


class TimingTower:
    """Top-left card: a category label, ranked rows ``position, sliver, name, time-or-gap``."""

    def __init__(self, glass: GlassRenderer, capacity: int):
        self.glass = glass
        self.title = theme.micro("The field", theme.ACCENT)
        self.meta = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.capacity = int(capacity)
        self.pos = [
            theme.make_text("", theme.TIME_PX, theme.TEXT_TERTIARY, face="mono")
            for _ in range(capacity)
        ]
        self.names = [
            theme.make_text("", theme.NAME_PX, theme.TEXT_PRIMARY, face="medium", tracking_px=0.4)
            for _ in range(capacity)
        ]
        self.values = [
            theme.make_text("", theme.TIME_PX, theme.TEXT_SECONDARY, face="mono", anchor_x="right")
            for _ in range(capacity)
        ]
        self.rows: list[TowerRow] = []
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.tower_height(0))
        self._win = (1.0, 1.0)

    def set(self, title: str, meta: str, rows: list[TowerRow]) -> None:
        _set(self.title, title.upper())
        _set(self.meta, meta.upper())
        rows = rows[: self.capacity]
        relayout = len(rows) != len(self.rows)
        self.rows = rows
        if relayout:
            self.layout(*self._win)
        for i, row in enumerate(rows):
            _set(self.pos[i], row.position)
            _set(self.names[i], row.name.upper()[:16])
            _set(self.values[i], row.value)
            _colour(self.names[i], theme.TEXT_PRIMARY if row.position else theme.TEXT_SECONDARY)
            _colour(self.values[i], theme.SELF_GAP if row.is_self else theme.TEXT_SECONDARY)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        r = self.rect = lay.tower_rect(win_w, win_h, len(self.rows))
        _place(self.title, r.x + PAD, lay.label_baseline(r))
        _place(self.meta, r.right - PAD, lay.label_baseline(r))
        for i in range(self.capacity):
            base = lay.tower_row_baseline(r, i)
            _place(self.pos[i], r.x + PAD, base)
            _place(self.names[i], r.x + PAD + 30, base)
            _place(self.values[i], r.right - PAD, base)

    def draw(self) -> None:
        r = self.rect
        self.glass.panel(r)
        self.title.draw()
        self.meta.draw()
        x0, x1 = r.x + PAD, r.right - PAD
        for i, row in enumerate(self.rows):
            cy = lay.tower_row_centre(r, i)
            if row.is_self:
                arcade.draw_lrbt_rectangle_filled(
                    r.x + 1,
                    r.right - 1,
                    cy - 0.5 * lay.TOWER_ROW_H + 0.5,
                    cy + 0.5 * lay.TOWER_ROW_H - 0.5,
                    theme.SELF_ROW,
                )
            arcade.draw_lrbt_rectangle_filled(x0 + 18, x0 + 21, cy - 6, cy + 6, row.colour)
            if i + 1 < len(self.rows):
                _hairline(x0, x1, lay.tower_separator_y(r, i), theme.SEPARATOR)
            self.pos[i].draw()
            self.names[i].draw()
            self.values[i].draw()


# ------------------------------------------------------------------------ the gap card


class GapCard:
    """Top-right card: ``GAP``, the big gap to the reference and who that is; its ↗ button
    cycles the reference."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.title = theme.micro("Gap", theme.ACCENT)
        self.value = TabularText(7, theme.GAP_PX, theme.TEXT_TERTIARY, "headline")
        self.unit = theme.micro("s", theme.TEXT_SECONDARY)
        self.sub = theme.micro("", theme.TEXT_PRIMARY)
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)
        self.value.set("--.---")

    def set(self, gap: float | None, ref_label: str | None) -> None:
        if ref_label is None:
            self.value.set("--.---")
            self.value.colour(theme.TEXT_TERTIARY)
            _set(self.sub, NO_REFERENCE.upper())
            return
        self.value.set(format_gap(gap))
        if gap is None or not math.isfinite(gap):
            self.value.colour(theme.TEXT_TERTIARY)
        else:
            self.value.colour(theme.BEHIND if gap > 0.0 else theme.AHEAD)
        _set(self.sub, f"VS {ref_label.upper()}")

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.gap_rect(win_w, win_h)
        _place(self.title, r.x + PAD, lay.label_baseline(r))
        self.value.place(r.x + PAD - 1.0, r.y + 38)
        _place(self.sub, r.x + PAD, r.y + PAD)

    def contains(self, x: float, y: float) -> bool:
        return self.rect.contains(x, y)

    def draw(self) -> None:
        r = self.rect
        self.glass.panel(r)
        self.title.draw()
        self.value.draw()
        _place(self.unit, r.x + PAD + self.value.width + 6.0, r.y + 38)
        self.unit.draw()
        self.sub.draw()
        draw_button(self.glass, *lay.button_centre(r))


# ---------------------------------------------------------------------- the speed card


class SpeedCard:
    """Bottom-left card: the speed as a big number with a ``KM/H`` unit, gear/position, and
    the lap clock."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.title = theme.micro("Speed", theme.ACCENT)
        self.meta = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.value = TabularText(3, theme.SPEED_PX, theme.TEXT_PRIMARY, "headline", "right")
        self.unit = theme.micro("km/h", theme.TEXT_SECONDARY)
        self.clock_label = theme.micro("Lap", theme.TEXT_SECONDARY, anchor_x="right")
        self.clock = theme.make_text(
            "", theme.CLOCK_PX, theme.TEXT_PRIMARY, face="mono", anchor_x="right"
        )
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)

    def set(self, speed_kmh: float, extra: str, clock: str = "") -> None:
        self.value.set(f"{max(speed_kmh, 0.0):.0f}")
        _set(self.meta, extra.upper())
        _set(self.clock, clock)

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.speed_rect(win_w, win_h)
        _place(self.title, r.x + PAD, lay.label_baseline(r))
        _place(self.meta, r.right - PAD, lay.label_baseline(r))
        right = r.x + PAD + 3 * self.value.digit_w
        self.value.place(right, r.y + PAD + 4)
        _place(self.unit, right + 8.0, r.y + PAD + 4)
        _place(self.clock, r.right - PAD, r.y + PAD + 4)
        _place(self.clock_label, r.right - PAD, r.y + PAD + 26)

    def draw(self) -> None:
        self.glass.panel(self.rect)
        self.title.draw()
        self.meta.draw()
        self.value.draw()
        self.unit.draw()
        if self.clock.text:
            self.clock_label.draw()
            self.clock.draw()


# --------------------------------------------------------------------- paging widget


class PagedWidget:
    """Bottom-right card with three vertically sliding pages: sectors, speed vs the
    reference over the lap, and the gap bar. Its ↗ button (or a click anywhere on it, the
    wheel over it, or Tab) pages it. See :class:`~f1sim.app.ui.paging.Pager`."""

    N_PAGES = 3

    def __init__(self, glass: GlassRenderer, lap_length: float, page: int = 0):
        self.glass = glass
        self.length = float(lap_length)
        self.pager = Pager(self.N_PAGES, page)
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)
        self.titles = [theme.micro(t, theme.ACCENT) for t in ("Sectors", "Speed", "Gap")]
        self.metas = [theme.micro("", theme.TEXT_PRIMARY) for _ in range(3)]
        # Page 1
        self.sector_labels = [theme.micro(f"S{k + 1}", theme.TEXT_SECONDARY) for k in range(3)]
        self.sector_times = [
            theme.make_text("--.---", theme.SECTOR_TIME_PX, theme.TEXT_TERTIARY, face="mono")
            for _ in range(3)
        ]
        self.sector_states = ["unset"] * 3
        # Page 2
        self._ref_speed: tuple[np.ndarray, np.ndarray] | None = None
        self._self_speed: tuple[np.ndarray, np.ndarray] | None = None
        self._ref_lines: list[list[tuple[float, float]]] = []
        self._self_lines: list[list[tuple[float, float]]] = []
        self.marker_s: float | None = None
        self.no_ref = [theme.micro(NO_REFERENCE, theme.TEXT_SECONDARY) for _ in range(2)]
        # Page 3
        self.gap: float | None = None
        self.gap_value = theme.make_text(
            "--.---", theme.GAP_PAGE_PX, theme.TEXT_TERTIARY, face="mono", anchor_x="right"
        )
        self.gap_at = theme.micro("", theme.TEXT_SECONDARY)
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
        """Short reference name for the page metadata, or ``None`` without a reference."""
        self.has_ref = ref_name is not None
        name = (ref_name or "").upper()
        _set(self.metas[0], "")
        _set(self.metas[1], f"VS {name}" if name else "")
        _set(self.metas[2], f"TO {name}" if name else "")

    def set_sectors(self, times: list[float], states: list[str]) -> None:
        n = len(self.sector_times)
        times = (list(times) + [math.nan] * n)[:n]
        states = (list(states) + ["unset"] * n)[:n]
        self.sector_states = states
        for text, t, st in zip(self.sector_times, times, states, strict=True):
            _set(text, format_sector(t))
            _colour(text, theme.TEXT_TERTIARY if st == "unset" else theme.TEXT_PRIMARY)

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
            _colour(self.gap_value, theme.TEXT_TERTIARY)
        else:
            _colour(self.gap_value, theme.BEHIND if gap > 0.0 else theme.AHEAD)
        _set(self.gap_at, at_label.upper())

    # -------------------------------------------------------------------- layout

    def _content(self) -> lay.Rect:
        """The page content area: left of the button column, under the title."""
        r = self.rect
        right = lay.button_centre(r)[0] - lay.BUTTON_R - 12.0
        return lay.Rect(r.x + PAD, r.y + PAD, right - r.x - PAD, r.h - PAD - 40.0)

    def _polylines(self, trace) -> list[list[tuple[float, float]]]:
        if trace is None:
            return []
        s, v = trace
        c = self._content()
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
        self.glass.panel(r)
        ctx = self.glass.ctx
        pos = self.pager.position(now)
        # Window pixels, inside the hairline (the screen scales it by the pixel ratio).
        ctx.scissor = (int(r.x + 1), int(r.y + 1), int(r.w - 2), int(r.h - 2))
        try:
            for i, draw_page in enumerate((self._draw_sectors, self._draw_speed, self._draw_gap)):
                dy = lay.page_offset_y(i, pos, r.h)
                if abs(dy) < r.h:
                    y = lay.label_baseline(r) + dy
                    _place(self.titles[i], r.x + PAD, y)
                    self.titles[i].draw()
                    if self.metas[i].text:
                        _place(self.metas[i], r.x + PAD + self.titles[i].content_width + 8, y)
                        self.metas[i].draw()
                    draw_page(dy)
        finally:
            ctx.scissor = None
        bx, by = lay.button_centre(r)
        draw_button(self.glass, bx, by)
        for k, (cx, cy) in enumerate(lay.page_dots(r, self.N_PAGES)):
            colour = theme.DOT_ACTIVE if k == self.pager.page else theme.DOT_INACTIVE
            self.glass.disc(cx, cy, 2.5, colour)

    def _draw_sectors(self, dy: float) -> None:
        c = self._content()
        gap = 8.0
        w = (c.w - 2 * gap) / 3.0
        for k in range(3):
            bx = c.x + k * (w + gap)
            colour = theme.SECTOR_COLOURS[self.sector_states[k]]
            arcade.draw_lrbt_rectangle_filled(bx, bx + w, c.y + dy, c.y + 2 + dy, colour)
            _place(self.sector_labels[k], bx, c.y + 30 + dy)
            _place(self.sector_times[k], bx, c.y + 12 + dy)
            self.sector_labels[k].draw()
            self.sector_times[k].draw()

    def _draw_no_ref(self, k: int, dy: float) -> None:
        c = self._content()
        _place(self.no_ref[k], c.x, c.y + 12 + dy)
        self.no_ref[k].draw()

    def _draw_speed(self, dy: float) -> None:
        if not self.has_ref:
            self._draw_no_ref(0, dy)
            return
        c = self._content()
        for frac in (0.0, 0.5, 1.0):
            _hairline(c.x, c.right, c.y + frac * c.h + dy, theme.CHART_GRID)
        for lines, colour, width in (
            (self._ref_lines, theme.CHART_REF, 1.0),
            (self._self_lines, theme.CHART_SELF, 1.5),
        ):
            for line in lines:
                # Shifted copies only while the page slides
                pts = line if dy == 0.0 else [(x, y + dy) for x, y in line]
                arcade.draw_line_strip(pts, colour, width)
        if self.marker_s is not None and math.isfinite(self.marker_s):
            mx = c.x + min(max(self.marker_s / self.length, 0.0), 1.0) * c.w
            arcade.draw_lrbt_rectangle_filled(
                mx - 0.5, mx + 0.5, c.y + dy, c.top + dy, theme.CHART_MARKER
            )

    def _draw_gap(self, dy: float) -> None:
        if not self.has_ref:
            self._draw_no_ref(1, dy)
            return
        c = self._content()
        _place(self.gap_value, c.right, c.y + 22 + dy)
        _place(self.gap_at, c.x, c.y + 22 + dy)
        self.gap_value.draw()
        self.gap_at.draw()
        x0, x1, y = c.x, c.right, c.y + 4 + dy
        arcade.draw_lrbt_rectangle_filled(x0, x1, y - 1, y + 1, theme.GAP_TRACK)
        cx = 0.5 * (x0 + x1)
        g = self.gap
        if g is not None and math.isfinite(g) and g != 0.0:
            length = min(abs(g) * GAP_PX_PER_S, 0.5 * (x1 - x0))
            colour = theme.BEHIND if g > 0.0 else theme.AHEAD
            # Behind grows to the right of the centre tick, ahead to the left.
            a, b = (cx, cx + length) if g > 0.0 else (cx - length, cx)
            arcade.draw_lrbt_rectangle_filled(a, b, y - 1, y + 1, colour)
        arcade.draw_lrbt_rectangle_filled(cx - 0.5, cx + 0.5, y - 4, y + 4, theme.CHART_MARKER)


# --------------------------------------------------------------------------- footer


class Footer:
    """The black strip along the bottom: the F1SIM wordmark, the controls as underlined
    UPPERCASE micro links, and a status on the right."""

    def __init__(self, glass: GlassRenderer, controls: list[str]):
        self.glass = glass
        self.wordmark = theme.make_text(
            "F1SIM", theme.WORDMARK_PX, theme.TEXT_PRIMARY, face="display"
        )
        self.links = [theme.micro(c, theme.TEXT_PRIMARY) for c in controls]
        self.status = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.rect = lay.Rect(0, 0, 1, lay.FOOTER_H)
        self._shown = 0
        self._status_shown = False
        self._win = (1.0, 1.0)

    def set_status(self, text: str, colour=theme.TEXT_PRIMARY) -> None:
        text = text.upper()
        if text != self.status.text:
            self.status.text = text
            self.layout(*self._win)
        _colour(self.status, colour)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        r = self.rect = lay.footer_rect(win_w, win_h)
        mid = r.y + 0.5 * r.h
        x0, xs, sx = lay.footer_positions(
            r,
            self.wordmark.content_width,
            [t.content_width for t in self.links],
            self.status.content_width if self.status.text else 0.0,
        )
        _place(self.wordmark, x0, mid - 5)
        for text, x in zip(self.links, xs, strict=False):
            _place(text, x, mid - 3)
        self._shown = len(xs)
        self._status_shown = sx is not None
        if sx is not None:
            _place(self.status, r.right - 16.0, mid - 3)

    def draw(self) -> None:
        self.glass.panel(self.rect)
        self.wordmark.draw()
        for text in self.links[: self._shown]:
            text.draw()
            y = text.y - 4
            _hairline(text.x, text.x + text.content_width - 1.0, y, theme.LINK_UNDERLINE)
        if self._status_shown:
            self.status.draw()


class Banner:
    """A top-centre card for transient messages (e.g. OFF TRACK)."""

    def __init__(self, glass: GlassRenderer, label: str, text: str):
        self.glass = glass
        self.label = theme.micro(label, theme.ACCENT, anchor_x="center")
        self.text = theme.make_text(
            text.upper(), 18, theme.TEXT_PRIMARY, face="display", anchor_x="center"
        )
        self.rect = lay.Rect(0, 0, 1, 1)

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.banner_rect(win_w, win_h)
        cx = r.x + 0.5 * r.w
        _place(self.label, cx, lay.label_baseline(r))
        _place(self.text, cx, r.y + PAD)

    def draw(self) -> None:
        self.glass.panel(self.rect)
        self.label.draw()
        self.text.draw()


# ----------------------------------------------------------------------- whole HUD


class Hud:
    """All HUD widgets of a viewer window. Owns the input handling of the cards."""

    def __init__(
        self,
        glass: GlassRenderer,
        tower_capacity: int,
        lap_length: float,
        controls: list[str],
    ):
        self.glass = glass
        self.ticker = Ticker()
        self.headline = Headline()
        self.tower = TimingTower(glass, tower_capacity)
        self.gap = GapCard(glass)
        self.speed = SpeedCard(glass)
        self.card = PagedWidget(glass, lap_length)
        self.footer = Footer(glass, controls)
        self.banner = Banner(glass, "Track limits", "Off track")
        self.show_banner = False
        self._win = (1.0, 1.0)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        self.ticker.layout(win_w, win_h)
        self.headline.layout(win_w, win_h)
        self.tower.layout(win_w, win_h)
        self.gap.layout(win_w, win_h)
        self.speed.layout(win_w, win_h)
        self.card.layout(win_w, win_h)
        self.footer.layout(win_w, win_h)
        self.banner.layout(win_w, win_h)

    def stage_centre(self) -> tuple[float, float]:
        """Where the followed car belongs on screen."""
        return lay.stage_centre(*self._win)

    def draw(self, now: float) -> None:
        # Each widget's text is batched (one projection change each: see theme.text_batch),
        # after its own panel and before the next widget
        for draw in (
            lambda: self.ticker.draw(now),
            self.headline.draw,
            self.tower.draw,
            self.gap.draw,
            self.speed.draw,
            lambda: self.card.draw(now),
            self.footer.draw,
            *((self.banner.draw,) if self.show_banner else ()),
        ):
            with theme.text_batch():
                draw()

    # ------------------------------------------------------------------- input

    def on_scroll(self, x: float, y: float, scroll_y: float, now: float) -> bool:
        """``True`` if the wheel was over the widget (it pages); else the caller zooms."""
        if self.card.contains(x, y):
            if scroll_y:
                self.card.wheel(scroll_y, now)
            return True
        return False

    def on_press(self, x: float, y: float, now: float) -> str | None:
        """What a left click hit: ``"widget"`` (it paged), ``"gap"`` (the caller cycles the
        reference) or ``None``."""
        if self.card.contains(x, y):
            self.card.click(now)
            return "widget"
        if self.gap.contains(x, y):
            return "gap"
        return None

    def on_tab(self, now: float) -> None:
        self.card.click(now)
