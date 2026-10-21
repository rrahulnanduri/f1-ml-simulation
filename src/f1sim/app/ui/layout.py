"""Screen-space layout of the HUD: pure functions from the window size to rectangles.

Screen pixels with the origin at the bottom-left (arcade's convention). The HUD is a bento grid
of cards on an 8 px grid, anchored to the window corners, so calling these again after a resize
keeps the layout correct::

    +------------------------------------------------------------------+
    | NOW: ... ticker                                      +-- gap --+ |
    | SILVERSTONE   headline                               |  card   | |
    | REPLAY                                               +---------+ |
    | +- tower -+                                                      |
    | |  card   |               (the track, the followed car           |
    | +---------+                at the stage centre)                  |
    | +- speed -+                                          +- widget-+ |
    | +---------+                                          +---------+ |
    | +------------------------ footer --------------------------------+ |
    +------------------------------------------------------------------+

The left and right columns are ``COL_W`` wide, cards are ``GUTTER`` apart and ``MARGIN`` from
the window edges; the footer spans the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MARGIN = 16.0
GUTTER = 8.0
COL_W = 256.0
PAD = 12.0
"""Inner padding of a card."""
MICRO_CAP = 7.0
"""Cap height of a 10 px micro label: its baseline sits this far below the caps' top."""

FOOTER_H = 40.0
CARD_H = 96.0
"""Height of the gap, speed and widget cards."""
BOTTOM_Y = MARGIN + FOOTER_H + GUTTER
"""Bottom edge of the bottom row of cards (above the footer)."""

HEAD_H = 88.0
"""Height of the ticker and headline above the timing tower."""
HEADLINE_LEAD = 28.0
"""Baseline-to-baseline distance of the headline lines (tight: = the font size)."""

TOWER_HEADER_H = 32.0
TOWER_ROW_H = 24.0
TOWER_PAD_BOTTOM = 8.0

BUTTON_R = 10.0
"""Radius of the round ↗ buttons in card corners."""


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: left ``x``, bottom ``y``, width ``w``, height ``h``."""

    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + 0.5 * self.w, self.y + 0.5 * self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.top

    def inset(self, dx: float, dy: float | None = None) -> Rect:
        dy = dx if dy is None else dy
        return Rect(self.x + dx, self.y + dy, max(0.0, self.w - 2 * dx), max(0.0, self.h - 2 * dy))

    def overlaps(self, other: Rect) -> bool:
        return (
            self.x < other.right
            and other.x < self.right
            and self.y < other.top
            and other.y < self.top
        )


# ------------------------------------------------------------------------- the cards


def footer_rect(win_w: float, win_h: float) -> Rect:
    """The black strip along the bottom: wordmark, controls and status."""
    return Rect(MARGIN, MARGIN, max(win_w - 2 * MARGIN, 0.0), FOOTER_H)


def ticker_origin(win_w: float, win_h: float) -> tuple[float, float]:
    """Baseline-left of the micro ticker line, its caps flush with the top margin."""
    return MARGIN, win_h - MARGIN - MICRO_CAP


def headline_baselines(win_w: float, win_h: float, n_lines: int = 2) -> list[float]:
    """Baselines of the headline lines, top first, ``HEADLINE_LEAD`` apart; the last one sits
    16 px above the timing tower."""
    last = win_h - MARGIN - HEAD_H + 16.0
    return [last + (n_lines - 1 - i) * HEADLINE_LEAD for i in range(n_lines)]


def tower_height(n_rows: int) -> float:
    """Height of the timing tower card for ``n_rows`` rows (it grows to fit them)."""
    return TOWER_HEADER_H + max(int(n_rows), 0) * TOWER_ROW_H + TOWER_PAD_BOTTOM


def tower_rect(win_w: float, win_h: float, n_rows: int) -> Rect:
    """Top-left card, under the ticker and headline."""
    h = tower_height(n_rows)
    return Rect(MARGIN, win_h - MARGIN - HEAD_H - h, COL_W, h)


def _first_row_top(tower: Rect) -> float:
    return tower.top - TOWER_HEADER_H


def tower_row_centre(tower: Rect, i: int) -> float:
    return _first_row_top(tower) - (i + 0.5) * TOWER_ROW_H


def tower_row_baseline(tower: Rect, i: int) -> float:
    """Text baseline of row ``i`` (0 = first row under the header): 12 px text centred."""
    return tower_row_centre(tower, i) - 4.0


def tower_separator_y(tower: Rect, i: int) -> float:
    """Hairline between row ``i`` and row ``i + 1``."""
    return _first_row_top(tower) - (i + 1) * TOWER_ROW_H


def gap_rect(win_w: float, win_h: float) -> Rect:
    """Top-right card."""
    return Rect(win_w - MARGIN - COL_W, win_h - MARGIN - CARD_H, COL_W, CARD_H)


def card_rect(win_w: float, win_h: float) -> Rect:
    """Bottom-right paging widget card, above the footer."""
    return Rect(win_w - MARGIN - COL_W, BOTTOM_Y, COL_W, CARD_H)


def speed_rect(win_w: float, win_h: float) -> Rect:
    """Bottom-left speed card, above the footer."""
    return Rect(MARGIN, BOTTOM_Y, COL_W, CARD_H)


def banner_rect(win_w: float, win_h: float) -> Rect:
    """Top-centre card for a transient banner (e.g. OFF TRACK)."""
    w = COL_W
    return Rect(round(0.5 * (win_w - w)), win_h - MARGIN - 56.0, w, 56.0)


def stage_centre(win_w: float, win_h: float) -> tuple[float, float]:
    """Where the followed car is kept: the middle of the window between the footer and the
    top edge, clear of the corner cards."""
    return 0.5 * win_w, 0.5 * (BOTTOM_Y + win_h - MARGIN)


def card_rects(win_w: float, win_h: float, n_rows: int) -> dict[str, Rect]:
    """Every card of the layout by name (for checks and hit testing)."""
    return {
        "tower": tower_rect(win_w, win_h, n_rows),
        "gap": gap_rect(win_w, win_h),
        "speed": speed_rect(win_w, win_h),
        "widget": card_rect(win_w, win_h),
        "footer": footer_rect(win_w, win_h),
    }


# ------------------------------------------------------------------- inside the cards


def label_baseline(card: Rect) -> float:
    """Baseline of a card's top micro label (caps ``PAD`` below the top edge)."""
    return card.top - PAD - MICRO_CAP


def button_centre(card: Rect) -> tuple[float, float]:
    """Centre of the ↗ button in a card's top-right corner."""
    return card.right - PAD - BUTTON_R, card.top - PAD - BUTTON_R


def page_dots(card: Rect, n: int, spacing: float = 10.0) -> list[tuple[float, float]]:
    """Centres of ``n`` page dots stacked vertically under the card's ↗ button, top first."""
    cx = button_centre(card)[0]
    cy = card.y + 0.5 * (card.h - 2 * BUTTON_R - PAD)
    top = cy + 0.5 * spacing * (n - 1)
    return [(cx, top - i * spacing) for i in range(n)]


def page_offset_y(page_index: int, position: float, card_h: float) -> float:
    """Vertical shift of page ``page_index`` while the pager shows ``position`` (a float page
    index): the current page at 0, later pages below it (negative y), earlier ones above."""
    return (position - page_index) * card_h


def footer_positions(
    footer: Rect,
    wordmark_w: float,
    link_ws: list[float],
    status_w: float,
    gap: float = 24.0,
) -> tuple[float, list[float], float | None]:
    """Left x of the wordmark, of each control link that fits (left to right, ``gap`` apart,
    after the wordmark and a 32 px break) and of the right-aligned status (``None`` if it's
    hidden). The controls come first: the status is hidden when they don't all fit beside it,
    and links that don't fit even then are dropped from the end."""
    x0 = footer.x + 16.0
    right = footer.right - 16.0
    start = x0 + wordmark_w + 32.0

    def place(limit: float) -> list[float]:
        xs: list[float] = []
        x = start
        for w in link_ws:
            if x + w > limit:
                break
            xs.append(x)
            x += w + gap
        return xs

    if status_w > 0:
        status_x = right - status_w
        xs = place(status_x - gap)
        if len(xs) == len(link_ws):
            return x0, xs, status_x
    return x0, place(right), None


def stack_labels(boxes: list[Rect], order: list[int], step: float) -> list[float]:
    """New bottom ``y`` of each label box so that none overlap: boxes are placed in
    ``order`` (the first keeps its place), each moved down by ``step`` until it clears the
    ones already placed."""
    ys = [b.y for b in boxes]
    placed: list[Rect] = []
    for i in order:
        b = boxes[i]
        r = Rect(b.x, ys[i], b.w, b.h)
        for _ in range(len(boxes) + 1):
            if not any(r.overlaps(p) for p in placed):
                break
            r = Rect(r.x, r.y - step, r.w, r.h)
        ys[i] = r.y
        placed.append(r)
    return ys


# --------------------------------------------------------------------------- shapes


def rounded_rect_points(rect: Rect, radius: float, arc_segments: int = 6) -> np.ndarray:
    """Outline of a rounded rectangle, counter-clockwise from the bottom-right corner arc,
    shape ``(4 * (arc_segments + 1), 2)``. ``radius`` is clamped to half the shorter side."""
    r = max(0.0, min(radius, 0.5 * rect.w, 0.5 * rect.h))
    corners = (
        (rect.right - r, rect.y + r, -0.5 * math.pi),  # bottom-right
        (rect.right - r, rect.top - r, 0.0),  # top-right
        (rect.x + r, rect.top - r, 0.5 * math.pi),  # top-left
        (rect.x + r, rect.y + r, math.pi),  # bottom-left
    )
    pts = []
    for cx, cy, a0 in corners:
        a = a0 + np.linspace(0.0, 0.5 * math.pi, arc_segments + 1)
        pts.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    return np.vstack(pts)


def capsule_points(
    cx: float, cy: float, heading: float, length: float, width: float, arc_segments: int = 6
) -> np.ndarray:
    """Outline of a capsule (a stadium: rectangle with semicircular ends) ``length x width``
    centred on ``(cx, cy)`` along ``heading``, counter-clockwise, shape
    ``(2 * (arc_segments + 1), 2)``. A capsule as long as it is wide is a disc."""
    r = 0.5 * min(width, length)
    a = max(0.0, 0.5 * length - r)
    front = np.linspace(-0.5 * math.pi, 0.5 * math.pi, arc_segments + 1)
    back = front + math.pi
    local = np.vstack(
        [
            np.column_stack([a + r * np.cos(front), r * np.sin(front)]),
            np.column_stack([-a + r * np.cos(back), r * np.sin(back)]),
        ]
    )
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def arrow_strokes(
    cx: float, cy: float, size: float = 3.5
) -> list[tuple[float, float, float, float]]:
    """The ↗ glyph as three strokes ``(x0, y0, x1, y1)`` centred on ``(cx, cy)``: the shaft
    from bottom-left to top-right and the two halves of the head."""
    tip = (cx + size, cy + size)
    return [
        (cx - size, cy - size, *tip),
        (tip[0] - 1.9 * size, tip[1], *tip),
        (tip[0], tip[1] - 1.9 * size, *tip),
    ]
