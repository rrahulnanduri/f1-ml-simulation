"""Screen-space layout of the HUD: pure functions from the window size to rectangles.

Screen pixels with the origin at the bottom-left (arcade's convention). Every panel is anchored
to a window corner, so calling these again after a resize keeps the layout correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MARGIN = 16.0

TOWER_W = 176.0
TOWER_PAD_TOP = 14.0
TOWER_HEADER_H = 12.0
TOWER_HEADER_GAP = 8.0
TOWER_ROW_H = 22.0
TOWER_PAD_BOTTOM = 8.0
TOWER_PAD_X = 12.0

GAP_W = 150.0
GAP_H = 58.0

CARD_W = 258.0
CARD_H = 82.0

SPEED_BLOCK_W = 230.0
"""Room the bottom-left speed readout needs (for placing the help line beside it)."""


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


def tower_height(n_rows: int) -> float:
    """Height of the timing tower for ``n_rows`` rows (it grows to fit them)."""
    n = max(int(n_rows), 0)
    return TOWER_PAD_TOP + TOWER_HEADER_H + TOWER_HEADER_GAP + n * TOWER_ROW_H + TOWER_PAD_BOTTOM


def tower_rect(win_w: float, win_h: float, n_rows: int) -> Rect:
    """Top-left glass panel."""
    h = tower_height(n_rows)
    return Rect(MARGIN, win_h - MARGIN - h, TOWER_W, h)


def tower_row_baseline(tower: Rect, i: int) -> float:
    """Text baseline of row ``i`` (0 = first row under the header)."""
    first_top = tower.top - TOWER_PAD_TOP - TOWER_HEADER_H - TOWER_HEADER_GAP
    return first_top - (i + 1) * TOWER_ROW_H + 7.0


def tower_row_centre(tower: Rect, i: int) -> float:
    first_top = tower.top - TOWER_PAD_TOP - TOWER_HEADER_H - TOWER_HEADER_GAP
    return first_top - (i + 0.5) * TOWER_ROW_H


def tower_separator_y(tower: Rect, i: int) -> float:
    """Hairline between row ``i`` and row ``i + 1``."""
    first_top = tower.top - TOWER_PAD_TOP - TOWER_HEADER_H - TOWER_HEADER_GAP
    return first_top - (i + 1) * TOWER_ROW_H


def gap_rect(win_w: float, win_h: float) -> Rect:
    """Top-right glass panel."""
    return Rect(win_w - MARGIN - GAP_W, win_h - MARGIN - GAP_H, GAP_W, GAP_H)


def card_rect(win_w: float, win_h: float) -> Rect:
    """Bottom-right paging card."""
    return Rect(win_w - MARGIN - CARD_W, MARGIN, CARD_W, CARD_H)


def speed_origin(win_w: float, win_h: float) -> tuple[float, float]:
    """Baseline-left of the big speed number (bottom-left, no panel)."""
    return MARGIN + 2.0, MARGIN + 22.0


def help_anchor(win_w: float, win_h: float, text_w: float) -> tuple[float, float] | None:
    """Bottom-centre position of the help line between the speed readout and the card, or
    ``None`` when it doesn't fit."""
    left = MARGIN + SPEED_BLOCK_W
    right = win_w - MARGIN - CARD_W - MARGIN
    if right - left < text_w:
        return None
    return 0.5 * (left + right), MARGIN + 4.0


def page_dots(card: Rect, n: int, spacing: float = 9.0) -> list[tuple[float, float]]:
    """Centres of ``n`` page dots stacked vertically on the card's right edge, top first."""
    cx = card.right - 12.0
    cy = card.y + 0.5 * card.h
    top = cy + 0.5 * spacing * (n - 1)
    return [(cx, top - i * spacing) for i in range(n)]


def page_offset_y(page_index: int, position: float, card_h: float) -> float:
    """Vertical shift of page ``page_index`` while the pager shows ``position`` (a float page
    index): the current page at 0, later pages below it (negative y), earlier ones above."""
    return (position - page_index) * card_h


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


def capsule_size_px(
    pixels_per_metre: float, length_m: float, width_m: float, min_l: float, min_w: float
) -> tuple[float, float]:
    """On-screen length and width of a car capsule: its world size, but at least
    ``min_l x min_w`` pixels so it stays visible zoomed out."""
    return max(length_m * pixels_per_metre, min_l), max(width_m * pixels_per_metre, min_w)


def capsule_points(
    cx: float, cy: float, heading: float, length: float, width: float, arc_segments: int = 6
) -> np.ndarray:
    """Outline of a capsule (a stadium: rectangle with semicircular ends) ``length x width``
    centred on ``(cx, cy)`` along ``heading``, counter-clockwise, shape
    ``(2 * (arc_segments + 1), 2)``."""
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
