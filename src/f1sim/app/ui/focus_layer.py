"""Drawing for the training view's focus overlay (:mod:`f1sim.app.focus`).

Every element is scaled by the fade level ``f`` (0..1), so the overlay eases in and out:

- :meth:`FocusLayer.draw_backdrop` (in the scene, window pixels, after the track): a dark
  veil over the track, then the three sectors traced along the centre line in
  :data:`SECTOR_COLOURS`.
- :meth:`FocusLayer.draw_pole` and :meth:`FocusLayer.draw_ring` (in the scene, after the
  field): the real pole car in its team livery, and a soft ring round the best AI car.
- :meth:`FocusLayer.draw_callouts` (after the scene, under the HUD): for each car a 1 px
  leader line from a dot on the car to a small dark card with its name, lap clock, a bar in
  the three sector colours that fills as the lap goes, and the three sector times; the AI's
  card adds its gap to the pole in each sector, green when it's quicker.

:func:`card_rect` places the cards (pure): the AI's above right of its car, the pole's below
left, kept inside the window. :func:`draw_card` is the card itself, which the "How it works"
drawer (:mod:`f1sim.app.ui.drawer`) also uses, so the two read as one style.
"""

from __future__ import annotations

import math

import arcade
import numpy as np

from f1sim.app import focus as fo
from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme
from f1sim.app.ui.hud import _place, _set

SECTOR_COLOURS = (
    theme.hex_colour("#2FC8F0"),  # S1 cyan
    theme.hex_colour("#B57BFF"),  # S2 violet
    theme.hex_colour("#FFC23D"),  # S3 amber
)
VEIL_ALPHA = 0.55
"""Darkness of the veil over the track at full focus."""
FIELD_FADE = 0.82
"""How much the rest of the field fades at full focus (0 = not at all, 1 = gone)."""
CARD_W = 268.0
CARD_H = 80.0
"""Card height; the AI's card is :data:`DELTA_H` taller for its gaps to the pole."""
DELTA_H = 16.0
ELBOW = 34.0
"""Leader line: up (or down) and out this far from the car before the card."""
TRACE_STEP_M = 4.0
"""Spacing of the points of the sector traces."""


def card_rect(
    sx: float, sy: float, side: int, win_w: float, win_h: float, h: float = CARD_H
) -> tuple[lay.Rect, tuple[float, float]]:
    """``(card, elbow)`` for a car at screen ``(sx, sy)``: ``side`` +1 puts the card above
    right, -1 below left. The card (``h`` high) stays inside the window margins."""
    ex, ey = sx + side * ELBOW, sy + side * ELBOW
    x = ex + 8.0 if side > 0 else ex - 8.0 - CARD_W
    y = ey if side > 0 else ey - h
    x = min(max(x, lay.MARGIN), win_w - lay.MARGIN - CARD_W)
    y = min(max(y, lay.BOTTOM_Y), win_h - lay.MARGIN - h)
    return lay.Rect(x, y, CARD_W, h), (ex, ey)


def sector_polylines(track, marks: np.ndarray) -> list[np.ndarray]:
    """The centre line split at the sector marks, ``(M, 2)`` world points each, about
    :data:`TRACE_STEP_M` apart (each piece shares its end point with the next)."""
    d = track.data
    s = np.append(d.s, track.length)
    x, y = np.append(d.x, d.x[0]), np.append(d.y, d.y[0])
    out = []
    for k in range(fo.N_SECTORS):
        a, b = marks[k], marks[k + 1]
        n = max(int((b - a) / TRACE_STEP_M), 2)
        q = np.linspace(a, b, n + 1)
        out.append(np.column_stack([np.interp(q, s, x), np.interp(q, s, y)]))
    return out


def _alpha(colour, a: float) -> tuple[int, int, int, int]:
    base = colour[3] if len(colour) > 3 else 255
    return (colour[0], colour[1], colour[2], int(round(base * min(max(a, 0.0), 1.0))))


CARD_FILL = (10, 10, 12, 235)
CARD_OUTLINE = 0.14
"""Alpha of the card's 1 px outline (primary text colour)."""
STRIPE_W = 2.0


def draw_card(rect: lay.Rect, colour, f: float = 1.0) -> None:
    """A dark card with a faint 1 px outline and a ``colour`` stripe down its left edge,
    faded by ``f``."""
    fill = (*CARD_FILL[:3], int(CARD_FILL[3] * f))
    arcade.draw_lrbt_rectangle_filled(rect.x, rect.right, rect.y, rect.top, fill)
    arcade.draw_lrbt_rectangle_outline(
        rect.x, rect.right, rect.y, rect.top, _alpha(theme.TEXT_PRIMARY, CARD_OUTLINE * f), 1.0
    )
    arcade.draw_lrbt_rectangle_filled(
        rect.x, rect.x + STRIPE_W, rect.y, rect.top, _alpha(colour, f)
    )


class _Card:
    """The text objects of one callout card."""

    def __init__(self, colour, with_deltas: bool):
        self.colour = colour
        self.label = theme.micro("", colour)
        self.clock = theme.make_text("", 15, theme.TEXT_PRIMARY, face="mono", anchor_x="right")
        self.times = [theme.make_text("", 11, theme.TEXT_PRIMARY, face="mono") for _ in range(3)]
        self.deltas = (
            [theme.make_text("", 10, theme.AHEAD, face="mono") for _ in range(3)]
            if with_deltas
            else []
        )


class FocusLayer:
    """See the module docstring. ``pole`` is the real lap, or None if the track has none."""

    def __init__(self, track, pole: fo.PoleGhost | None):
        from f1sim.app.ui.pixelcar import CarLayer

        self.marks = fo.sector_marks(track.data.sectors_s, track.length)
        self.traces = sector_polylines(track, self.marks)
        self.pole = pole
        self.pole_car = None
        if pole is not None:
            lv = theme.livery(pole.team)
            self.pole_car = CarLayer([""], [lv], [True], true_size=True)
            self.pole_car.only_highlight_label = True
        self.ai_card = _Card(theme.ACCENT, with_deltas=pole is not None)
        pole_colour = theme.team_colour(pole.team) if pole is not None else None
        self.pole_card = _Card(pole_colour or theme.TEXT_SECONDARY, with_deltas=False)

    # ----------------------------------------------------------------- the scene

    def draw_backdrop(self, camera, win_w: float, win_h: float, f: float) -> None:
        arcade.draw_lrbt_rectangle_filled(0, win_w, 0, win_h, (6, 6, 8, int(255 * VEIL_ALPHA * f)))
        for pts, colour in zip(self.traces, SECTOR_COLOURS, strict=True):
            sx, sy = camera.world_to_screen(pts[:, 0], pts[:, 1])
            arcade.draw_line_strip(
                list(zip(sx.tolist(), sy.tolist(), strict=True)), _alpha(colour, 0.9 * f), 2.5
            )

    def draw_pole(self, camera, pose, f: float, pixel_ratio: float) -> None:
        if self.pole_car is None or pose is None:
            return
        self.pole_car.sprites.sprites[0].alpha = int(255 * f)
        self.pole_car.draw(camera, [(pose.x, pose.y, pose.heading)], None, pixel_ratio)

    def draw_ring(self, sx: float, sy: float, f: float) -> None:
        arcade.draw_circle_filled(sx, sy, 22.0, _alpha(theme.ACCENT, 0.16 * f), 32)
        arcade.draw_circle_outline(sx, sy, 22.0, _alpha(theme.ACCENT, 0.9 * f), 1.5, 32)

    # -------------------------------------------------------------- the callouts

    def draw_callouts(
        self,
        f: float,
        win_w: float,
        win_h: float,
        ai: tuple[float, float, fo.Callout] | None,
        pole: tuple[float, float, fo.Callout] | None,
    ) -> None:
        for item, card, side in ((ai, self.ai_card, 1), (pole, self.pole_card, -1)):
            if item is not None:
                self._draw_card(card, *item, side, win_w, win_h, f)

    def _draw_card(self, card: _Card, sx, sy, c: fo.Callout, side, win_w, win_h, f) -> None:
        h = CARD_H + (DELTA_H if card.deltas and c.deltas is not None else 0.0)
        rect, (ex, ey) = card_rect(sx, sy, side, win_w, win_h, h)
        line = _alpha(theme.TEXT_PRIMARY, 0.55 * f)
        # Leader: a dot on the car, out to the elbow, then along to the card's edge
        edge_x = rect.x if side > 0 else rect.right
        arcade.draw_circle_filled(sx, sy, 2.5, _alpha(card.colour, f), 12)
        arcade.draw_line(sx, sy, ex, ey, line, 1.0)
        arcade.draw_line(ex, ey, edge_x, ey, line, 1.0)
        draw_card(rect, card.colour, f)
        pad = 12.0
        x0, x1 = rect.x + pad, rect.right - pad
        top = rect.top - pad
        # The sector bar: three segments, each filling in its colour as the lap goes
        bar_y = top - 30.0
        gap = 3.0
        x = x0
        width = x1 - x0 - 2 * gap
        for k in range(fo.N_SECTORS):
            w = width * c.widths[k]
            arcade.draw_lrbt_rectangle_filled(
                x, x + w, bar_y, bar_y + 3, _alpha(SECTOR_COLOURS[k], 0.22 * f)
            )
            if c.fills[k] > 0:
                arcade.draw_lrbt_rectangle_filled(
                    x, x + w * c.fills[k], bar_y, bar_y + 3, _alpha(SECTOR_COLOURS[k], f)
                )
            x += w + gap
        with theme.text_batch():
            _set(card.label, c.label.upper())
            _place(card.label, x0 + 2, top - 8)
            _set(card.clock, _clock(c.clock))
            _place(card.clock, x1, top - 13)
            texts = [card.label, card.clock]
            col_w = (x1 - x0) / 3.0
            for k in range(fo.N_SECTORS):
                t = card.times[k]
                v = c.sector_times[k]
                _set(t, f"S{k + 1} {v:.3f}" if math.isfinite(v) else f"S{k + 1} --.---")
                _place(t, x0 + k * col_w, bar_y - 18)
                texts.append(t)
                if card.deltas and c.deltas is not None:
                    d = card.deltas[k]
                    dv = c.deltas[k]
                    _set(d, f"{dv:+.3f}" if math.isfinite(dv) else "")
                    base = theme.AHEAD if (math.isfinite(dv) and dv <= 0) else theme.BEHIND
                    _place(d, x0 + k * col_w + 18, bar_y - 34)
                    if d.text:
                        texts.append((d, base))
            for t in texts:
                text, colour = t if isinstance(t, tuple) else (t, None)
                if colour is None:
                    colour = card.colour if text is card.label else theme.TEXT_PRIMARY
                want = _alpha(colour, f)
                if tuple(text.color) != want:
                    text.color = want
                text.draw()


def _clock(t: float) -> str:
    if not math.isfinite(t):
        return "-:--.---"
    if t < 0:
        return f"-{abs(t):.1f}"
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:06.3f}"
