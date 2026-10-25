"""The training view's HUD (:mod:`f1sim.app.training`), on the replay HUD's bento grid.

- :class:`~f1sim.app.ui.hud.Ticker`, :class:`~f1sim.app.ui.hud.Headline`,
  :class:`~f1sim.app.ui.hud.TimingTower` (the live order) and
  :class:`~f1sim.app.ui.hud.Footer`, as in the viewers.
- :class:`GenerationCard` (top right, where the gap card sits) - the generation on screen as
  a big number, its stage, its best car's lap, and the newest generation trained.
- :class:`FieldCard` (bottom left) - cars still running out of the population, the playback
  speed and the episode clock.
- :class:`ProgressCard` (bottom right) - the best lap of every generation so far, as a chart,
  and the run's best.

Like :mod:`.hud`, each widget creates its ``arcade.Text`` objects once and afterwards only
changes their text, colour and position.
"""

from __future__ import annotations

import math

import arcade
import numpy as np

from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme
from f1sim.app.ui.glass import GlassRenderer
from f1sim.app.ui.hud import (
    Footer,
    Headline,
    TabularText,
    Ticker,
    TimingTower,
    _hairline,
    _place,
    _set,
)
from f1sim.app.ui.timing import format_lap

PAD = lay.PAD


class GenerationCard:
    """Top-right card: ``GENERATION`` and the stage, the generation number, its best."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.title = theme.micro("Generation", theme.ACCENT)
        self.meta = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.value = TabularText(5, theme.GAP_PX, theme.TEXT_TERTIARY, "headline")
        self.sub = theme.micro("", theme.TEXT_PRIMARY)
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)
        self.value.set("--")

    def set(self, gen: int | None, stage: int | None, best: str, newest: int | None) -> None:
        if gen is None:
            self.value.set("--")
            self.value.colour(theme.TEXT_TERTIARY)
        else:
            self.value.set(str(gen))
            self.value.colour(theme.TEXT_PRIMARY)
        _set(self.meta, "" if stage is None else f"STAGE {stage}")
        parts = [best.upper()] if best else []
        if newest is not None and newest != gen:
            parts.append(f"TRAINED {newest}")
        _set(self.sub, "  ·  ".join(parts))

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.gap_rect(win_w, win_h)
        _place(self.title, r.x + PAD, lay.label_baseline(r))
        _place(self.meta, r.right - PAD, lay.label_baseline(r))
        self.value.place(r.x + PAD - 1.0, r.y + 38)
        _place(self.sub, r.x + PAD, r.y + PAD)

    def draw(self) -> None:
        self.glass.panel(self.rect)
        self.title.draw()
        self.meta.draw()
        self.value.draw()
        self.sub.draw()


class FieldCard:
    """Bottom-left card: cars still running out of the field, the speed and the clock."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.title = theme.micro("On track", theme.ACCENT)
        self.meta = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.value = TabularText(3, theme.SPEED_PX, theme.TEXT_PRIMARY, "headline", "right")
        self.unit = theme.micro("", theme.TEXT_SECONDARY)
        self.clock_label = theme.micro("Episode", theme.TEXT_SECONDARY, anchor_x="right")
        self.clock = theme.make_text(
            "", theme.CLOCK_PX, theme.TEXT_PRIMARY, face="mono", anchor_x="right"
        )
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)

    def set(self, running: int, total: int, clock: str, mode: str) -> None:
        self.value.set(str(running) if total else "--")
        _set(self.unit, f"/ {total} cars" if total else "")
        _set(self.meta, mode.upper())
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


def chart_points(
    gens: np.ndarray, laps: np.ndarray, rect: lay.Rect
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """``(per-generation points, best-so-far line)`` in ``rect``: generations left to right,
    faster laps higher. Empty with no laps; a single lap is a point at mid height."""
    if gens.size == 0:
        return [], []
    g0, g1 = float(gens.min()), float(gens.max())
    lo, hi = float(laps.min()), float(laps.max())
    span_g = max(g1 - g0, 1.0)
    span_t = hi - lo

    def to_xy(g: float, t: float) -> tuple[float, float]:
        x = rect.x + (g - g0) / span_g * rect.w
        y = rect.y + (0.5 if span_t <= 0 else (hi - t) / span_t) * rect.h
        return float(x), float(y)

    points = [to_xy(g, t) for g, t in zip(gens, laps, strict=True)]
    best = np.minimum.accumulate(laps)
    line = [to_xy(g, t) for g, t in zip(gens, best, strict=True)]
    return points, line


class ProgressCard:
    """Bottom-right card: the best lap of each generation (dots) and the best so far (line)."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.rect = lay.Rect(0, 0, lay.COL_W, lay.CARD_H)
        self.title = theme.micro("Best lap", theme.ACCENT)
        self.meta = theme.micro("", theme.TEXT_PRIMARY, anchor_x="right")
        self.empty = theme.micro("No laps yet", theme.TEXT_SECONDARY)
        self.lo = theme.micro("", theme.TEXT_TERTIARY, anchor_x="right")
        self.hi = theme.micro("", theme.TEXT_TERTIARY, anchor_x="right")
        self._gens = np.zeros(0)
        self._laps = np.zeros(0)
        self._points: list[tuple[float, float]] = []
        self._line: list[tuple[float, float]] = []

    def set(self, gens: np.ndarray, laps: np.ndarray, run_best: float) -> None:
        self._gens = np.asarray(gens, dtype=np.float64)
        self._laps = np.asarray(laps, dtype=np.float64)
        _set(self.meta, f"RUN {format_lap(run_best)}" if math.isfinite(run_best) else "")
        if self._laps.size:
            _set(self.hi, format_lap(float(self._laps.min())))
            _set(self.lo, format_lap(float(self._laps.max())))
        self._rebuild()

    def _content(self) -> lay.Rect:
        r = self.rect
        return lay.Rect(r.x + PAD, r.y + PAD, r.w - 2 * PAD - 64.0, r.h - PAD - 40.0)

    def _rebuild(self) -> None:
        self._points, self._line = chart_points(self._gens, self._laps, self._content())

    def layout(self, win_w: float, win_h: float) -> None:
        r = self.rect = lay.card_rect(win_w, win_h)
        _place(self.title, r.x + PAD, lay.label_baseline(r))
        _place(self.meta, r.right - PAD, lay.label_baseline(r))
        c = self._content()
        _place(self.empty, c.x, c.y + 12)
        _place(self.hi, r.right - PAD, c.top - 7)
        _place(self.lo, r.right - PAD, c.y)
        self._rebuild()

    def draw(self) -> None:
        self.glass.panel(self.rect)
        self.title.draw()
        self.meta.draw()
        if not self._points:
            self.empty.draw()
            return
        c = self._content()
        for frac in (0.0, 0.5, 1.0):
            _hairline(c.x, c.right, c.y + frac * c.h, theme.CHART_GRID)
        for x, y in self._points[-400:]:
            arcade.draw_lrbt_rectangle_filled(x - 1.0, x + 1.0, y - 1.0, y + 1.0, theme.CHART_REF)
        if len(self._line) >= 2:
            arcade.draw_line_strip(self._line, theme.CHART_SELF, 1.5)
        self.hi.draw()
        if self.lo.text != self.hi.text:
            self.lo.draw()


class TrainingHud:
    """All HUD widgets of the training view."""

    def __init__(self, glass: GlassRenderer, tower_capacity: int, controls: list[str]):
        self.glass = glass
        self.ticker = Ticker()
        self.headline = Headline()
        self.tower = TimingTower(glass, tower_capacity)
        self.generation = GenerationCard(glass)
        self.field = FieldCard(glass)
        self.progress = ProgressCard(glass)
        self.footer = Footer(glass, controls)
        self._win = (1.0, 1.0)

    def layout(self, win_w: float, win_h: float) -> None:
        self._win = (win_w, win_h)
        for widget in (
            self.ticker,
            self.headline,
            self.tower,
            self.generation,
            self.field,
            self.progress,
            self.footer,
        ):
            widget.layout(win_w, win_h)

    def stage_centre(self) -> tuple[float, float]:
        return lay.stage_centre(*self._win)

    def draw(self, now: float) -> None:
        for draw in (
            lambda: self.ticker.draw(now),
            self.headline.draw,
            self.tower.draw,
            self.generation.draw,
            self.field.draw,
            self.progress.draw,
            self.footer.draw,
        ):
            with theme.text_batch():
                draw()


__all__ = ["FieldCard", "GenerationCard", "ProgressCard", "TrainingHud", "chart_points"]
