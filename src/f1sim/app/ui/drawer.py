"""A side drawer that slides in from the right edge: the training view's "How it works".

Closed, only its handle shows: a small glass tab, ``? HOW IT WORKS``, on the window's right
edge, half-way up. A click on it (or ``H``) slides the drawer out over the track, easing
over :data:`ANIM_S`; the same again, or Esc, slides it back. Inside: a row of tabs (one per
page of :func:`f1sim.app.explain.pages`), the page title, and the page's blocks, which
scroll with the mouse wheel when they don't fit. Paragraphs are wrapped here
(:func:`wrap_words`), not by pyglet: at 2x, pyglet's word wrap split words whose glyphs sat
in different atlas textures.

The geometry (:func:`drawer_rect`, :func:`handle_rect`, :func:`bar_fraction`) and the
slide (:class:`Slide`) are pure functions and classes; :class:`Drawer` creates its
``arcade.Text`` objects when a page's content changes, not every frame.
"""

from __future__ import annotations

import math

import arcade

from f1sim.app import explain as ex
from f1sim.app.ui import layout as lay
from f1sim.app.ui import theme
from f1sim.app.ui.glass import GlassRenderer
from f1sim.app.ui.hud import _hairline, _place, _set
from f1sim.app.ui.paging import ease_out_cubic

ANIM_S = 0.32
"""Slide duration, s."""
WIDTH = 400.0
"""Drawer width; narrower on a small window (:func:`drawer_rect`)."""
HANDLE_W, HANDLE_H = 132.0, 30.0
PAD = 20.0
BODY_PX = 13
TITLE_PX = 20
TABS_H = 40.0
"""Height of the tab row at the top."""
GAP = 10.0
"""Space between blocks."""
BAR_H = 4.0
SCROLL_STEP = 40.0
"""Pixels per wheel notch."""


# ------------------------------------------------------------------------------ geometry


def drawer_rect(win_w: float, win_h: float, frac: float) -> lay.Rect:
    """The drawer ``frac`` (0 closed .. 1 open) of the way out: its right edge
    ``lay.MARGIN`` from the window's when open and just past it when closed, from the top
    margin down to the footer."""
    w = min(WIDTH, max(win_w * 0.45, 240.0))
    open_x = win_w - lay.MARGIN - w
    x = open_x + (1.0 - frac) * (w + lay.MARGIN)
    return lay.Rect(x, lay.BOTTOM_Y, w, max(win_h - lay.MARGIN - lay.BOTTOM_Y, 0.0))


def handle_rect(drawer: lay.Rect, win_w: float) -> lay.Rect:
    """The handle tab, left of the drawer's left edge and half-way up it (so it peeks in
    from the right edge when the drawer is closed)."""
    x = min(drawer.x, win_w) - HANDLE_W - lay.GUTTER
    return lay.Rect(x, drawer.y + 0.5 * (drawer.h - HANDLE_H), HANDLE_W, HANDLE_H)


def bar_fraction(lap_s: float, laps: list[float]) -> float:
    """How long a lap's bar is, 0.3..1: the slowest lap of ``laps`` fills it, and each
    second faster takes off a share of the spread (so small gaps stay visible)."""
    lo, hi = min(laps), max(laps)
    if hi - lo <= 1e-9:
        return 1.0
    return 0.3 + 0.7 * (lap_s - lo) / (hi - lo)


def wrap_words(text: str, width: float, measure) -> list[str]:
    """``text`` broken into lines no wider than ``width`` (``measure(s)`` = the width of
    ``s``), at spaces; a word wider than ``width`` gets a line of its own."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}" if line else word
        if line and measure(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _measure(text_obj):
    """A width function for strings in ``text_obj``'s font, in logical pixels."""
    label = text_obj.label
    font = label.document.get_font(0, dpi=label.dpi)  # without dpi: the 96-dpi font
    ratio = label.dpi / 96.0

    def measure(s: str) -> float:
        glyphs, offsets = font.get_glyphs(s)
        return sum(g.advance + o.x_advance for g, o in zip(glyphs, offsets, strict=True)) / ratio

    return measure


class Slide:
    """Open/closed with the eased slide between them (times passed in, seconds)."""

    def __init__(self) -> None:
        self.open = False
        self._from = 0.0
        self._t0 = -math.inf

    def frac(self, now: float) -> float:
        target = 1.0 if self.open else 0.0
        u = (now - self._t0) / ANIM_S
        if u >= 1.0:
            return target
        return self._from + (target - self._from) * ease_out_cubic(u)

    def set(self, is_open: bool, now: float) -> None:
        if is_open != self.open:
            self._from = self.frac(now)
            self._t0 = now
            self.open = is_open


# -------------------------------------------------------------------------------- drawer


class Drawer:
    """The "How it works" drawer; see the module docstring."""

    def __init__(self, glass: GlassRenderer):
        self.glass = glass
        self.slide = Slide()
        self.page = 0
        self.scroll = 0.0
        self._opened_once = False
        self._win = (1.0, 1.0)
        self.rect = drawer_rect(1.0, 1.0, 0.0)
        self.handle_label = theme.micro("?  How it works", theme.TEXT_PRIMARY, anchor_x="center")
        self.tabs = [theme.micro(t, theme.TEXT_SECONDARY) for t in ex.PAGE_TABS]
        self.close = theme.micro("Esc ×", theme.TEXT_TERTIARY, anchor_x="right")
        self.title = theme.make_text("", TITLE_PX, theme.TEXT_PRIMARY, face="display")
        self._pages: list[ex.Page] = []
        self._built: dict[int, tuple[ex.Page, float, list]] = {}
        self._facts: ex.Facts | None = None

    # ------------------------------------------------------------------- state

    @property
    def is_open(self) -> bool:
        return self.slide.open

    def toggle(self, now: float) -> None:
        self.set_open(not self.slide.open, now)

    def set_open(self, is_open: bool, now: float) -> None:
        if is_open and not self._opened_once and self._facts is not None:
            self.page = ex.opening_page(self._facts)  # the first time: what fits the run
            self._opened_once = True
        self.slide.set(is_open, now)

    def next_page(self) -> None:
        self.go_to((self.page + 1) % len(ex.PAGE_TABS))

    def go_to(self, page: int) -> None:
        if page != self.page:
            self.page = page
            self.scroll = 0.0

    def set_facts(self, facts: ex.Facts) -> None:
        """New live facts; pages whose content changed are rebuilt when next drawn."""
        if facts != self._facts:
            self._facts = facts
            self._pages = ex.pages(facts)

    # ------------------------------------------------------------------- input

    def _tab_hit(self, x: float, y: float) -> int | None:
        r = self.rect
        top = r.top - TABS_H
        if not (top <= y <= r.top and r.x <= x <= r.right):
            return None
        for i, t in enumerate(self.tabs):
            if t.x - 6 <= x <= t.x + t.content_width + 6:
                return i
        return None

    def on_press(self, x: float, y: float, now: float) -> bool:
        """Whether the drawer took the click (handle, a tab, or anywhere on the open
        drawer)."""
        win_w, _ = self._win
        if handle_rect(self.rect, win_w).contains(x, y):
            self.toggle(now)
            return True
        if not self.slide.open or not self.rect.contains(x, y):
            return False
        tab = self._tab_hit(x, y)
        if tab is not None:
            self.go_to(tab)
        elif x >= self.close.x - self.close.content_width - 6 and y >= self.rect.top - TABS_H:
            self.set_open(False, now)
        return True

    def on_scroll(self, x: float, y: float, scroll_y: float) -> bool:
        """Whether the wheel was over the open drawer (it scrolls the page)."""
        if not self.slide.open or not self.rect.contains(x, y):
            return False
        self.scroll = max(self.scroll - scroll_y * SCROLL_STEP, 0.0)  # clamped when drawn
        return True

    # ------------------------------------------------------------------ layout

    def layout(self, win_w: float, win_h: float) -> None:
        if (win_w, win_h) != self._win:
            self._built.clear()  # the width may have changed: re-wrap
        self._win = (win_w, win_h)

    def _content_w(self) -> float:
        return drawer_rect(*self._win, 1.0).w - 2 * PAD

    def _build(self, page: ex.Page) -> tuple[float, list]:
        """Text objects and their offsets below the page title: ``(height, items)`` with
        items ``(kind, offset_from_top, payload)``."""
        w = self._content_w()
        items: list = []
        y = 0.0
        laps = [b.lap_s for b in page.blocks if isinstance(b, ex.Bar)]
        for b in page.blocks:
            if isinstance(b, ex.Heading):
                y += 6.0
                t = theme.micro(b.text, theme.ACCENT)
                y += 10.0
                items.append(("text", y, t))
                y += GAP
            elif isinstance(b, ex.Para):
                # Wrapped here, with explicit line breaks: pyglet's own word wrap can split a
                # word whose glyphs sit in different atlas textures (seen at 2x)
                colour = theme.TEXT_PRIMARY if b.live else theme.TEXT_SECONDARY
                wrap_w = w - (10 if b.live else 0)
                t = theme.make_text("", BODY_PX, colour, multiline=True, width=wrap_w + 200)
                t.text = "\n".join(wrap_words(b.text, wrap_w, _measure(t)))
                t.anchor_y = "top"
                items.append(("para", y, (t, b.live)))
                y += t.content_height + GAP
            elif isinstance(b, ex.Row):
                name = theme.make_text(b.name, 12, theme.TEXT_SECONDARY)
                value = theme.make_text(b.value, 12, theme.TEXT_PRIMARY, anchor_x="right")
                y += 13.0
                items.append(("row", y, (name, value)))
                y += 9.0
            elif isinstance(b, ex.Bar):
                label = theme.micro(
                    b.label, theme.TEXT_PRIMARY if b.highlight else theme.TEXT_SECONDARY
                )
                value = theme.make_text(
                    ex.fmt_lap(b.lap_s), 12, theme.TEXT_PRIMARY, face="mono", anchor_x="right"
                )
                y += 11.0
                frac = bar_fraction(b.lap_s, laps)
                items.append(("bar", y, (label, value, frac, b.highlight)))
                y += 6.0 + BAR_H + GAP
        return y, items

    def _built_page(self) -> tuple[float, list]:
        page = self._pages[self.page]
        cached = self._built.get(self.page)
        if cached is None or cached[0] != page:
            h, items = self._build(page)
            self._built[self.page] = (page, h, items)
            _set(self.title, page.title.upper())
            return h, items
        return cached[1], cached[2]

    # -------------------------------------------------------------------- draw

    def draw(self, now: float) -> None:
        win_w, win_h = self._win
        frac = self.slide.frac(now)
        r = self.rect = drawer_rect(win_w, win_h, frac)
        h = handle_rect(r, win_w)
        self.glass.panel(h, radius=HANDLE_H / 2)
        _place(self.handle_label, h.x + h.w / 2, h.y + 11)
        with theme.text_batch():
            self.handle_label.draw()
        if frac <= 0.0 or not self._pages:
            return
        self.glass.panel(r)
        with theme.text_batch():
            self._draw_header(r)
        height, items = self._built_page()
        _set(self.title, self._pages[self.page].title.upper())
        body_top = r.top - TABS_H - 16.0 - TITLE_PX - 14.0
        visible = body_top - (r.y + PAD)
        self.scroll = min(self.scroll, max(height - visible, 0.0))
        ctx = self.glass.ctx
        ctx.scissor = (int(r.x + 1), int(r.y + 1), int(r.w - 2), int(body_top - r.y))
        try:
            with theme.text_batch():
                self._draw_items(r, items, body_top + self.scroll)
        finally:
            ctx.scissor = None
        if height > visible:  # a thin scroll indicator on the right edge
            span = visible * visible / height
            top = body_top - (self.scroll / (height - visible)) * (visible - span)
            arcade.draw_lrbt_rectangle_filled(
                r.right - 5, r.right - 3, top - span, top, theme.TEXT_TERTIARY
            )

    def _draw_header(self, r: lay.Rect) -> None:
        x = r.x + PAD
        base = r.top - 24.0
        for i, t in enumerate(self.tabs):
            colour = theme.ACCENT if i == self.page else theme.TEXT_SECONDARY
            if tuple(t.color) != tuple(colour):
                t.color = colour
            _place(t, x, base)
            t.draw()
            if i == self.page:
                _hairline(x, x + t.content_width, base - 6, theme.ACCENT)
            x += t.content_width + 18.0
        _place(self.close, r.right - PAD, base)
        self.close.draw()
        _hairline(r.x + 1, r.right - 1, r.top - TABS_H, theme.SEPARATOR)
        _place(self.title, r.x + PAD, r.top - TABS_H - 16.0 - TITLE_PX + 4)
        self.title.draw()

    def _draw_items(self, r: lay.Rect, items: list, top: float) -> None:
        x0, x1 = r.x + PAD, r.right - PAD
        for kind, dy, payload in items:
            y = top - dy
            if kind == "text":
                _place(payload, x0, y)
                payload.draw()
            elif kind == "para":
                t, live = payload
                if live:
                    arcade.draw_lrbt_rectangle_filled(
                        x0, x0 + 2, y - t.content_height + 3, y - 1, theme.ACCENT
                    )
                    _place(t, x0 + 10, y)
                else:
                    _place(t, x0, y)
                t.draw()
            elif kind == "row":
                name, value = payload
                _place(name, x0, y)
                _place(value, x1, y)
                name.draw()
                value.draw()
                _hairline(x0, x1, y - 7, theme.SEPARATOR)
            elif kind == "bar":
                label, value, frac, highlight = payload
                _place(label, x0, y)
                _place(value, x1, y)
                label.draw()
                value.draw()
                by = y - 8 - BAR_H
                arcade.draw_lrbt_rectangle_filled(x0, x1, by, by + BAR_H, theme.SEPARATOR)
                colour = theme.ACCENT if highlight else theme.TEXT_SECONDARY
                arcade.draw_lrbt_rectangle_filled(x0, x0 + frac * (x1 - x0), by, by + BAR_H, colour)
