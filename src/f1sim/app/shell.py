"""One application window that switches between screens (arcade Views).

The menu, the drive viewer and the replay viewer are :class:`Screen` views shown in one
:class:`AppWindow`: choosing Drive or Replay in the menu switches the window to that screen,
and Esc there comes back to the menu (the same menu view, so its textures aren't reloaded).
``f1sim drive`` and ``f1sim replay`` open the same screens on their own, where Esc quits.

:class:`AppWindow` owns what's per window: the frame pacing (a 1/120 s tick with vsync,
:data:`FRAME_INTERVAL_S`), the rendering quality (:mod:`f1sim.app.ui.quality`: it feeds every
frame's duration to the governor) and, with ``F1SIM_MENU_STATE`` set to a file (the live-reload
script sets it), the window's place and size, saved on every move and restored on start.

:class:`Screen` gives a view the few window calls the viewers were written against
(``ctx``, ``get_pixel_ratio``, ``close``), so their drawing code is unchanged. ``close`` goes
back (``on_back``) rather than closing the window when there is somewhere to go back to.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import arcade

from f1sim.app.ui import quality

log = logging.getLogger(__name__)
ENV_SET = f"set by {quality.ENV}"

STATE_ENV = "F1SIM_MENU_STATE"
FRAME_INTERVAL_S = 1.0 / 120.0
"""Window tick interval, s. Where the driver honours vsync, ``flip`` waits for the display and
frames follow its refresh. Where it doesn't (windowed OpenGL on the dual-GPU dev laptop), a
1/60 s timer drifts against a 59-60 Hz display: 10-12% of frames took over 20 ms, a visible
hitch. A 1/120 s tick gave none (measured). Animation advances by the measured frame time,
so the rate doesn't change its speed (D15)."""
DEFAULT_SIZE = (1280, 760)


def load_state() -> dict:
    path = os.environ.get(STATE_ENV)
    if not path:
        return {}
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(state: dict) -> None:
    path = os.environ.get(STATE_ENV)
    if path:
        try:
            Path(path).write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            pass


MIN_VISIBLE_PX = 120
"""A restored window must show at least this much of its title-bar area on some screen."""
PARKED = -10000
"""Windows parks a minimised window at (-32000, -32000); never save positions below this."""


def on_a_screen(x: int, y: int, w: int, screens) -> bool:
    """Whether a window whose top-left corner is at ``(x, y)`` (``w`` wide) would have
    ``MIN_VISIBLE_PX`` of its top strip on one of ``screens`` (``(x, y, width, height)``)."""
    for sx, sy, sw, sh in screens:
        overlap = min(x + w, sx + sw) - max(x, sx)
        if overlap >= min(MIN_VISIBLE_PX, w) and sy <= y < sy + sh - 40:
            return True
    return False


def _screens() -> list[tuple[int, int, int, int]]:
    import pyglet

    try:
        return [(s.x, s.y, s.width, s.height) for s in pyglet.display.get_display().get_screens()]
    except Exception:  # noqa: BLE001 - no screen info: don't restore a position
        return []


class AppWindow(arcade.Window):
    """The one window of the app; see the module docstring."""

    def __init__(self, title: str = "f1sim", size: tuple[int, int] = DEFAULT_SIZE):
        state = load_state()
        super().__init__(
            int(state.get("w", size[0])),
            int(state.get("h", size[1])),
            title,
            resizable=True,
            update_rate=FRAME_INTERVAL_S,
            draw_rate=FRAME_INTERVAL_S,
            vsync=True,
        )
        if "x" in state and "y" in state:
            x, y = int(state["x"]), int(state["y"])
            # Only where it's visible: a screen may be unplugged, or the window was saved
            # minimised (Windows parks those at -32000 and the app opened invisibly)
            if on_a_screen(x, y, self.width, _screens()):
                self.set_location(x, y)
        self.menu = None  # the menu view, kept while other screens are shown
        # Rendering quality: a tier from the GPU, stepped down if frames stay slow
        gov = quality.setup(str(getattr(self.ctx.info, "RENDERER", "")))
        log.info(
            "Rendering quality: %s (%s; %s)",
            gov.tier.name,
            "auto" if gov.auto else ENV_SET,
            getattr(self.ctx.info, "RENDERER", "?"),
        )
        self._last_frame = None

    def flip(self) -> None:
        """Present the frame, and feed its real duration to the quality governor. (Measured
        here, once per drawn frame: a clock callback every loop turn would keep pyglet's loop
        from ever sleeping.)"""
        super().flip()
        now = time.perf_counter()
        if self._last_frame is not None and quality.current.feed(now - self._last_frame):
            log.info("Frames are slow: rendering quality down to %s", quality.tier().name)
        self._last_frame = now

    def on_hide(self) -> None:
        quality.current.paused = True  # minimised: the frame rate says nothing

    def on_show(self) -> None:
        quality.current.paused = False
        self._last_frame = None

    def on_move(self, x: int, y: int) -> None:
        self._persist()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._persist()

    def _persist(self) -> None:
        x, y = self.get_location()
        if x <= PARKED or y <= PARKED or self.width < 50 or self.height < 50:
            return  # minimised: keep the last real place
        save_state({"x": x, "y": y, "w": self.width, "h": self.height})

    def on_close(self) -> None:
        self._persist()
        super().on_close()


class Screen(arcade.View):
    """A view that behaves like the window the viewers were written for."""

    title = "f1sim"

    def __init__(self, on_back: Callable[[], object] | None = None):
        super().__init__()
        self.on_back = on_back

    @property
    def ctx(self):
        return self.window.ctx

    def get_pixel_ratio(self) -> float:
        return float(self.window.get_pixel_ratio())

    def close(self) -> None:
        """Back to where this screen came from, or close the window if it's on its own."""
        if self.on_back is not None:
            self.on_back()
        else:
            self.window.close()

    def on_show_view(self) -> None:
        self.window.set_caption(self.title)
        self.on_resize(self.window.width, self.window.height)  # the window may have changed


def run(make_screen: Callable[[AppWindow], Screen], title: str = "f1sim") -> None:
    """Open the app window on the screen ``make_screen(window)`` builds; returns when the
    window closes."""
    window = AppWindow(title)
    window.show_view(make_screen(window))
    arcade.run()
