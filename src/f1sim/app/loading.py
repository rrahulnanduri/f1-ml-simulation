"""The loading transition into the drive, replay and training screens (D50).

Opening one of them builds about a second of text, track shapes and numba code on the main
thread, which owns the GL context, so the build can't run behind a moving picture. Instead the
transition is timed so the freeze can't be seen:

1. **In:** the menu fades to the loader (:mod:`f1sim.app.ui.loader`) over ``FADE_IN_S``,
   which starts drawing the chosen circuit from its start line.
2. **Draw:** the circuit draws once (``loader.DRAW_S``); the camera's orbit slows evenly to a
   stop as the lap closes (:func:`loader.opening_clock`).
3. **Hold:** the finished circuit holds still (``loader.HOLD_S``). Then the screen is built.
   The frame shown meanwhile is the same still one, so the freeze is invisible.
4. **Out:** the window switches to the new screen with the held loader drawn over it, kept
   opaque for its first ``OPAQUE_FRAMES`` frames (its first update and draw compile and
   upload things too), then faded out over ``FADE_OUT_S``.

The menu drives steps 1-3 (:class:`Opening`); :func:`reveal` does step 4.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from f1sim.app.ui import loader

FADE_IN_S = 0.4
FADE_OUT_S = 0.6
OPAQUE_FRAMES = 2
MAX_STEP_S = 1.0 / 30.0
"""The transition's clock advances at most this much a frame, so a slow frame never makes
the drawing jump."""


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


class Opening:
    """Steps 1-3 for one screen: ``build()`` returns the screen (or raises)."""

    def __init__(self, scene, build: Callable[[], object], label: str = ""):
        self.scene = scene
        self.build = build
        self.label = label
        self.t = 0.0
        self.built = False

    def update(self, delta_time: float) -> None:
        self.t += min(max(delta_time, 0.0), MAX_STEP_S)

    @property
    def alpha(self) -> float:
        return ease(self.t / FADE_IN_S)

    @property
    def ready(self) -> bool:
        """The circuit is drawn and held still: time to build the screen."""
        return self.t >= loader.DRAW_S + loader.HOLD_S

    def draw(self, size_px: tuple[int, int]) -> None:
        """The backdrop and the loader over the whole bound viewport."""
        loader_t, orbit_t = loader.opening_clock(self.t)
        a = self.alpha
        self.scene.backdrop(a)
        self.scene.render(loader_t, size_px, alpha=a, orbit_t=orbit_t)


def reveal(window, screen, opening: Opening) -> None:
    """Show ``screen`` with the held loader over it, fading out (step 4)."""
    draw_screen = screen.on_draw
    state = {"frames": 0, "t0": 0.0, "done": False}

    def on_draw():
        result = draw_screen()
        if state["done"]:
            return result
        state["frames"] += 1
        if state["frames"] <= OPAQUE_FRAMES:
            a = 1.0
            state["t0"] = time.perf_counter()
        else:
            a = 1.0 - ease((time.perf_counter() - state["t0"]) / FADE_OUT_S)
        if a <= 0.0:
            state["done"] = True
            return result
        ctx = window.ctx
        ctx.screen.use()
        old = ctx.screen.viewport
        ctx.screen.viewport = (0, 0, window.width, window.height)  # window units
        loader_t, orbit_t = loader.opening_clock(opening.t)
        opening.scene.backdrop(a)
        opening.scene.render(loader_t, window.get_framebuffer_size(), alpha=a, orbit_t=orbit_t)
        ctx.screen.viewport = old
        return result

    screen.on_draw = on_draw  # before show_view, which binds the view's handlers
    window.show_view(screen)
