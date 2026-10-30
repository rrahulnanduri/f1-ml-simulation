"""The loading transition into the drive, replay and training screens (D50, D51).

Opening one of them builds text, track shapes and numba code on the main thread, which owns
the GL context, so the build can't run behind a moving picture. Instead the transition is
timed so the freeze can't be seen:

1. **In:** the menu fades to the loader (:mod:`f1sim.app.ui.loader`) over ``FADE_IN_S``,
   which starts drawing the chosen circuit from its start line.
2. **Draw:** the circuit draws once (``loader.DRAW_S``); the camera's orbit slows evenly to a
   stop as the lap closes (:func:`loader.opening_clock`).
3. **Hold:** the finished circuit holds still (``loader.HOLD_S``). Then the screen is built.
   The frame shown meanwhile is the same still one, so the freeze is invisible.
4. **Out:** the window switches to the new screen with the held loader drawn over it, kept
   opaque for its first ``OPAQUE_FRAMES`` frames (its first update and draw compile and
   upload things too), then faded out over ``FADE_OUT_S``.

The menu drives steps 1-3 (:class:`Opening`); :func:`reveal` does step 4. What makes it
smooth (D51): the clock is the wall clock read each frame, not the update tick (frames are
presented 4-15 ms apart while the tick says 8.3); the screens' modules are imported ahead, off
the main thread (:func:`warm_up`: scipy and numba took 0.7 s inside the click); and text is
laid out once per label (:func:`f1sim.app.ui.theme.make_text`), which cut the build itself.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import Callable

from f1sim.app.ui import loader

log = logging.getLogger(__name__)

FADE_IN_S = 0.4
FADE_OUT_S = 0.6
OPAQUE_FRAMES = 2
MAX_STEP_S = 1.0 / 30.0
"""The transition's clock advances at most this much a frame, so a slow frame never makes
the drawing jump."""

WARM_MODULES = (
    "f1sim.data.track_pipeline",
    "f1sim.app.drive",
    "f1sim.app.replay",
    "f1sim.app.training",
)
"""Imported in the background when the menu opens, so the first click doesn't import them."""


def _warm_numba() -> None:
    """Compile (load from numba's cache) the kernels a screen's first update calls: on the
    main thread that was a 0.35 s frame as the screen appeared."""
    import numpy as np

    from f1sim.data.track_builder import build_track
    from f1sim.track.track import Track

    a = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    w = np.full(a.size, 5.0)
    track = Track(build_track(100 * np.cos(a), 100 * np.sin(a), w, w, track_id="w", name="w"))
    track.project(np.array([100.0]), np.array([0.0]), hint=np.array([0]))


def warm_up(modules=WARM_MODULES) -> threading.Thread:
    """Import ``modules`` and compile the numba kernels on a background thread (neither
    touches GL)."""

    def run():
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 - it'll be imported (and fail loudly) on use
                log.debug("warm-up import of %s failed", name, exc_info=True)
        try:
            _warm_numba()
        except Exception:  # noqa: BLE001 - it compiles on first use instead
            log.debug("numba warm-up failed", exc_info=True)

    thread = threading.Thread(target=run, name="f1sim-warm-up", daemon=True)
    thread.start()
    return thread


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


class Opening:
    """Steps 1-3 for one screen: ``build()`` returns the screen (or raises). ``clock``
    reads the time (seconds); tests pass their own."""

    def __init__(
        self,
        scene,
        build: Callable[[], object],
        label: str = "",
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.scene = scene
        self.build = build
        self.label = label
        self.clock = clock
        self.t = 0.0
        self.built = False
        self._last: float | None = None

    def update(self) -> None:
        """Advance by the wall time since the last frame (at most ``MAX_STEP_S``)."""
        now = self.clock()
        if self._last is not None:
            self.t += min(max(now - self._last, 0.0), MAX_STEP_S)
        self._last = now

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


class _Reveal:
    """Step 4: draws the shown screen, then the held loader over it, until it has faded.

    It is a handler pushed on the window above the screen's own (and returns "handled", so the
    screen isn't drawn twice), never a wrapper set on the screen: arcade's ``show_view``
    removes a view's handlers as bound methods, and a function set on the instance made it
    raise, so Esc couldn't leave the screen (D51)."""

    def __init__(self, window, screen, opening: Opening):
        self.window, self.screen, self.opening = window, screen, opening
        self.frames = 0
        self.t0 = 0.0

    def on_draw(self):
        window = self.window
        if window.current_view is not self.screen:  # left already: let the new view draw
            self._remove()
            return None
        self.screen.on_draw()
        self.frames += 1
        if self.frames <= OPAQUE_FRAMES:
            a = 1.0
            self.t0 = time.perf_counter()
        else:
            a = 1.0 - ease((time.perf_counter() - self.t0) / FADE_OUT_S)
        if a <= 0.0:
            self._remove()
            return True
        ctx = window.ctx
        ctx.screen.use()
        old = ctx.screen.viewport
        ctx.screen.viewport = (0, 0, window.width, window.height)  # window units
        loader_t, orbit_t = loader.opening_clock(self.opening.t)
        self.opening.scene.backdrop(a)
        self.opening.scene.render(loader_t, window.get_framebuffer_size(), alpha=a, orbit_t=orbit_t)
        ctx.screen.viewport = old
        return True

    def _remove(self) -> None:
        self.window.remove_handlers(on_draw=self.on_draw)
        _live.discard(self)


_live: set[_Reveal] = set()
"""The overlays showing now: pyglet holds handler methods weakly, so these keep them alive."""


def reveal(window, screen, opening: Opening) -> _Reveal:
    """Show ``screen`` with the held loader over it, fading out (step 4)."""
    window.show_view(screen)
    overlay = _Reveal(window, screen, opening)
    _live.add(overlay)
    window.push_handlers(on_draw=overlay.on_draw)
    return overlay
