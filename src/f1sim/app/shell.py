"""One application window that switches between screens (arcade Views).

The menu, the drive viewer and the replay viewer are :class:`Screen` views shown in one
:class:`AppWindow`: choosing Drive or Replay in the menu switches the window to that screen,
and Esc there comes back to the menu (the same menu view, so its textures aren't reloaded).
``f1sim drive`` and ``f1sim replay`` open the same screens on their own, where Esc quits.

:class:`AppWindow` owns what's per window: the frame pacing (a 1/120 s tick with vsync,
:data:`FRAME_INTERVAL_S`) and, with ``F1SIM_MENU_STATE`` set to a file (the live-reload
script sets it), the window's place and size, saved on every move and restored on start.

:class:`Screen` gives a view the few window calls the viewers were written against
(``ctx``, ``get_pixel_ratio``, ``close``), so their drawing code is unchanged. ``close`` goes
back (``on_back``) rather than closing the window when there is somewhere to go back to.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import arcade

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
            self.set_location(int(state["x"]), int(state["y"]))
        self.menu = None  # the menu view, kept while other screens are shown

    def on_move(self, x: int, y: int) -> None:
        self._persist()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._persist()

    def _persist(self) -> None:
        x, y = self.get_location()
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
