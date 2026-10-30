"""The loading transition's timing (f1sim.app.loading) and the loader's fallback asset."""

from __future__ import annotations

import types

import numpy as np
import pyglet
import pytest

from f1sim.app import loading
from f1sim.app.ui import loader


def test_orbit_slows_to_a_stop_and_the_hold_is_still():
    ts = np.linspace(0.0, loader.DRAW_S + loader.HOLD_S + 1.0, 400)
    lt, ot = np.array([loader.opening_clock(t) for t in ts]).T
    assert np.all(np.diff(ot) >= 0)  # the orbit never turns back
    v = np.diff(ot) / np.diff(ts)
    assert v[0] == pytest.approx(1.0, abs=0.01)  # starts at the loop's speed
    held = ts >= loader.DRAW_S + loader.HOLD_S
    assert np.ptp(lt[held]) == 0 and np.ptp(ot[ts >= loader.DRAW_S]) == 0


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_opening_runs_on_the_wall_clock_and_is_ready_after_draw_and_hold():
    clock = FakeClock()
    op = loading.Opening(scene=None, build=lambda: None, clock=clock)
    op.update()  # the first frame starts the clock
    assert op.t == 0.0 and op.alpha == 0.0
    frames = 0
    rng = np.random.default_rng(0)
    while not op.ready:
        clock.now += rng.uniform(0.004, 0.015)  # frames presented unevenly, as measured
        op.update()
        frames += 1
    assert op.t == pytest.approx(clock.now - 100.0)  # follows the wall clock exactly
    assert op.t >= loader.DRAW_S + loader.HOLD_S and op.alpha == 1.0
    clock.now += 5.0
    t = op.t
    op.update()  # one long frame doesn't skip the drawing
    assert op.t - t == pytest.approx(loading.MAX_STEP_S)


class FakeWindow(pyglet.event.EventDispatcher):
    """arcade's view switching: a view's handlers pushed as bound methods, then removed."""

    def __init__(self):
        self.current_view = None
        self.width, self.height = 640, 360
        self.ctx = types.SimpleNamespace(
            screen=types.SimpleNamespace(use=lambda: None, viewport=(0, 0, 640, 360))
        )

    def get_framebuffer_size(self):
        return 640, 360

    def show_view(self, view):
        if self.current_view is not None:
            self.remove_handlers(self.current_view)
        self.current_view = view
        self.push_handlers(on_draw=view.on_draw)


FakeWindow.register_event_type("on_draw")


class FakeView:
    def __init__(self):
        self.draws = 0

    def on_draw(self):
        self.draws += 1


def test_reveal_draws_the_screen_once_a_frame_and_esc_can_leave_it():
    scene = types.SimpleNamespace(backdrop=lambda a: None, render=lambda *a, **k: None)
    window, menu, screen = FakeWindow(), FakeView(), FakeView()
    window.show_view(menu)
    loading.reveal(window, screen, loading.Opening(scene, build=lambda: screen))
    for _ in range(3):
        window.dispatch_event("on_draw")
    assert screen.draws == 3 and menu.draws == 0  # the overlay draws it, not twice
    window.show_view(menu)  # Esc: this raised with the old wrapper, leaving Esc dead
    window.dispatch_event("on_draw")
    window.dispatch_event("on_draw")
    assert menu.draws == 2 and screen.draws == 3
    assert not loading._live  # the overlay took itself off


def test_asset_from_a_built_track():
    a = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    x, y = 300 * np.cos(a), 200 * np.sin(a)
    length = float(np.hypot(*np.diff(np.c_[x, y].T, append=np.c_[x[:1], y[:1]].T)).sum())
    asset = loader.asset_from_track("oval", x, y, length)
    centre = asset["centre_xy"]
    assert np.allclose(centre[0], (300, 0), atol=1e-3)  # from the start line
    assert len(asset["elev_rel_m"]) == int(np.ceil(length))
    assert not asset["grid_rel_m"].any() and asset["osm_xy"].shape == (0, 2)
    lo = asset["grid_origin"]
    ny, nx = asset["grid_rel_m"].shape
    hi = lo + float(asset["grid_step_m"]) * (np.array([nx, ny]) - 1)
    assert np.all(centre.min(0) > lo) and np.all(centre.max(0) < hi)


def test_asset_for_prefers_the_baked_one():
    assert loader.has_asset("silverstone")
    assert "attribution" in loader.asset_for("silverstone")
    with pytest.raises(ValueError):
        loader.asset_for("no_such_track")
