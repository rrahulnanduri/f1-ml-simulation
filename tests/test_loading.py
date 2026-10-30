"""The loading transition's timing (f1sim.app.loading) and the loader's fallback asset."""

from __future__ import annotations

import numpy as np
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


def test_opening_is_ready_after_draw_and_hold_whatever_the_frame_rate():
    op = loading.Opening(scene=None, build=lambda: None)
    op.update(0.0)
    assert op.alpha == 0.0
    frames = 0
    while not op.ready:
        op.update(1 / 60)
        frames += 1
    assert abs(frames - (loader.DRAW_S + loader.HOLD_S) * 60) <= 1 and op.alpha == 1.0
    slow = loading.Opening(scene=None, build=lambda: None)
    slow.update(5.0)  # one long frame doesn't skip the drawing
    assert slow.t == pytest.approx(loading.MAX_STEP_S)


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
