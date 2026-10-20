"""The compiled kernels (ray walk, hinted projection) must match their numpy versions."""

import numpy as np
import pytest

from _geometry_helpers import teardrop
from f1sim import _accel
from f1sim.data.track_builder import build_track
from f1sim.track import grid as grid_mod
from f1sim.track import track as track_mod
from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track


@pytest.mark.skipif(grid_mod._numba_walk is None, reason="numba not installed")
def test_numba_and_numpy_walks_agree(monkeypatch):
    x, y = teardrop(30.0, 120.0, 400.0)
    w = np.full(x.size, 7.0)
    track = Track(build_track(x, y, w, w.copy(), track_id="t", name="t"))
    grid = SegmentGrid.from_track(track, cell_size_m=10.0)
    rng = np.random.default_rng(0)
    s = rng.uniform(0, track.length, 500)
    p = track.point_at(s) + rng.normal(0, 3.0, (500, 2))
    angles = rng.uniform(-np.pi, np.pi, (500, 9))
    angles[:5] = 0.0  # axis-aligned rays
    angles[5:10] = np.pi / 2
    fast = grid.raycast(p[:, 0], p[:, 1], angles, 200.0)
    monkeypatch.setenv(_accel.NO_NUMBA_ENV, "1")
    slow = grid.raycast(p[:, 0], p[:, 1], angles, 200.0)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)


@pytest.mark.skipif(track_mod._numba_project_hinted is None, reason="numba not installed")
def test_numba_and_numpy_hinted_projection_agree(monkeypatch):
    x, y = teardrop(30.0, 120.0, 400.0)
    w = np.full(x.size, 7.0)
    track = Track(build_track(x, y, w, w.copy(), track_id="t", name="t"))
    rng = np.random.default_rng(1)
    s_true = rng.uniform(0, track.length, 2000)
    p = track.point_at(s_true) + rng.normal(0, 4.0, (2000, 2))
    _, _, idx0 = track.project(p[:, 0], p[:, 1])  # global, numpy
    hint = (idx0 + rng.integers(-6, 7, idx0.size)) % len(track.data.x)
    fast = track.project(p[:, 0], p[:, 1], hint=hint)
    monkeypatch.setenv(_accel.NO_NUMBA_ENV, "1")
    slow = track.project(p[:, 0], p[:, 1], hint=hint)
    np.testing.assert_array_equal(fast[2], slow[2])
    np.testing.assert_allclose(fast[0], slow[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fast[1], slow[1], rtol=0, atol=1e-9)
