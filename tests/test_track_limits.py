"""Track limits from real laps: envelope, limits, sim track, and the build pipeline (offline)."""

from dataclasses import replace

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data import track_limits as tl
from f1sim.data.track_builder import build_track
from f1sim.sim.world import World, WorldConfig
from f1sim.track.track import Track

HALF_WIDTH = 6.0


@pytest.fixture(scope="module")
def track() -> Track:
    x, y = stadium(400.0, 60.0, 3.0)
    w = np.full(x.size, HALF_WIDTH)
    return Track(build_track(x, y, w, w.copy(), track_id="stad", name="Stadium"))


def _lap_at_offset(track: Track, n_of_s, step_m: float = 1.0) -> tl.LapSample:
    s = np.arange(0.0, track.length, step_m)
    p = track.point_at(s)
    h = track.heading_at(s)
    n = n_of_s(s)
    return tl.LapSample("TST", p[:, 0] - np.sin(h) * n, p[:, 1] + np.cos(h) * n)


def test_limits_widen_only_where_laps_go_wide(track):
    # Laps hug the centre, except 8 m left (2 m past the white line) between s = 300 and 400
    def offsets(s):
        return np.where((s > 300) & (s < 400), 8.0, 0.0)

    samples = [_lap_at_offset(track, offsets) for _ in range(5)]
    data = tl.with_limits(track.data, samples, (2024,))
    t = Track(data)
    s_mid = np.array([350.0])
    left, right = t.limits_at(s_mid)
    # 8 m centre + 1 m half car + margin, past the 6 m white line
    assert left[0] >= 8.0 + tl.CAR_HALF_WIDTH_M
    assert right[0] == pytest.approx(HALF_WIDTH)
    far = np.array([900.0])
    assert t.limits_at(far)[0][0] == pytest.approx(HALF_WIDTH)
    assert np.all(data.limit_left >= data.w_left) and np.all(data.limit_right >= data.w_right)
    assert tl.limit_self_crossings(data) == (0, 0)
    assert data.limits["laps"] == 5 and data.limits["years"] == [2024]


def test_limits_never_exceed_the_cap(track):
    # 11 m = 5 m past the white line: under the glitch filter (6 m), over the 4 m cap
    samples = [_lap_at_offset(track, lambda s: np.full_like(s, 11.0)) for _ in range(3)]
    data = tl.with_limits(track.data, samples, (2024,))
    assert np.all(data.limit_left <= data.w_left + tl.DEFAULT_PARAMS.max_extra_m + 1e-9)
    assert data.limit_left.max() == pytest.approx(HALF_WIDTH + tl.DEFAULT_PARAMS.max_extra_m)


def test_sim_track_uses_limits_and_world_lets_cars_use_the_kerbs(track):
    def offsets(s):
        return np.full_like(s, 7.5)  # everyone 1.5 m past the left white line

    data = tl.with_limits(track.data, [_lap_at_offset(track, offsets) for _ in range(3)], (2024,))
    limited = Track(data)
    assert limited.has_limits and not limited.sim_track().has_limits
    np.testing.assert_allclose(limited.sim_track().data.w_left, data.limit_left)

    # A car parked 1.5 m past the left white line: off track within the white line,
    # fine within the limits
    def place(world):
        world.reset(start_s=200.0, start_v=0.0)
        h = track.heading_at(200.0)
        world.car.x[:] += -np.sin(h) * 7.5
        world.car.y[:] += np.cos(h) * 7.5
        world.progress.reset(world.car.x, world.car.y)
        return world.step(np.array([[0.0, 0.3]]))

    on_limits = place(World(limited, WorldConfig(n_cars=1)))
    on_white = place(World(limited, WorldConfig(n_cars=1, use_track_limits=False)))
    assert not on_limits.done[0]
    assert on_white.done[0]


def test_tracks_without_limits_are_unchanged(track):
    assert not track.has_limits
    assert track.sim_track() is track
    left, right = track.limits_at(np.array([10.0]))
    assert left[0] == pytest.approx(HALF_WIDTH) and right[0] == pytest.approx(HALF_WIDTH)


def test_limit_arrays_are_validated(track):
    with pytest.raises(ValueError, match="both"):
        replace(track.data, limit_left=track.data.w_left)
    with pytest.raises(ValueError, match="inside the white line"):
        replace(track.data, limit_left=track.data.w_left - 1, limit_right=track.data.w_right)


def test_envelope_needs_laps(track):
    with pytest.raises(ValueError, match="no laps"):
        tl.driving_envelope(track, [])
