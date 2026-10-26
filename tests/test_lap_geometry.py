import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data import lap_geometry as lg
from f1sim.data.laps import LapXY
from f1sim.data.track_builder import build_track
from f1sim.data.track_limits import LapSample
from f1sim.track.track import Track

SPEED = 50.0


@pytest.fixture(scope="module")
def centre() -> Track:
    x, y = stadium(400.0, 80.0, 3.0)
    w = np.full(x.size, 6.0)
    return Track(build_track(x, y, w, w, track_id="stad", name="Stadium"))


def lap_on(track: Track, offset_m: float, start_s: float = 0.0, rate_hz: float = 4.0) -> LapXY:
    """A constant-speed lap ``offset_m`` left of the centre line, as OpenF1 would sample it."""
    lap_time = track.length / SPEED
    t = np.arange(0.0, lap_time, 1.0 / rate_hz)
    s = (start_s + SPEED * t) % track.length
    p = track.point_at(s)
    h = track.heading_at(s)
    x = p[:, 0] - np.sin(h) * offset_m
    y = p[:, 1] + np.cos(h) * offset_m
    return LapXY("REF", lap_time, t, x, y, (lap_time / 3, 2 * lap_time / 3))


def lapset(track: Track, offsets) -> lg.LapSet:
    ref = lap_on(track, 0.0, start_s=100.0)
    samples = []
    for i, off in enumerate(offsets):
        lap = lap_on(track, off, start_s=100.0)
        x, y = lg._smooth(lap)
        samples.append(LapSample(f"D{i} 2025", x, y))
    return lg.LapSet(ref, 2025, [2025], samples, [])


def test_reference_track_starts_where_the_lap_starts(centre):
    ref = lap_on(centre, 0.0, start_s=100.0)
    track = lg._reference_track(ref)
    assert track.length == pytest.approx(centre.length, rel=0.01)
    start = track.point_at(np.array([0.0]))[0]
    assert np.hypot(start[0] - ref.x_m[0], start[1] - ref.y_m[0]) < 2.0
    # No kink where the lap's ends meet: the stadium's tightest radius is 80 m
    assert np.max(np.abs(track.data.curvature)) < 1.0 / 60.0


def test_envelope_spans_the_laps(centre):
    """Per side, the 90th percentile of the laps' furthest excursions."""
    ls = lapset(centre, [-2.0, -2.0, -2.0, 0.0, 1.5, 1.5, 1.5])
    track = lg._reference_track(ls.reference)
    env_l, env_r = lg.envelope(track, ls.samples)
    np.testing.assert_allclose(np.median(env_l), 1.5, atol=0.3)
    np.testing.assert_allclose(np.median(env_r), 2.0, atol=0.3)


def test_narrow_envelope_is_widened_to_the_minimum(centre):
    ls = lapset(centre, [-0.5, 0.0, 0.5])
    track = lg._reference_track(ls.reference)
    env_l, env_r = lg.envelope(track, ls.samples)
    edge_l, edge_r = lg.road_edges(track, env_l, env_r, min_width=12.0)
    np.testing.assert_allclose(edge_l + edge_r, 12.0, atol=0.3)
    # Each edge keeps the laps and half a car inside it
    assert np.all(edge_l >= env_l + lg.CAR_HALF_WIDTH_M - 1e-6)
    assert np.all(edge_r >= env_r + lg.CAR_HALF_WIDTH_M - 1e-6)


def test_extra_width_goes_to_the_outside_of_bends(centre):
    """The stadium turns left (CCW): in its bends the outside is the right."""
    ls = lapset(centre, [-0.5, 0.0, 0.5])
    track = lg._reference_track(ls.reference)
    env_l, env_r = lg.envelope(track, ls.samples)
    edge_l, edge_r = lg.road_edges(track, env_l, env_r, min_width=12.0)
    k = track.data.curvature
    bend = k > 0.8 / 80.0
    straight = np.abs(k) < 1e-4
    assert np.median(edge_r[bend] - edge_l[bend]) > 3.0
    assert abs(np.median(edge_r[straight] - edge_l[straight])) < 1.0


def test_geometry_builds_a_track_holding_every_lap(centre):
    ls = lapset(centre, [-2.0, -1.0, 0.0, 1.0, 2.0])
    x, y, w_right, w_left = lg.geometry_from_laps(ls, min_width=12.0)
    data = build_track(x, y, w_right, w_left, track_id="stad", name="Stadium")
    track = Track(data)
    for lap in ls.samples:
        s, n, _ = track.project(lap.x, lap.y)
        wl, wr = track.widths_at(s)
        assert np.all(n <= wl - lg.CAR_HALF_WIDTH_M + 0.2)
        assert np.all(-n <= wr - lg.CAR_HALF_WIDTH_M + 0.2)


def test_narrow_circuits_have_their_own_minimum():
    assert lg.min_width_m("monaco") < lg.MIN_WIDTH_M
    assert lg.min_width_m("lusail") == lg.MIN_WIDTH_M
