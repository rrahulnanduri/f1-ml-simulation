import numpy as np
import pytest
import shapely

from _geometry_helpers import circle, figure_eight, stadium, teardrop
from f1sim.data.track_builder import TrackBuildError, build_track, wall_self_crossings
from f1sim.track.track import Track, TrackData


def build(x, y, w_right=5.0, w_left=5.0, **kw) -> TrackData:
    n = len(x)
    return build_track(
        x,
        y,
        np.broadcast_to(np.asarray(w_right, float), (n,)),
        np.broadcast_to(np.asarray(w_left, float), (n,)),
        track_id=kw.pop("track_id", "t"),
        name=kw.pop("name", "T"),
        **kw,
    )


def spacing(d: TrackData) -> np.ndarray:
    return np.hypot(np.diff(d.x, append=d.x[0]), np.diff(d.y, append=d.y[0]))


def check_invariants(d: TrackData, resolution: float = 2.0) -> None:
    assert d.s[0] == 0.0
    assert np.all(np.diff(d.s) > 0)
    assert d.length_m > d.s[-1]
    np.testing.assert_allclose(spacing(d), resolution, rtol=0.01)
    np.testing.assert_allclose(np.diff(d.s, append=d.length_m), resolution, rtol=0.01)
    # Heading matches the direction of travel.
    seg = np.arctan2(np.roll(d.y, -1) - np.roll(d.y, 1), np.roll(d.x, -1) - np.roll(d.x, 1))
    dh = (d.heading - seg + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(dh).max() < 0.02


def max_shift(d: TrackData, x, y) -> float:
    """Largest distance from an input point to the resampled polyline, less the chord sagitta
    (the bound is on the spline; the polyline cuts its bends by up to k * ds^2 / 8)."""
    ring = shapely.LineString(np.vstack([np.column_stack([d.x, d.y]), [[d.x[0], d.y[0]]]]))
    sagitta = np.abs(d.curvature).max() * d.resolution_m**2 / 8
    return float(shapely.distance(ring, shapely.points(x, y)).max()) - 1.5 * sagitta


def test_circle_length_curvature_spacing():
    R = 50.0
    d = build(*circle(R, 200))
    check_invariants(d)
    assert d.length_m == pytest.approx(2 * np.pi * R, rel=0.005)
    np.testing.assert_allclose(d.curvature, 1 / R, rtol=0.01)
    np.testing.assert_allclose(np.hypot(d.x, d.y), R, atol=0.01)
    assert d.direction == "counterclockwise"
    assert d.resolution_m == 2.0


def test_resolution_is_configurable():
    d = build(*circle(80.0, 150), resolution_m=5.0)
    check_invariants(d, resolution=5.0)


def test_clockwise_circle_direction_and_curvature_sign():
    d = build(*circle(40.0, 150, ccw=False))
    assert d.direction == "clockwise"
    np.testing.assert_allclose(d.curvature, -1 / 40.0, rtol=0.01)
    # Clockwise: the left side is the outside.
    left = Track(d).left_wall()
    np.testing.assert_allclose(np.hypot(*left.T), 45.0, atol=0.02)


def test_stadium_oval():
    straight, R = 200.0, 40.0
    x, y = stadium(straight, R, 5.0)
    d = build(x, y)
    check_invariants(d)
    assert d.length_m == pytest.approx(2 * straight + 2 * np.pi * R, rel=0.005)
    assert d.direction == "counterclockwise"
    on_straight = np.abs(d.x) < straight / 2 - 15
    in_bend = np.abs(d.x) > straight / 2 + 20
    assert np.abs(d.curvature[on_straight]).max() < 2e-3
    assert np.median(d.curvature[in_bend]) == pytest.approx(1 / R, rel=0.03)
    assert wall_self_crossings(d) == (0, 0)


def test_start_is_first_input_point():
    x, y = stadium(100.0, 30.0, 4.0)
    x, y = np.roll(x, 17), np.roll(y, 17)
    d = build(x, y)
    assert np.hypot(d.x[0] - x[0], d.y[0] - y[0]) <= 0.5


@pytest.mark.parametrize("noise", [0.05, 0.25])
def test_noisy_circle_respects_max_shift(noise):
    rng = np.random.default_rng(42)
    R = 60.0
    x, y = circle(R, 300)
    x = x + rng.normal(0, noise, x.size)
    y = y + rng.normal(0, noise, y.size)
    d = build(x, y)
    if noise < 0.1:
        check_invariants(d)
    else:
        # Noise near the bound: the spline has to bend sharply towards the worst outliers,
        # so chords there are shorter than the 2 m arc steps; the arc steps stay exact.
        np.testing.assert_allclose(np.diff(d.s, append=d.length_m), 2.0, rtol=0.01)
    assert max_shift(d, x, y) <= 0.5 + 1e-3
    # The noise is smoothed out: curvature stays near 1/R, far smoother than interpolation.
    assert np.median(d.curvature) == pytest.approx(1 / R, rel=0.05 if noise < 0.1 else 0.1)
    raw = build(x, y, max_shift_m=0.0)
    assert np.std(d.curvature) < 0.5 * np.std(raw.curvature)
    # A tighter bound than the noise still holds (the spline then follows the noise).
    for max_shift_m in (0.2, 0.0):
        d = build(x, y, max_shift_m=max_shift_m)
        assert max_shift(d, x, y) <= max_shift_m + 1e-3


def test_repeated_end_point_accepted():
    x, y = stadium(120.0, 30.0, 4.0)
    wr = np.linspace(4.0, 6.0, x.size)
    wl = np.full(x.size, 5.0)
    d1 = build(x, y, wr, wl)
    d2 = build(np.append(x, x[0]), np.append(y, y[0]), np.append(wr, wr[0]), np.append(wl, wl[0]))
    assert d1 == d2


def test_widths_interpolated_and_median_filtered():
    x, y = circle(100.0, 300)
    wr = np.full(x.size, 5.0)
    wr[100] = 30.0  # a one-point spike is filtered out
    wl = np.where(np.arange(x.size) < 150, 4.0, 8.0)  # a step survives the filter
    d = build(x, y, wr, wl)
    np.testing.assert_allclose(d.w_right, 5.0)
    assert d.w_left.min() == 4.0 and d.w_left.max() == 8.0
    assert d.w_left[np.argmin(np.abs(d.s - d.length_m * 0.25))] == 4.0
    assert d.w_left[np.argmin(np.abs(d.s - d.length_m * 0.75))] == 8.0
    assert np.mean((d.w_left == 4.0) | (d.w_left == 8.0)) > 0.95


@pytest.mark.parametrize(
    "shape",
    [lambda: stadium(150.0, 8.0, 2.0), lambda: teardrop(8.0, 40.0, 120.0)],
    ids=["narrow_hairpin", "teardrop_hairpin"],
)
def test_tight_hairpin_walls_repaired(shape):
    x, y = shape()
    w = 10.0  # wider than the 8 m hairpin radius
    d = build(x, y, w, w)
    tr = Track(d)
    assert wall_self_crossings(d) == (0, 0)
    for wall in (tr.left_wall(), tr.right_wall()):
        assert shapely.LineString(np.vstack([wall, wall[:1]])).is_simple
    # Counterclockwise: the inside (left) was clipped below 0.95/k, the outside untouched.
    tight = d.curvature > 1 / w
    assert tight.any()
    assert np.all(d.w_left[tight] <= 0.95 / d.curvature[tight] + 1e-9)
    assert d.w_left.min() < w
    np.testing.assert_allclose(d.w_right, w)


def test_figure_eight_rejected():
    x, y = figure_eight()
    with pytest.raises(TrackBuildError, match="crosses itself"):
        build(x, y)


def test_bad_inputs_rejected():
    x, y = circle(50.0, 100)
    with pytest.raises(TrackBuildError):
        build_track(x, y, np.full(99, 5.0), np.full(100, 5.0), track_id="t", name="T")
    with pytest.raises(TrackBuildError, match="non-negative"):
        build(x, y, -1.0, 5.0)
    with pytest.raises(TrackBuildError, match="at least 4"):
        build(x[:3], y[:3])
    with pytest.raises(TrackBuildError, match="non-finite"):
        build(np.where(np.arange(100) == 3, np.nan, x), y)


def test_wall_self_crossings_counts_folded_wall():
    d = build(*circle(50.0, 200))
    assert wall_self_crossings(d) == (0, 0)
    # Hand-made: a left (inner) wall offset beyond the 8 m hairpin radius folds over there.
    good = build(*teardrop(8.0, 40.0, 120.0), 3.0, 3.0)
    bad = TrackData(**{**good.__dict__, "w_left": np.full(good.s.size, 14.0)})
    left, right = wall_self_crossings(bad)
    assert left > 0 and right == 0


def test_source_is_copied():
    src = {"geometry": "synthetic"}
    d = build(*circle(50.0, 100), source=src)
    assert d.source == src and d.source is not src
    assert d.id == "t" and d.name == "T"
