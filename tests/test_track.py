import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track, TrackData


def circle_data(radius=50.0, m=180, w_left=4.0, w_right=6.0, ccw=True) -> TrackData:
    """An exact circle built by hand, independent of the builder."""
    th = np.arange(m) * (2 * np.pi / m)
    sign = 1.0 if ccw else -1.0
    phi = sign * th
    return TrackData(
        id="circle",
        name="Circle",
        x=radius * np.cos(phi),
        y=radius * np.sin(phi),
        s=radius * th,
        w_left=np.full(m, w_left),
        w_right=np.full(m, w_right),
        heading=np.arctan2(sign * np.cos(phi), -sign * np.sin(phi)),
        curvature=np.full(m, sign / radius),
        length_m=2 * np.pi * radius,
        resolution_m=radius * 2 * np.pi / m,
        direction="counterclockwise" if ccw else "clockwise",
    )


@pytest.fixture(scope="module")
def oval_track() -> Track:
    x, y = stadium(120.0, 30.0, 3.0)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 5.0), track_id="oval", name="Oval")
    return Track(data)


def wrap_diff(a, b, period):
    return (np.asarray(a) - np.asarray(b) + period / 2) % period - period / 2


def left_normal(h):
    return np.stack([-np.sin(h), np.cos(h)], axis=-1)


# ------------------------------------------------------------------------- TrackData


def test_trackdata_arrays_are_readonly_float64():
    d = circle_data()
    assert d.x.dtype == np.float64
    with pytest.raises(ValueError):
        d.x[0] = 1.0


def test_trackdata_validation():
    d = circle_data()
    kw = {f: getattr(d, f) for f in ("id", "name", "x", "y", "s", "w_left", "w_right")}
    kw.update(heading=d.heading, curvature=d.curvature, resolution_m=1.0)
    with pytest.raises(ValueError, match="start at 0"):
        TrackData(**{**kw, "s": d.s + 1.0}, length_m=d.length_m + 1.0)
    with pytest.raises(ValueError, match="closing segment"):
        TrackData(**kw, length_m=float(d.s[-1]))
    with pytest.raises(ValueError, match="points"):
        TrackData(**{**kw, "x": d.x[:-1]}, length_m=d.length_m)
    with pytest.raises(ValueError, match="strictly increasing"):
        TrackData(**{**kw, "s": np.zeros_like(d.s)}, length_m=d.length_m)


def test_trackdata_equality():
    assert circle_data() == circle_data()
    assert circle_data() != circle_data(w_left=3.0)


# ------------------------------------------------------------------------ geometry


def test_walls_follow_left_normal_convention():
    tr = Track(circle_data(radius=50.0, w_left=4.0, w_right=6.0))
    # Counterclockwise: left is the inside.
    np.testing.assert_allclose(np.hypot(*tr.left_wall().T), 46.0)
    np.testing.assert_allclose(np.hypot(*tr.right_wall().T), 56.0)
    tr_cw = Track(circle_data(radius=50.0, w_left=4.0, w_right=6.0, ccw=False))
    np.testing.assert_allclose(np.hypot(*tr_cw.left_wall().T), 54.0)
    assert tr.left_wall().shape == (180, 2)


def test_point_at_wraps_and_shapes():
    tr = Track(circle_data())
    L = tr.length
    assert tr.point_at(10.0).shape == (2,)
    np.testing.assert_allclose(tr.point_at(0.0), [50.0, 0.0])
    np.testing.assert_allclose(tr.point_at(L), tr.point_at(0.0), atol=1e-9)
    np.testing.assert_allclose(tr.point_at(-5.0), tr.point_at(L - 5.0), atol=1e-9)
    np.testing.assert_allclose(tr.point_at(3 * L + 7.0), tr.point_at(7.0), atol=1e-9)
    s = np.linspace(-L, 2 * L, 12).reshape(3, 4)
    assert tr.point_at(s).shape == (3, 4, 2)
    # Points lie on the chords: never outside the circle, at most sagitta inside.
    r = np.hypot(*tr.point_at(np.linspace(0, L, 1000)).T)
    assert r.max() <= 50.0 + 1e-9
    assert r.min() >= 50.0 * np.cos(np.pi / 180) - 1e-9


def test_heading_interpolates_across_pi_wrap():
    tr = Track(circle_data(m=180))
    # On a CCW circle heading = theta + pi/2, which crosses +-pi at theta = pi/2.
    s = np.linspace(0, tr.length, 2001)
    expected = s / 50.0 + np.pi / 2
    err = wrap_diff(tr.heading_at(s), expected, 2 * np.pi)
    assert np.abs(err).max() < 1e-9
    h = tr.heading_at(s)
    assert h.min() >= -np.pi and h.max() < np.pi
    assert np.ndim(tr.heading_at(1.0)) == 0


def test_curvature_and_widths_interpolate_linearly():
    d = circle_data()
    w_left = np.arange(d.s.size, dtype=float)
    d2 = TrackData(**{**d.__dict__, "w_left": w_left})
    tr = Track(d2)
    s_mid = 0.5 * (d.s[3] + d.s[4])
    wl, wr = tr.widths_at(s_mid)
    assert wl == pytest.approx(3.5)
    assert wr == pytest.approx(6.0)
    # Last segment interpolates back to point 0.
    wl_last, _ = tr.widths_at(0.5 * (d.s[-1] + tr.length))
    assert wl_last == pytest.approx(0.5 * w_left[-1])
    np.testing.assert_allclose(tr.curvature_at(np.array([1.0, -1.0, 1e4])), 1 / 50.0)


# ---------------------------------------------------------------------- projection


def _points_at(tr: Track, s, n):
    return tr.point_at(s) + n[..., None] * left_normal(tr.heading_at(s))


def test_project_round_trip_global(oval_track):
    tr = oval_track
    rng = np.random.default_rng(1)
    s = rng.uniform(0, tr.length, 400)
    s[:4] = [0.0, 0.05, tr.length - 0.05, tr.length - 1e-7]  # around start/finish
    wl, wr = tr.widths_at(s)
    n = rng.uniform(-0.9, 0.9, s.size) * np.where(rng.random(s.size) < 0.5, wl, wr)
    p = _points_at(tr, s, n)
    s2, n2, idx = tr.project(p[:, 0], p[:, 1])
    assert np.abs(wrap_diff(s2, s, tr.length)).max() < 1e-2
    assert np.abs(n2 - n).max() < 1e-2
    assert s2.min() >= 0.0 and s2.max() < tr.length
    assert idx.dtype.kind == "i" and idx.min() >= 0 and idx.max() < tr.data.n_points


def test_project_with_hint_matches_global_and_wraps(oval_track):
    tr = oval_track
    M = tr.data.n_points
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.uniform(0, tr.length, 200), [0.1, tr.length - 0.1]])
    n = rng.uniform(-4.0, 4.0, s.size)
    p = _points_at(tr, s, n)
    s_g, n_g, idx_g = tr.project(p[:, 0], p[:, 1])
    # Hints a few segments off, including hints on the other side of start/finish.
    hint = (idx_g + rng.integers(-6, 7, s.size)) % M
    hint[-2], hint[-1] = M - 3, 2
    s_h, n_h, idx_h = tr.project(p[:, 0], p[:, 1], hint=hint)
    np.testing.assert_allclose(s_h, s_g, atol=1e-9)
    np.testing.assert_allclose(n_h, n_g, atol=1e-9)
    np.testing.assert_array_equal(idx_h, idx_g)
    assert abs(wrap_diff(s_h[-2], 0.1, tr.length)) < 1e-2
    assert abs(wrap_diff(s_h[-1], tr.length - 0.1, tr.length)) < 1e-2


def test_project_hint_only_searches_window(oval_track):
    tr = oval_track
    M = tr.data.n_points
    p = tr.point_at(tr.data.s[M // 2])
    _, _, idx = tr.project(p[0], p[1], hint=0, window=10)
    # The true segment is half a lap away; the windowed search must stay near the hint.
    assert min(idx, M - idx) <= 10 + 4


def test_project_vectorised_shapes(oval_track):
    tr = oval_track
    s = np.linspace(0, tr.length, 24, endpoint=False).reshape(4, 6)
    p = tr.point_at(s)
    s2, n2, idx = tr.project(p[..., 0], p[..., 1])
    assert s2.shape == n2.shape == idx.shape == (4, 6)
    np.testing.assert_allclose(s2, s, atol=1e-6)
    np.testing.assert_allclose(n2, 0.0, atol=1e-9)
    s3, _, idx3 = tr.project(p[..., 0], p[..., 1], hint=idx)
    np.testing.assert_allclose(s3, s2)
    s_scalar, n_scalar, i_scalar = tr.project(float(p[0, 1, 0]), float(p[0, 1, 1]))
    assert np.ndim(s_scalar) == 0 and s_scalar == pytest.approx(s[0, 1], abs=1e-6)


def test_project_wall_points_have_n_equal_to_width(oval_track):
    tr = oval_track
    d = tr.data
    s, n, idx = tr.project(*tr.left_wall().T)
    np.testing.assert_allclose(n, d.w_left, atol=1e-9)
    assert np.abs(wrap_diff(s, d.s, tr.length)).max() < 1e-9
    _, n_r, _ = tr.project(*tr.right_wall().T)
    np.testing.assert_allclose(n_r, -d.w_right, atol=1e-9)
    # Midway along a wall segment, n is the interpolated width.
    mid = 0.5 * (tr.left_wall() + np.roll(tr.left_wall(), -1, axis=0))
    s_m, n_m, _ = tr.project(*mid.T)
    np.testing.assert_allclose(n_m, tr.widths_at(s_m)[0], atol=1e-6)


def test_project_circle_exact_n():
    tr = Track(circle_data(radius=50.0, m=360))
    ang = np.linspace(0, 2 * np.pi, 50, endpoint=False) + 0.003
    for r in (44.0, 50.0, 57.0):
        s, n, _ = tr.project(r * np.cos(ang), r * np.sin(ang))
        np.testing.assert_allclose(n, 50.0 - r, atol=5e-3)
        # s is linear along each chord, the angle isn't: tiny (sub-0.1 mm) mismatch.
        np.testing.assert_allclose(wrap_diff(s, 50.0 * ang, tr.length), 0.0, atol=1e-4)


def test_project_global_many_segments_bounded():
    m = 3000
    th = np.arange(m) * (2 * np.pi / m)
    R = 1000.0
    d = TrackData(
        id="big",
        name="Big",
        x=R * np.cos(th),
        y=R * np.sin(th),
        s=R * th,
        w_left=np.full(m, 6.0),
        w_right=np.full(m, 6.0),
        heading=th + np.pi / 2,
        curvature=np.full(m, 1 / R),
        length_m=2 * np.pi * R,
        resolution_m=2 * np.pi * R / m,
    )
    tr = Track(d)
    rng = np.random.default_rng(3)
    ang = rng.uniform(0, 2 * np.pi, 256)
    s, n, idx = tr.project(1003.0 * np.cos(ang), 1003.0 * np.sin(ang))
    np.testing.assert_allclose(n, -3.0, atol=1e-3)
    np.testing.assert_allclose(s, R * ang, atol=1e-6)
