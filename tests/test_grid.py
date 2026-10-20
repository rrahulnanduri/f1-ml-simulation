import numpy as np
import pytest

from _geometry_helpers import circle, teardrop
from f1sim.data.track_builder import build_track
from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track

R_CIRCLE, W_LEFT, W_RIGHT = 150.0, 5.0, 6.0


def brute_force(segments, ox, oy, angles, max_dist):
    """Reference: every ray against every segment (no grid), end points inclusive."""
    seg = np.asarray(segments, dtype=np.float64)
    px, py = seg[:, 0, 0], seg[:, 0, 1]
    ex, ey = seg[:, 1, 0] - px, seg[:, 1, 1] - py
    ang = np.asarray(angles, dtype=np.float64)
    dx = np.cos(ang)[..., None]
    dy = np.sin(ang)[..., None]
    wx = px - np.asarray(ox, dtype=np.float64)[:, None, None]
    wy = py - np.asarray(oy, dtype=np.float64)[:, None, None]
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / den
        u = (wx * dy - wy * dx) / den
    hit = (den != 0.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9) & (t >= -1e-9)
    t = np.where(hit, np.maximum(t, 0.0), np.inf)
    return np.minimum(t.min(axis=-1), max_dist)


def loop_segments(points):
    pts = np.asarray(points, dtype=np.float64)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


@pytest.fixture(scope="module")
def circle_track() -> Track:
    x, y = circle(R_CIRCLE, n=470)
    m = x.size
    data = build_track(
        x, y, np.full(m, W_RIGHT), np.full(m, W_LEFT), track_id="circle", name="Circle"
    )
    return Track(data)


@pytest.fixture(scope="module")
def teardrop_track() -> Track:
    x, y = teardrop(25.0, 60.0, 180.0)
    m = x.size
    data = build_track(x, y, np.full(m, 6.0), np.full(m, 5.0), track_id="tear", name="Tear")
    return Track(data)


def random_rays(rng, track, n, r, spread=0.95):
    s = rng.uniform(0.0, track.length, n)
    wl, wr = track.widths_at(s)
    side = rng.uniform(-spread, spread, n)
    off = np.where(side >= 0, side * wl, side * wr)
    h = track.heading_at(s)
    p = track.point_at(s)
    x = p[:, 0] - np.sin(h) * off
    y = p[:, 1] + np.cos(h) * off
    return x, y, rng.uniform(-np.pi, np.pi, (n, r))


# ------------------------------------------------------------------------- construction


def test_from_track_uses_both_walls_closed(circle_track):
    grid = SegmentGrid.from_track(circle_track)
    m = circle_track.data.n_points
    assert grid.segments.shape == (2 * m, 2, 2)
    left, right = circle_track.left_wall(), circle_track.right_wall()
    np.testing.assert_array_equal(grid.segments[m - 1], [left[-1], left[0]])
    np.testing.assert_array_equal(grid.segments[2 * m - 1], [right[-1], right[0]])
    np.testing.assert_array_equal(grid.segments[m, 0], right[0])


def test_every_segment_is_registered_in_every_cell_it_crosses():
    rng = np.random.default_rng(3)
    seg = rng.uniform(0.0, 100.0, (300, 2, 2))
    grid = SegmentGrid(seg, cell_size_m=7.0)
    # Dense samples along each segment must all fall in cells that list the segment.
    u = np.linspace(0.0, 1.0, 401)
    for k in range(seg.shape[0]):
        pts = seg[k, 0] + u[:, None] * (seg[k, 1] - seg[k, 0])
        ix = np.floor((pts[:, 0] - grid.x0) / grid.cell_size).astype(int)
        iy = np.floor((pts[:, 1] - grid.y0) / grid.cell_size).astype(int)
        for cell in np.unique(ix * grid.ny + iy):
            start = grid._cell_start[cell]
            assert k in grid._cell_segs[start : start + grid._cell_count[cell]]


@pytest.mark.parametrize(
    "segments",
    [np.zeros((0, 2, 2)), np.zeros((3, 2)), np.full((2, 2, 2), np.nan)],
)
def test_bad_segments_rejected(segments):
    with pytest.raises(ValueError):
        SegmentGrid(segments)


def test_bad_cell_size_rejected():
    with pytest.raises(ValueError):
        SegmentGrid(np.array([[[0.0, 0.0], [1.0, 0.0]]]), cell_size_m=0.0)


def test_bad_raycast_shapes_rejected():
    grid = SegmentGrid(np.array([[[0.0, 0.0], [1.0, 0.0]]]))
    with pytest.raises(ValueError):
        grid.raycast(np.zeros(2), np.zeros(3), np.zeros((2, 1)), 10.0)
    with pytest.raises(ValueError):
        grid.raycast(np.zeros(2), np.zeros(2), np.zeros((3, 1)), 10.0)
    with pytest.raises(ValueError):
        grid.raycast(np.zeros(2), np.zeros(2), np.zeros(2), 10.0)
    with pytest.raises(ValueError):
        grid.raycast(np.zeros(2), np.zeros(2), np.zeros((2, 1)), 0.0)


# ----------------------------------------------------------------------------- accuracy


def test_circle_matches_brute_force_and_analytic(circle_track):
    grid = SegmentGrid.from_track(circle_track)
    rng = np.random.default_rng(0)
    x, y, ang = random_rays(rng, circle_track, 300, 12)
    got = grid.raycast(x, y, ang, 400.0)
    np.testing.assert_allclose(got, brute_force(grid.segments, x, y, ang, 400.0), atol=1e-9)

    # Analytic: first intersection with the inner (left) and outer (right) wall circles.
    r_in, r_out = R_CIRCLE - W_LEFT, R_CIRCLE + W_RIGHT
    dx, dy = np.cos(ang), np.sin(ang)
    b = x[:, None] * dx + y[:, None] * dy  # o.d
    c0 = x**2 + y**2
    disc_in = b**2 - (c0[:, None] - r_in**2)
    with np.errstate(invalid="ignore"):
        t_in = -b - np.sqrt(disc_in)
    t_out = -b + np.sqrt(b**2 - (c0[:, None] - r_out**2))
    hits_inner = (disc_in > 0) & (t_in > 0)
    expected = np.where(hits_inner, t_in, t_out)
    # The walls are polylines (chords ~2 m, sagitta ~3.5 mm): skip near-tangent rays, where
    # the chord-vs-arc gap along the ray blows up.
    hx, hy = x[:, None] + expected * dx, y[:, None] + expected * dy
    cos_inc = np.abs(hx * dx + hy * dy) / np.hypot(hx, hy)
    ok = cos_inc > 0.5
    assert ok.mean() > 0.6
    np.testing.assert_allclose(got[ok], expected[ok], atol=0.01)


def test_teardrop_matches_brute_force(teardrop_track):
    grid = SegmentGrid.from_track(teardrop_track)
    rng = np.random.default_rng(1)
    x, y, ang = random_rays(rng, teardrop_track, 400, 9)
    for max_dist in (200.0, 12.0, 1000.0):
        got = grid.raycast(x, y, ang, max_dist)
        ref = brute_force(grid.segments, x, y, ang, max_dist)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize("cell", [0.7, 3.0, 25.0, 400.0])
def test_any_cell_size_and_any_origin_matches_brute_force(teardrop_track, cell):
    grid = SegmentGrid.from_track(teardrop_track, cell_size_m=cell)
    rng = np.random.default_rng(2)
    seg = grid.segments
    lo, hi = seg.reshape(-1, 2).min(axis=0) - 50.0, seg.reshape(-1, 2).max(axis=0) + 50.0
    x = rng.uniform(lo[0], hi[0], 250)  # inside and outside the walls and the grid
    y = rng.uniform(lo[1], hi[1], 250)
    ang = rng.uniform(-np.pi, np.pi, (250, 5))
    got = grid.raycast(x, y, ang, 300.0)
    np.testing.assert_allclose(got, brute_force(seg, x, y, ang, 300.0), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------- edge cases

SQUARE = loop_segments([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])


def test_no_hit_returns_max_dist():
    grid = SegmentGrid(SQUARE, cell_size_m=4.0)
    # Pointing away from the square from outside it, and too short to reach a wall.
    got = grid.raycast(
        np.array([-5.0, 10.0]), np.array([10.0, 10.0]), np.array([[np.pi], [0.3]]), 7.5
    )
    np.testing.assert_array_equal(got, [[7.5], [7.5]])
    far = grid.raycast(np.array([1e4]), np.array([1e4]), np.array([[0.0, 1.0]]), 200.0)
    np.testing.assert_array_equal(far, [[200.0, 200.0]])


def test_axis_aligned_rays():
    grid = SegmentGrid(SQUARE, cell_size_m=4.0)
    ang = np.array([[0.0, np.pi / 2, np.pi, -np.pi / 2, -np.pi]])
    got = grid.raycast(np.array([3.0]), np.array([8.0]), ang, 100.0)
    np.testing.assert_allclose(got, [[17.0, 12.0, 3.0, 8.0, 3.0]], atol=1e-12)


def test_rays_through_vertices():
    grid = SegmentGrid(SQUARE, cell_size_m=4.0)
    corners = np.array([[np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]])
    got = grid.raycast(np.array([10.0]), np.array([10.0]), corners, 100.0)
    np.testing.assert_allclose(got, np.full((1, 4), 10.0 * np.sqrt(2.0)), atol=1e-9)

    # A ray that only touches the tip of a "V" and one through a polyline joint.
    v = np.array([[[4.0, 1.0], [5.0, 0.0]], [[5.0, 0.0], [4.0, -1.0]]])
    joint = np.array([[[8.0, -1.0], [8.0, 0.0]], [[8.0, 0.0], [8.0, 1.0]]])
    for segs, want in ((v, 5.0), (joint, 8.0)):
        g = SegmentGrid(segs, cell_size_m=1.0)
        np.testing.assert_allclose(g.raycast([0.0], [0.0], [[0.0]], 50.0), [[want]], atol=1e-12)


def test_segments_on_cell_boundaries_and_rays_through_grid_corners():
    # Cell size 10 and bounding box from -5 (half a cell of margin): grid lines at 10 k.
    seg = np.array(
        [
            [[-5.0, -5.0], [-5.0, -4.0]],
            [[45.0, 45.0], [45.0, 44.0]],
            [[10.0, -5.0], [10.0, 45.0]],  # on a vertical grid line
            [[-5.0, 20.0], [45.0, 20.0]],  # on a horizontal grid line
            [[30.0, 30.0], [40.0, 40.0]],  # diagonal through grid corners
        ]
    )
    grid = SegmentGrid(seg, cell_size_m=10.0)
    assert grid.x0 == -10.0 and grid.y0 == -10.0
    ox = np.array([0.0, 0.0, 20.0, 10.0, 2.0, 40.0])
    oy = np.array([0.0, 10.0, 0.0, 7.0, 2.0, 30.0])
    ang = np.array([np.pi / 4, 0.0, np.pi / 2, 0.0, np.arctan2(3.0, 1.0), np.pi / 2])
    ang = np.column_stack([ang, ang + np.pi, ang + np.pi / 2, ang - np.pi / 4])
    got = grid.raycast(ox, oy, ang, 100.0)
    np.testing.assert_allclose(got, brute_force(seg, ox, oy, ang, 100.0), atol=1e-9)
    assert got[0, 0] == pytest.approx(10.0 * np.sqrt(2.0))  # x=10 line at a grid corner
    assert got[3, 0] == pytest.approx(0.0, abs=1e-9)  # starts on the x=10 line
    assert got[5, 0] == pytest.approx(10.0)  # diagonal wall end point (40, 40), a grid corner


def test_ray_starting_on_a_wall_returns_zero():
    grid = SegmentGrid(SQUARE, cell_size_m=4.0)
    ox = np.array([7.3, 20.0, 0.0])
    oy = np.array([0.0, 11.1, 0.0])
    ang = np.array([[0.4, 2.0, -1.0], [3.0, 1.0, -2.0], [0.7, 1.2, 4.0]])
    got = grid.raycast(ox, oy, ang, 100.0)
    np.testing.assert_allclose(got, 0.0, atol=1e-9)


def test_ray_collinear_with_a_wall():
    seg = np.array([[[5.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [12.0, 3.0]]])
    grid = SegmentGrid(seg, cell_size_m=2.0)
    got = grid.raycast(np.array([0.0, 7.0]), np.array([0.0, 0.0]), np.array([[0.0], [0.0]]), 50.0)
    np.testing.assert_allclose(got, [[5.0], [0.0]], atol=1e-12)


def test_single_car_and_shapes(teardrop_track):
    grid = SegmentGrid.from_track(teardrop_track)
    p = teardrop_track.point_at(10.0)
    got = grid.raycast(np.array([p[0]]), np.array([p[1]]), np.zeros((1, 9)), 200.0)
    assert got.shape == (1, 9) and got.dtype == np.float64
    assert grid.raycast(np.zeros(0), np.zeros(0), np.zeros((0, 9)), 200.0).shape == (0, 9)
    assert grid.raycast(np.zeros(3), np.zeros(3), np.zeros((3, 0)), 200.0).shape == (3, 0)


def test_non_finite_origin_gives_nan():
    grid = SegmentGrid(SQUARE)
    got = grid.raycast(np.array([np.nan, 10.0]), np.array([1.0, 10.0]), np.zeros((2, 1)), 50.0)
    assert np.isnan(got[0, 0]) and got[1, 0] == pytest.approx(10.0)
