"""Kerbs, run-off, the chequered strip and the white-line geometry: pure numpy, no window."""

from dataclasses import replace

import numpy as np
import pytest
import shapely

from _geometry_helpers import circle
from f1sim.app import render_track as rt
from f1sim.app.ui import theme
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track


def _area(tris: np.ndarray) -> float:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(
        0.5
        * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ).sum()
    )


@pytest.fixture(scope="module")
def plain_track() -> Track:
    x, y = circle(150.0, 471)  # ~2 m between points, counter-clockwise: left is inside
    n = x.size
    # build_track takes (x, y, w_right, w_left): white lines at r = 145 (left) and 156.
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 5.0), track_id="circle", name="Circle")
    return Track(data)


@pytest.fixture(scope="module")
def kerb_track(plain_track) -> Track:
    d = plain_track.data
    s, length = d.s, d.length_m
    left = np.where((s >= 100.0) & (s <= 160.0), 2.0, 0.0)  # a kerb plus 0.5 m of run-off
    right = np.full(s.size, 0.2)  # just over the line everywhere: no kerb
    right[(s >= 400.0) & (s <= 430.0)] = 1.0  # a 1 m kerb, no run-off
    right[(s >= length - 20.0) | (s <= 25.0)] = 0.8  # over the start line
    return Track(replace(d, limit_left=d.w_left + left, limit_right=d.w_right + right))


# ---------------------------------------------------------------------------- the runs


def test_kerb_runs_interpolate_the_threshold_crossings():
    s = np.arange(100.0)
    extra = np.zeros(100)
    extra[20:31] = 1.0
    (run,) = rt.kerb_runs(extra, s, 100.0, threshold=0.3)
    assert run == pytest.approx((19.3, 30.7))


def test_kerb_runs_merge_small_gaps_and_drop_slivers():
    s = np.arange(200.0)
    extra = np.zeros(200)
    extra[10:30] = 1.0
    extra[32:50] = 1.0  # 2 m gap: bridged
    extra[80:100] = 1.0  # 30 m gap: separate
    extra[150:152] = 1.0  # ~2.4 m long: dropped
    runs = rt.kerb_runs(extra, s, 200.0, threshold=0.3, min_len=4.0, merge_gap=4.0)
    assert runs == [pytest.approx((9.3, 49.7)), pytest.approx((79.3, 99.7))]


def test_kerb_runs_over_the_line_and_edge_cases():
    s = np.arange(100.0)
    extra = np.zeros(100)
    extra[95:] = 1.0
    extra[:6] = 1.0
    (run,) = rt.kerb_runs(extra, s, 100.0, threshold=0.3)
    assert run == pytest.approx((94.3, 105.7))  # starts before the line, ends after it
    assert rt.kerb_runs(np.zeros(100), s, 100.0) == []
    assert rt.kerb_runs(np.full(100, 0.3), s, 100.0) == []  # at the threshold: no kerb
    assert rt.kerb_runs(np.ones(100), s, 100.0) == [(0.0, 100.0)]


def test_block_edges_are_whole_blocks_of_about_3_m():
    e = rt.block_edges(31.0)
    assert e[0] == 0.0 and e[-1] == 31.0 and len(e) == 11
    assert np.diff(e) == pytest.approx(np.full(10, 3.1))
    assert len(rt.block_edges(1.0)) == 2  # at least one block
    assert np.diff(rt.block_edges(4.6)) == pytest.approx([2.3, 2.3])


# ------------------------------------------------------------------------ the geometry


def test_no_limits_no_kerbs(plain_track):
    for side in ("left", "right"):
        g = rt.kerb_geometry(plain_track, side)
        assert g.kerb.shape == (0, 3, 2) and g.runoff.shape == (0, 3, 2) and g.runs == []
    with pytest.raises(ValueError):
        rt.kerb_geometry(plain_track, "middle")


def test_left_kerb_layout(kerb_track):
    g = rt.kerb_geometry(kerb_track, "left")
    (run,) = g.runs
    ds = kerb_track.data.resolution_m + 1.0
    assert run[0] == pytest.approx(100.0, abs=ds) and run[1] == pytest.approx(160.0, abs=ds)
    (blocks,) = g.block_lengths
    assert np.all(np.abs(blocks - theme.KERB_BLOCK_M) < 0.5 * theme.KERB_BLOCK_M)
    assert np.ptp(blocks) == pytest.approx(0.0, abs=1e-9)  # equal blocks along the line
    # The white line inside the bend is shorter than the centre line: blocks are measured on
    # the white line itself.
    r_white = 150.0 - 5.0
    assert blocks.sum() == pytest.approx((run[1] - run[0]) * r_white / 150.0, rel=0.01)
    # Red and white alternate, starting with red; two triangles per strip segment.
    assert g.red.size == g.kerb.shape[0] and g.red[0]
    changes = np.count_nonzero(np.diff(g.red.astype(int)))
    assert changes == len(blocks) - 1
    # 1.5 m wide (the limit is 2 m out), then 0.5 m of run-off up to the limit.
    length = (run[1] - run[0]) * r_white / 150.0
    assert _area(g.kerb) == pytest.approx(length * 1.5, rel=0.03)
    assert _area(g.runoff) == pytest.approx(length * 0.5, rel=0.06)
    assert _area(g.plain) == pytest.approx(_area(g.kerb) + _area(g.runoff), rel=1e-6)
    radii = np.hypot(g.kerb[..., 0], g.kerb[..., 1])
    assert radii.max() == pytest.approx(r_white, abs=0.01)  # inner edge: the white line
    assert radii.min() == pytest.approx(r_white - 1.5, abs=0.02)
    radii = np.hypot(g.runoff[..., 0], g.runoff[..., 1])
    assert radii.min() == pytest.approx(r_white - 2.0, abs=0.02)  # out to the limit


def test_kerb_inner_edge_lies_on_the_white_line(kerb_track):
    g = rt.kerb_geometry(kerb_track, "left")
    ring = shapely.LinearRing(kerb_track.left_wall())
    pts = g.kerb.reshape(-1, 2)
    d = shapely.distance(ring, shapely.points(pts))
    on_line = d < 1e-6
    assert on_line.mean() == pytest.approx(0.5, abs=0.01)  # half of the strip's corners
    # The others are the outer edge: 1.5 m out, narrowing to the threshold at the ends.
    assert d[~on_line].max() == pytest.approx(1.5, abs=0.01)
    assert d[~on_line].min() == pytest.approx(theme.KERB_MIN_EXTRA_M, abs=0.01)


def test_right_kerbs_narrow_and_over_the_line(kerb_track):
    g = rt.kerb_geometry(kerb_track, "right")
    length = kerb_track.length
    assert len(g.runs) == 2  # the 0.2 m extra elsewhere is below the threshold
    over, mid = sorted(g.runs, key=lambda r: r[1] - r[0], reverse=True)
    assert over[0] == pytest.approx(length - 20.0, abs=3.0)
    assert over[1] == pytest.approx(length + 25.0, abs=3.0)
    assert mid[0] == pytest.approx(400.0, abs=3.0)
    # Both kerbs are narrower than 1.5 m, so they reach the limit: no run-off.
    assert g.runoff.shape[0] == 0
    radii = np.hypot(g.kerb[..., 0], g.kerb[..., 1])
    r_white = 150.0 + 6.0
    assert radii.min() == pytest.approx(r_white, abs=0.01)
    assert radii.max() == pytest.approx(r_white + 1.0, abs=0.02)


# --------------------------------------------------------------------- strip and lines


def test_chequer_strip(plain_track):
    tris, light = rt.chequer_triangles(plain_track)
    cols = round((6.0 + 5.0) / theme.CHEQUER_M)
    assert tris.shape == (2 * 2 * cols, 3, 2) and light.shape == (tris.shape[0],)
    assert light.sum() == light.size // 2  # half the squares light
    assert light[0] and not light[2]  # neighbours alternate
    assert _area(tris) == pytest.approx(2 * theme.CHEQUER_M * 11.0, rel=0.01)
    # Centred on the line: as much of the strip before s = 0 as after.
    s, _, _ = plain_track.project(tris[..., 0].ravel(), tris[..., 1].ravel())
    s = (s + 0.5 * plain_track.length) % plain_track.length - 0.5 * plain_track.length
    assert s.min() == pytest.approx(-theme.CHEQUER_M, abs=0.05)
    assert s.max() == pytest.approx(theme.CHEQUER_M, abs=0.05)


def test_sector_ticks_sit_on_the_white_lines(plain_track):
    track = Track(replace(plain_track.data, sectors_s=(0.0, 300.0, 600.0)))
    ticks = rt.sector_ticks(track)
    assert ticks.shape == (4, 2, 2)
    for (x, y), (dx, dy) in ticks:
        assert np.hypot(dx, dy) == pytest.approx(1.0)
        r = np.hypot(x, y)
        assert r == pytest.approx(145.0, abs=0.01) or r == pytest.approx(156.0, abs=0.01)
        # Pointing away from the road: inwards on the inner line, outwards on the outer.
        outward = (x * dx + y * dy) / r
        assert outward == pytest.approx(-1.0 if r < 150 else 1.0, abs=1e-6)
    assert rt.sector_ticks(plain_track).shape == (0, 2, 2)


def test_constant_width_line_vertices():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    off = rt.miter_offsets(square)
    # At a right-angle corner the miter is the diagonal, sqrt(2) long, so the strip's
    # edges stay one half-width from both segments.
    assert np.hypot(off[:, 0], off[:, 1]) == pytest.approx(np.full(4, np.sqrt(2)))
    assert off[0] == pytest.approx([1.0, 1.0])  # left = inside of a counter-clockwise loop
    verts = rt.line_vertices(square)
    assert verts.shape == (4 * 6, 5)
    assert set(verts[:, 4]) == {-1.0, 1.0}
    # A straight open line: plain unit normals.
    line = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    assert rt.miter_offsets(line, closed=False) == pytest.approx(np.tile([0.0, 1.0], (3, 1)))
    assert rt.line_vertices(line, closed=False).shape == (2 * 6, 5)
