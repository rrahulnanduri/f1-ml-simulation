"""Tests for f1sim.data.alignment on synthetic, non-symmetric circuits (offline)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from f1sim.data.alignment import Alignment, align_to_centreline, apply_alignment

TRACK_LENGTH_M = 5000.0
RESOLUTION_M = 2.0


# --------------------------------------------------------------------------------------------
# Synthetic data


def _filleted_loop(vertices: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Closed polygon with each corner replaced by a tangent circular arc, sampled every ~1 m."""
    out = []
    n = len(vertices)
    for i in range(n):
        prev, v, nxt = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
        d1 = (v - prev) / np.linalg.norm(v - prev)
        d2 = (nxt - v) / np.linalg.norm(nxt - v)
        turn = np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)  # > 0 = left
        tl = radii[i] * np.tan(abs(turn) / 2.0)
        start = v - d1 * tl
        side = np.sign(turn) * np.array([-d1[1], d1[0]])  # towards the arc centre
        centre = start + side * radii[i]
        a0 = np.arctan2(*(start - centre)[::-1])
        m = max(2, int(np.ceil(radii[i] * abs(turn))))
        ang = a0 + np.sign(turn) * np.linspace(0.0, abs(turn), m)
        out.append(centre + radii[i] * np.column_stack([np.cos(ang), np.sin(ang)]))
        # Straight to the next corner's arc start (densified).
        end = out[-1][-1]
        nxt2 = vertices[(i + 2) % n]
        d3 = (nxt2 - nxt) / np.linalg.norm(nxt2 - nxt)
        turn2 = np.arctan2(d2[0] * d3[1] - d2[1] * d3[0], d2 @ d3)
        nxt_start = nxt - d2 * radii[(i + 1) % n] * np.tan(abs(turn2) / 2.0)
        k = max(2, int(np.ceil(np.linalg.norm(nxt_start - end))))
        out.append(end + np.linspace(0.0, 1.0, k)[1:-1, None] * (nxt_start - end))
    return np.vstack(out)


def make_circuit(seed: int) -> dict[str, np.ndarray]:
    """A ~5 km closed circuit: a random star-shaped polygon with straights and filleted corners.

    Random vertex angles and radii give no rotational or mirror symmetry; corner radii from
    15 m (hairpin) to 150 m (fast sweeper). Returned at ~2 m resolution without a repeated end
    point, with widths of 6 to 8 m per side.
    """
    rng = np.random.default_rng(seed)
    n_vertices = 11
    ang = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    ang += rng.uniform(-0.2, 0.2, n_vertices)
    rad = rng.uniform(0.55, 1.0, n_vertices)
    poly = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    radii = rng.choice([15.0, 25.0, 40.0, 70.0, 110.0, 150.0], n_vertices)
    poly *= TRACK_LENGTH_M / np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1))
    # Shrink corner radii until each pair of fillets fits on its straight.
    edge = np.roll(poly, -1, axis=0) - poly
    heading = np.arctan2(edge[:, 1], edge[:, 0])
    turn = np.abs(np.angle(np.exp(1j * (heading - np.roll(heading, 1)))))  # at each vertex
    for _ in range(20):
        tl = radii * np.tan(turn / 2.0)
        too_long = tl + np.roll(tl, -1) > 0.8 * np.linalg.norm(edge, axis=1)
        if not too_long.any():
            break
        radii[too_long | np.roll(too_long, 1)] *= 0.7
    for _ in range(3):  # rescale the polygon until the filleted loop is ~5 km
        loop = _filleted_loop(poly, radii)
        length = np.sum(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1))
        poly *= TRACK_LENGTH_M / length
    loop = _filleted_loop(poly, radii)
    x, y = loop[:, 0], loop[:, 1]
    step = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s_dense = np.concatenate([[0.0], np.cumsum(step)])
    n = int(round(s_dense[-1] / RESOLUTION_M))
    s = np.arange(n) * (s_dense[-1] / n)
    cx = np.interp(s, s_dense, np.append(x, x[0]))
    cy = np.interp(s, s_dense, np.append(y, y[0]))
    length = s_dense[-1]
    ph = rng.uniform(0.0, 2.0 * np.pi, 2)
    w_left = 7.0 + np.sin(2.0 * np.pi * 3.0 * s / length + ph[0])
    w_right = 7.0 + np.sin(2.0 * np.pi * 4.0 * s / length + ph[1])
    return {"x": cx, "y": cy, "s": s, "length": length, "w_left": w_left, "w_right": w_right}


def left_normals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit left normals (-sin h, cos h) of a closed polyline, from central differences."""
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    h = np.arctan2(dy, dx)
    return np.column_stack([-np.sin(h), np.cos(h)])


def _smooth_periodic(v: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Circular Gaussian smoothing via FFT."""
    freq = np.fft.rfftfreq(len(v))
    return np.fft.irfft(np.fft.rfft(v) * np.exp(-2.0 * (np.pi * freq * sigma_samples) ** 2), len(v))


def make_racing_line(
    track: dict[str, np.ndarray],
    seed: int,
    n_samples: int = 1000,
    noise_m: float = 0.5,
    style: str = "corners",
    max_offset_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unevenly sampled, noisy lap offset laterally by a smooth function within +/-max_offset_m.

    ``style="corners"``: a racing-like line, outside-apex-outside through every corner (a
    difference of Gaussians of the curvature) plus a +/-1 m random wobble.
    ``style="random"``: random smooth sinusoids (7, 11 and 17 per lap), uncorrelated with the
    corners; an adversarial case where a whole corner can be taken offset to one side.

    Starts at a random point and leaves a ~15 m start/finish gap. Returns (points in the track
    frame, designed offsets n at each sample before noise).
    """
    rng = np.random.default_rng(seed)
    s, length = track["s"], track["length"]
    ds = length / len(s)
    phase = 2.0 * np.pi * s / length
    if style == "corners":
        h = np.arctan2(np.roll(track["y"], -1) - track["y"], np.roll(track["x"], -1) - track["x"])
        kappa = np.angle(np.exp(1j * (h - np.roll(h, 1)))) / ds  # > 0 = left turn
        offset = _smooth_periodic(kappa, 15.0 / ds) - _smooth_periodic(kappa, 60.0 / ds)
        offset *= 4.0 / np.max(np.abs(offset))  # apex on the inside, entry/exit outside
        offset += 1.0 * np.sin(11 * phase + rng.uniform(0.0, 2.0 * np.pi))
    elif style == "random":
        offset = np.zeros_like(s)
        for cycles, amp in ((7, 0.5), (11, 0.3), (17, 0.2)):
            offset += amp * np.sin(cycles * phase + rng.uniform(0.0, 2.0 * np.pi))
    else:
        raise ValueError(style)
    offset *= max_offset_m / np.max(np.abs(offset))
    line = np.column_stack([track["x"], track["y"]]) + offset[:, None] * left_normals(
        track["x"], track["y"]
    )

    # Uneven spacing (dense in some places, sparse in others), small start/finish gap.
    spacing = 1.0 + 0.8 * np.sin(np.linspace(0.0, 6.0 * np.pi, n_samples)) ** 2
    spacing *= rng.uniform(0.5, 1.5, n_samples)
    s_samples = np.concatenate([[0.0], np.cumsum(spacing[:-1])])
    s_samples *= (length - 15.0) / s_samples[-1]
    s_samples = (s_samples + rng.uniform(0.0, length)) % length

    s_closed = np.append(s, length)
    px = np.interp(s_samples, s_closed, np.append(line[:, 0], line[0, 0]))
    py = np.interp(s_samples, s_closed, np.append(line[:, 1], line[0, 1]))
    n_true = np.interp(s_samples, s_closed, np.append(offset, offset[0]))
    pts = np.column_stack([px, py]) + rng.normal(0.0, noise_m, (n_samples, 2))
    return pts, n_true


def invert(a: Alignment, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of apply_alignment: track frame -> telemetry frame."""
    c, k = np.cos(a.rotation_rad), np.sin(a.rotation_rad)
    dx, dy = x - a.tx_m, y - a.ty_m
    u = (c * dx + k * dy) / a.scale
    v = (-k * dx + c * dy) / a.scale
    return u, (-v if a.mirrored else v)


def brute_force_offsets(track: dict[str, np.ndarray], pts: np.ndarray) -> np.ndarray:
    """Signed distance (left positive) to the nearest centre-line segment, checking every one."""
    a = np.column_stack([track["x"], track["y"]])
    d = np.roll(a, -1, axis=0) - a
    seg_len = np.hypot(d[:, 0], d[:, 1])
    tan = d / seg_len[:, None]
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        rel = p - a
        along = np.clip((rel * tan).sum(axis=1), 0.0, seg_len)
        dist = np.hypot(*(rel - along[:, None] * tan).T)
        j = np.argmin(dist)
        cross = tan[j, 0] * rel[j, 1] - tan[j, 1] * rel[j, 0]
        out[i] = dist[j] if cross >= 0 else -dist[j]
    return out


def random_transform(seed: int, mirrored: bool, scale: float = 1.0) -> Alignment:
    rng = np.random.default_rng(seed)
    return Alignment(
        rotation_rad=float(rng.uniform(-np.pi, np.pi)),
        tx_m=float(rng.uniform(-3000.0, 3000.0)),
        ty_m=float(rng.uniform(-3000.0, 3000.0)),
        scale=scale,
        mirrored=mirrored,
        median_dist_m=0.0,
        inside_ratio=0.0,
    )


def angle_diff(a: float, b: float) -> float:
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


# --------------------------------------------------------------------------------------------
# apply_alignment


def test_apply_alignment_matches_documented_formula() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 500, 50), rng.normal(0, 500, 50)
    for mirrored in (False, True):
        a = Alignment(0.7, 120.0, -45.0, 1.03, mirrored, 0.0, 0.0)
        m = -1.0 if mirrored else 1.0
        c, k = np.cos(a.rotation_rad), np.sin(a.rotation_rad)
        ex = a.scale * (c * x - k * m * y) + a.tx_m
        ey = a.scale * (k * x + c * m * y) + a.ty_m
        gx, gy = apply_alignment(a, x, y)
        np.testing.assert_allclose(gx, ex, atol=1e-9)
        np.testing.assert_allclose(gy, ey, atol=1e-9)

        # Same thing as a composition: mirror, then scale, then rotate, then translate.
        rot = np.array([[c, -k], [k, c]])
        mat = rot @ (a.scale * np.eye(2)) @ np.diag([1.0, m])
        composed = (mat @ np.vstack([x, y])).T + [a.tx_m, a.ty_m]
        np.testing.assert_allclose(np.column_stack([gx, gy]), composed, atol=1e-9)

        ix, iy = invert(a, gx, gy)
        np.testing.assert_allclose(ix, x, atol=1e-9)
        np.testing.assert_allclose(iy, y, atol=1e-9)


def test_apply_alignment_mirror_is_before_rotation() -> None:
    # Mirror then rotate 90 deg: (1, 0) -> (1, 0) -> (0, 1); (0, 1) -> (0, -1) -> (1, 0).
    a = Alignment(np.pi / 2, 0.0, 0.0, 1.0, True, 0.0, 0.0)
    gx, gy = apply_alignment(a, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(gx, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(gy, [1.0, 0.0], atol=1e-12)


def test_apply_alignment_accepts_scalars_and_lists() -> None:
    a = Alignment(0.0, 1.0, 2.0, 1.0, False, 0.0, 0.0)
    gx, gy = apply_alignment(a, [0.0, 1.0], [0.0, 1.0])
    np.testing.assert_allclose(gx, [1.0, 2.0])
    np.testing.assert_allclose(gy, [2.0, 3.0])
    sx, sy = apply_alignment(a, 3.0, 4.0)
    assert float(sx) == 4.0 and float(sy) == 6.0


# --------------------------------------------------------------------------------------------
# align_to_centreline


@pytest.mark.parametrize(
    ("track_seed", "transform_seed", "mirrored"),
    [(1, 11, False), (2, 12, True), (3, 13, False), (4, 14, True), (5, 15, True)],
)
def test_recovers_random_rigid_transform(track_seed: int, transform_seed: int, mirrored: bool):
    track = make_circuit(track_seed)
    truth_pts, n_design = make_racing_line(track, seed=100 + track_seed)
    truth = random_transform(transform_seed, mirrored)
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])

    a = align_to_centreline(px, py, track["x"], track["y"], track["w_left"], track["w_right"])

    assert a.mirrored == mirrored
    assert a.scale == 1.0
    assert np.degrees(angle_diff(a.rotation_rad, truth.rotation_rad)) < 0.5
    gx, gy = apply_alignment(a, px, py)
    err = np.hypot(gx - truth_pts[:, 0], gy - truth_pts[:, 1])
    # A racing line can be shifted ~1 m sideways without the data contradicting it, so the
    # worst point is allowed a little over 1 m (measured: <= 1.2 m over 40 random circuits).
    assert np.sqrt(np.mean(err**2)) < 1.0, f"rms error {np.sqrt(np.mean(err**2)):.3f} m"
    assert err.max() < 1.5, f"max error {err.max():.3f} m"

    # Statistics agree with an independent brute-force projection and with the design.
    n_exact = brute_force_offsets(track, truth_pts)
    assert abs(a.median_dist_m - np.median(np.abs(n_exact))) < 0.25
    assert abs(a.median_dist_m - np.median(np.abs(n_design))) < 0.3
    assert a.inside_ratio >= 0.98


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_adversarial_racing_line_still_aligns(seed: int) -> None:
    # Offsets uncorrelated with the corners (a whole corner taken 5 m to one side) bias any
    # centre-line fit; check it stays bounded (measured worst over 40 circuits: 2.8 m, 0.17 deg).
    track = make_circuit(20 + seed)
    truth_pts, _ = make_racing_line(track, seed=120 + seed, style="random")
    truth = random_transform(30 + seed, mirrored=bool(seed % 2))
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])
    a = align_to_centreline(px, py, track["x"], track["y"], track["w_left"], track["w_right"])
    assert a.mirrored == truth.mirrored
    assert np.degrees(angle_diff(a.rotation_rad, truth.rotation_rad)) < 0.5
    gx, gy = apply_alignment(a, px, py)
    assert np.hypot(gx - truth_pts[:, 0], gy - truth_pts[:, 1]).max() < 3.5
    assert a.inside_ratio >= 0.98


@pytest.mark.parametrize("seed", [8, 16, 17])
def test_corridor_polish_keeps_full_width_line_inside(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A line that holds 6.5 m offsets on 7 m half-widths: the centre-line ICP pulls it inwards
    # and pushes samples over the far edge. These seeds are ones where that happens, so the
    # corridor polish has to fix it.
    import f1sim.data.alignment as alignment

    track = make_circuit(seed)
    w = np.full(len(track["x"]), 7.0)
    truth_pts, _ = make_racing_line(
        track, seed=200 + seed, style="random", max_offset_m=6.5, noise_m=0.2
    )
    truth = random_transform(seed + 30, mirrored=bool(seed % 2))
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])
    args = (px, py, track["x"], track["y"], w, w)

    polished = align_to_centreline(*args)
    monkeypatch.setattr(alignment, "_POLISH_MIN_GAIN", 1.0)  # disables the polish
    icp_only = align_to_centreline(*args)

    def max_err(a: Alignment) -> float:
        gx, gy = apply_alignment(a, px, py)
        return float(np.hypot(gx - truth_pts[:, 0], gy - truth_pts[:, 1]).max())

    assert icp_only.inside_ratio < 0.998
    assert polished.inside_ratio >= 0.998
    assert max_err(polished) < max_err(icp_only)


def test_inside_ratio_uses_left_positive_convention() -> None:
    track = make_circuit(6)
    nl = left_normals(track["x"], track["y"])
    centre = np.column_stack([track["x"], track["y"]])
    idx = np.arange(0, len(centre), 3)
    w_left = np.full(len(centre), 9.0)
    w_right = np.full(len(centre), 3.0)
    # 6 m to the left: inside (w_left = 9). 6 m to the right: outside (w_right = 3).
    left_pts = centre[idx] + 6.0 * nl[idx]
    right_pts = centre[idx] - 6.0 * nl[idx]
    a_left = align_to_centreline(left_pts[:, 0], left_pts[:, 1], *centre.T, w_left, w_right)
    a_right = align_to_centreline(right_pts[:, 0], right_pts[:, 1], *centre.T, w_left, w_right)
    assert a_left.inside_ratio > 0.95
    assert a_right.inside_ratio < 0.3


def test_fit_scale_recovers_two_percent_error() -> None:
    track = make_circuit(7)
    truth_pts, _ = make_racing_line(track, seed=107)
    truth = random_transform(17, mirrored=True, scale=1.02)
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])
    args = (px, py, track["x"], track["y"], track["w_left"], track["w_right"])

    a = align_to_centreline(*args, fit_scale=True)
    assert abs(a.scale - 1.02) < 0.001
    assert a.mirrored
    assert np.degrees(angle_diff(a.rotation_rad, truth.rotation_rad)) < 0.5
    gx, gy = apply_alignment(a, px, py)
    assert np.hypot(gx - truth_pts[:, 0], gy - truth_pts[:, 1]).max() < 1.5
    assert a.inside_ratio >= 0.98

    # Without scale fitting the 2% error shows up in the statistics.
    rigid = align_to_centreline(*args)
    assert rigid.scale == 1.0
    assert rigid.median_dist_m > a.median_dist_m


def test_fit_scale_is_clamped() -> None:
    track = make_circuit(8)
    truth_pts, _ = make_racing_line(track, seed=108)
    truth = random_transform(18, mirrored=False, scale=1.12)
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])
    a = align_to_centreline(
        px, py, track["x"], track["y"], track["w_left"], track["w_right"], fit_scale=True
    )
    assert 0.95 <= a.scale <= 1.05


def test_different_circuit_is_rejectable() -> None:
    track = make_circuit(1)
    other = make_circuit(9)
    pts, _ = make_racing_line(other, seed=109)
    a = align_to_centreline(
        pts[:, 0], pts[:, 1], track["x"], track["y"], track["w_left"], track["w_right"]
    )
    assert a.inside_ratio < 0.5
    assert a.median_dist_m > 10.0


def test_ignores_non_finite_samples_and_closed_duplicate() -> None:
    track = make_circuit(10)
    truth_pts, _ = make_racing_line(track, seed=110)
    truth = random_transform(20, mirrored=False)
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])
    px[::97] = np.nan
    # Centre line passed with the first point repeated at the end still works.
    cx, cy = np.append(track["x"], track["x"][0]), np.append(track["y"], track["y"][0])
    wl, wr = np.append(track["w_left"], 7.0), np.append(track["w_right"], 7.0)
    a = align_to_centreline(px, py, cx, cy, wl, wr)
    ok = np.isfinite(px)
    gx, gy = apply_alignment(a, px[ok], py[ok])
    assert np.hypot(gx - truth_pts[ok, 0], gy - truth_pts[ok, 1]).max() < 1.0


def test_rejects_bad_input() -> None:
    track = make_circuit(1)
    c = (track["x"], track["y"], track["w_left"], track["w_right"])
    with pytest.raises(ValueError):
        align_to_centreline(np.zeros(5), np.zeros(5), *c)
    with pytest.raises(ValueError):
        align_to_centreline(np.zeros(50), np.zeros(49), *c)
    with pytest.raises(ValueError):
        align_to_centreline(np.zeros(50), np.zeros(50), *c[:3], c[3][:-1])


def test_runtime_1000_by_3000() -> None:
    track = make_circuit(11)
    # Resample the centre line to 3000 points.
    s3 = np.linspace(0.0, track["length"], 3000, endpoint=False)
    sc = np.append(track["s"], track["length"])
    cx = np.interp(s3, sc, np.append(track["x"], track["x"][0]))
    cy = np.interp(s3, sc, np.append(track["y"], track["y"][0]))
    w = np.full(3000, 7.0)
    hi_res = {"x": cx, "y": cy, "s": s3, "length": track["length"], "w_left": w, "w_right": w}
    truth_pts, _ = make_racing_line(hi_res, seed=111, n_samples=1000)
    truth = random_transform(21, mirrored=True)
    px, py = invert(truth, truth_pts[:, 0], truth_pts[:, 1])

    align_to_centreline(px[:50], py[:50], cx, cy, w, w)  # warm-up (imports, caches)
    t0 = time.perf_counter()
    a = align_to_centreline(px, py, cx, cy, w, w)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"{elapsed:.2f} s"
    assert a.inside_ratio >= 0.98
