"""Build a :class:`TrackData` from a raw closed centre line with widths (revamp-plan 4.2)."""

from __future__ import annotations

import warnings

import numpy as np
import shapely
from scipy.interpolate import splev, splprep
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from f1sim.track.track import Track, TrackData

# Fraction of the local limit (1/|curvature|, or the local reach) a width is clipped to.
_WALL_CLIP = 0.95
# Gauss-Legendre nodes for the arc-length integral of the spline.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(5)


class TrackBuildError(ValueError):
    """The input can't be turned into a valid track."""


def _as_1d(name: str, v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim != 1:
        raise TrackBuildError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise TrackBuildError(f"{name} contains non-finite values")
    return a


def _clean_loop(x, y, w_right, w_left):
    """Drop a repeated end point and zero-length steps; return the open loop arrays."""
    step = np.hypot(np.diff(x), np.diff(y))
    scale = float(np.median(step[step > 0])) if np.any(step > 0) else 0.0
    if scale == 0.0:
        raise TrackBuildError("all input points coincide")
    tol = 1e-6 * scale
    keep = np.concatenate([[True], step > tol])  # drop points repeating their predecessor
    x, y, w_right, w_left = x[keep], y[keep], w_right[keep], w_left[keep]
    while x.size > 1 and np.hypot(x[-1] - x[0], y[-1] - y[0]) <= tol:  # repeated end point
        x, y, w_right, w_left = x[:-1], y[:-1], w_right[:-1], w_left[:-1]
    return x, y, w_right, w_left


def _noise_sigma(x: np.ndarray, y: np.ndarray) -> float:
    """Robust estimate of per-coordinate point noise from wrapped fourth differences.

    For i.i.d. noise the fourth difference has variance ``70 sigma^2``; a smooth curve sampled
    every few metres contributes little, and the median ignores the tight corners. Below a
    millimetre the input counts as clean (0).
    """
    d4 = np.concatenate(
        [
            np.roll(v, -2) - 4.0 * np.roll(v, -1) + 6.0 * v - 4.0 * np.roll(v, 1) + np.roll(v, 2)
            for v in (x, y)
        ]
    )
    sigma = 1.4826 * float(np.median(np.abs(d4 - np.median(d4)))) / np.sqrt(70.0)
    return sigma if sigma >= 1e-3 else 0.0


def _fit_spline(x: np.ndarray, y: np.ndarray, max_shift_m: float, track_id: str):
    """Periodic smoothing spline that moves no input point more than ``max_shift_m``.

    The smoothing factor starts at the noise level (``s = 2 m sigma^2``, so clean input is
    interpolated and keeps its curvature); if that moves a point too far, the largest
    ``s`` below it that respects the bound is found by bisection. The bound is then verified.
    Returns ``(tck, u)`` with ``u`` the chord-length parameter of the input points.
    """
    xc, yc = np.append(x, x[0]), np.append(y, y[0])
    u = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))])
    m = x.size

    def fit(smooth: float):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (tck, _), _, ier, _ = splprep(
                [xc, yc], u=u, s=smooth, per=1, k=3, full_output=1, quiet=1
            )
        if ier > 0:
            return None, np.inf
        ex, ey = splev(u[:-1], tck)
        return tck, float(np.max(np.hypot(ex - x, ey - y)))

    s_noise = 2.0 * m * _noise_sigma(x, y) ** 2 if max_shift_m > 0.0 else 0.0
    best, shift = fit(s_noise)
    if best is None or shift > max_shift_m:
        best, shift = fit(0.0)
        if best is None or shift > max_shift_m + 1e-6:
            raise TrackBuildError(f"{track_id}: interpolating spline failed (shift {shift:.3g})")
        lo, hi = 0.0, s_noise
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            tck, shift = fit(mid)
            if tck is not None and shift <= max_shift_m:
                lo, best = mid, tck
            else:
                hi = mid
    ex, ey = splev(u[:-1], best)
    shift = float(np.max(np.hypot(ex - x, ey - y)))
    if shift > max_shift_m + 1e-6:  # pragma: no cover - guarded by the search above
        raise TrackBuildError(f"{track_id}: spline moved a point {shift:.3f} m > {max_shift_m} m")
    return best, u


def _arc_length_table(tck, u_end: float, n_intervals: int):
    """Cumulative arc length of the spline on a uniform parameter grid (Gauss-Legendre)."""
    uf = np.linspace(0.0, u_end, n_intervals + 1)
    half = 0.5 * np.diff(uf)
    mid = 0.5 * (uf[1:] + uf[:-1])
    nodes = mid[:, None] + half[:, None] * _GL_X[None, :]
    dx, dy = splev(nodes.ravel(), tck, der=1)
    speed = np.hypot(dx, dy).reshape(nodes.shape)
    seg = half * (speed @ _GL_W)
    return uf, np.concatenate([[0.0], np.cumsum(seg)])


def _arc_length_at(tck, uf, sf, u):
    """Arc length at parameter values ``u`` using the table plus a partial-interval integral."""
    j = np.clip(np.searchsorted(uf, u, side="right") - 1, 0, uf.size - 2)
    half = 0.5 * (u - uf[j])
    mid = 0.5 * (u + uf[j])
    nodes = mid[:, None] + half[:, None] * _GL_X[None, :]
    dx, dy = splev(nodes.ravel(), tck, der=1)
    speed = np.hypot(dx, dy).reshape(nodes.shape)
    return sf[j] + half * (speed @ _GL_W)


def _resample(tck, u_end: float, resolution_m: float):
    """Parameters of points equally spaced in arc length, and the total length."""
    n_int = max(2000, int(np.ceil(16 * u_end / resolution_m)))
    uf, sf = _arc_length_table(tck, u_end, n_int)
    length = float(sf[-1])
    m = max(8, int(round(length / resolution_m)))
    s_target = np.arange(m) * (length / m)
    ut = np.interp(s_target, sf, uf)
    for _ in range(2):  # Newton refinement of s(u) = s_target
        err = _arc_length_at(tck, uf, sf, ut) - s_target
        dx, dy = splev(ut, tck, der=1)
        ut = np.clip(ut - err / np.hypot(dx, dy), 0.0, u_end)
    ut[0] = 0.0
    return ut, s_target, length, (uf, sf)


def _local_reach(px, py, nl, w_max: float):
    """Per point, radius of the largest disc tangent to the centre line on each side.

    The disc at point k on the left has its centre on ``p_k + r nL_k`` and passes through the
    nearest other centre-line point: ``r = |d|^2 / (2 d.nL)`` minimised over neighbours ``d``.
    Widths below this reach keep wall "rungs" from crossing, i.e. the walls from folding.
    Only neighbours within ``2 w_max / clip`` matter, since ``r >= |d| / 2``.
    """
    m = px.size
    reach_l = np.full(m, np.inf)
    reach_r = np.full(m, np.inf)
    pts = np.column_stack([px, py])
    pairs = cKDTree(pts).query_pairs(2.0 * w_max / _WALL_CLIP + 1e-9, output_type="ndarray")
    if pairs.size == 0:
        return reach_l, reach_r
    for k, j in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        d = pts[j] - pts[k]
        dn = np.einsum("ij,ij->i", d, nl[k])
        d2 = np.einsum("ij,ij->i", d, d)
        with np.errstate(divide="ignore"):
            r = d2 / (2.0 * np.abs(dn))
        left = dn > 0.0
        np.minimum.at(reach_l, k[left], r[left])
        right = dn < 0.0
        np.minimum.at(reach_r, k[right], r[right])
    return reach_l, reach_r


def _ring(points: np.ndarray) -> shapely.LineString:
    return shapely.LineString(np.vstack([points, points[:1]]))


def _self_crossing_points(points: np.ndarray) -> np.ndarray:
    """Points where the closed polyline crosses or touches itself, ``(K, 2)``."""
    ring = _ring(points)
    if ring.is_simple:
        return np.empty((0, 2))
    parts = shapely.get_parts(shapely.unary_union(ring))
    ends = []
    for part in parts:
        c = np.asarray(part.coords)
        ends.append(c[0])
        ends.append(c[-1])
    ends = np.round(np.asarray(ends), 6)
    nodes, counts = np.unique(ends, axis=0, return_counts=True)
    found = nodes[counts > 2]
    if found.size == 0:  # not simple but no proper node (e.g. collinear overlap)
        return np.asarray(ring.coords)[:1]
    return found


def wall_self_crossings(data: TrackData) -> tuple[int, int]:
    """Number of self-crossing points of the ``(left, right)`` walls (0 = simple)."""
    track = Track(data)
    return (
        len(_self_crossing_points(track.left_wall())),
        len(_self_crossing_points(track.right_wall())),
    )


def _where(points: np.ndarray, px: np.ndarray, py: np.ndarray, s: np.ndarray) -> str:
    near = [s[np.argmin(np.hypot(px - qx, py - qy))] for qx, qy in points[:5]]
    more = ", ..." if len(points) > 5 else ""
    return ", ".join(f"{v:.0f}" for v in near) + more


def build_track(
    x,
    y,
    w_right,
    w_left,
    *,
    track_id: str,
    name: str,
    resolution_m: float = 2.0,
    max_shift_m: float = 0.5,
    width_filter_m: float = 20.0,
    source: dict | None = None,
) -> TrackData:
    """Build a smoothed, evenly resampled track from a raw closed centre line with widths.

    The input loop may repeat its first point at the end. Steps (revamp-plan 4.2): periodic
    smoothing spline moving no point more than ``max_shift_m``; arc-length resampling at
    ``resolution_m``; widths interpolated and median-filtered over ``width_filter_m``;
    heading and curvature from the spline; wall repair; ``TrackBuildError`` if a wall still
    self-intersects.
    """
    x = _as_1d("x", x)
    y = _as_1d("y", y)
    w_right = _as_1d("w_right", w_right)
    w_left = _as_1d("w_left", w_left)
    if not (x.size == y.size == w_right.size == w_left.size):
        raise TrackBuildError("x, y, w_right and w_left must have the same length")
    if np.any(w_right < 0) or np.any(w_left < 0):
        raise TrackBuildError(f"{track_id}: widths must be non-negative")
    if resolution_m <= 0 or max_shift_m < 0 or width_filter_m < 0:
        raise TrackBuildError("resolution_m must be > 0, max_shift_m and width_filter_m >= 0")
    x, y, w_right, w_left = _clean_loop(x, y, w_right, w_left)
    if x.size < 4:
        raise TrackBuildError(f"{track_id}: need at least 4 distinct points, got {x.size}")
    if not _ring(np.column_stack([x, y])).is_simple:
        raise TrackBuildError(
            f"{track_id}: the input centre line crosses itself (figure-eight or bad data); "
            "a track must be a simple closed loop"
        )

    # 1. Smoothing spline, 2. arc-length resampling.
    tck, u = _fit_spline(x, y, max_shift_m, track_id)
    ut, s, length, (uf, sf) = _resample(tck, float(u[-1]), resolution_m)
    px, py = (np.asarray(v) for v in splev(ut, tck))
    m = s.size
    spacing = length / m

    # 3. Widths: interpolate periodically in s, then a wrapping median filter.
    s_in = np.interp(u[:-1], uf, sf)
    wl = np.interp(s, s_in, w_left, period=length)
    wr = np.interp(s, s_in, w_right, period=length)
    k = int(round(width_filter_m / spacing))
    k = min(k + (1 - k % 2), m - (1 - m % 2))  # odd, at most m
    if k >= 3:
        wl = median_filter(wl, size=k, mode="wrap")
        wr = median_filter(wr, size=k, mode="wrap")

    # 4. Heading and signed curvature from the spline derivatives.
    dx, dy = (np.asarray(v) for v in splev(ut, tck, der=1))
    ddx, ddy = (np.asarray(v) for v in splev(ut, tck, der=2))
    heading = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / np.hypot(dx, dy) ** 3

    # 5. Wall repair. Inside of the bend: clip to 0.95/|k| (left side when turning left).
    with np.errstate(divide="ignore"):
        lim = _WALL_CLIP / np.abs(curvature)
    wl = np.where((curvature > 0) & (wl > lim), lim, wl)
    wr = np.where((curvature < 0) & (wr > lim), lim, wr)
    # The curvature rule alone can't undo the inner-wall loop where a tight bend meets a
    # straight (that loop crosses the straights' walls), so also clip to the local reach.
    nl = np.column_stack([-np.sin(heading), np.cos(heading)])
    reach_l, reach_r = _local_reach(px, py, nl, float(max(wl.max(), wr.max(), 0.0)))
    wl = np.minimum(wl, _WALL_CLIP * reach_l)
    wr = np.minimum(wr, _WALL_CLIP * reach_r)

    # 7. Direction from the signed (shoelace) area.
    area2 = float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
    direction = "counterclockwise" if area2 > 0 else "clockwise"

    data = TrackData(
        id=track_id,
        name=name,
        x=px,
        y=py,
        s=s,
        w_left=wl,
        w_right=wr,
        heading=heading,
        curvature=curvature,
        length_m=length,
        resolution_m=resolution_m,
        direction=direction,
        source=dict(source or {}),
    )

    # 6. Fail if either wall still isn't simple.
    track = Track(data)
    problems = []
    for side, wall in (("left", track.left_wall()), ("right", track.right_wall())):
        crossings = _self_crossing_points(wall)
        if len(crossings):
            problems.append(
                f"{side} wall crosses itself at {len(crossings)} point(s) near "
                f"s = {_where(crossings, px, py, s)} m"
            )
    if problems:
        raise TrackBuildError(
            f"{track_id}: " + "; ".join(problems) + ". Widths may overlap another part of "
            "the circuit; check the source widths."
        )
    return data
