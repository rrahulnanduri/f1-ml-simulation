"""Align a FastF1 telemetry path to a track centre line (revamp-plan section 4.3).

The telemetry path (already in metres) lives in an unknown frame. We look for the similarity
transform (rotation, translation, optional mirroring, optional scale) that puts it onto the
centre line:

1. Coarse search: both clouds centred on their arc-length centroids, rotations every 5 degrees,
   both mirror states, scored by a robust two-way median nearest distance on subsamples.
2. Refinement: point-to-segment ICP from the best few coarse candidates, with trimmed,
   arc-length weighted correspondences and a closed-form 2D Procrustes update per iteration.
3. Corridor polish: a racing line legitimately uses the full track width, and a pure centre-line
   fit drags it inwards (on real Silverstone data by ~3 m, pushing 10% of samples over the
   edges). When the ICP fit leaves samples outside, a short least-squares pass penalises
   samples beyond the edges, with a weak centre-line term; its result is kept only if it
   raises the inside share.
4. Fit statistics from exact point-to-segment projection: median distance to the centre line and
   the share of samples inside the track edges.

Conventions follow ``docs/m1-interfaces.md``: closed loops have no repeated end point, the left
normal is ``(-sin h, cos h)`` and the lateral offset ``n`` is positive to the left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

__all__ = ["Alignment", "align_to_centreline", "apply_alignment"]

COARSE_STEP_DEG = 5.0
"""Rotation step of the coarse search, degrees."""
SCALE_LIMIT = 0.05
"""With ``fit_scale=True`` the scale is clamped to ``1 +/- SCALE_LIMIT``."""

_DENSE_STEP_M = 1.0  # centre-line densification for the KD tree (nearest-segment candidates)
_COARSE_SAMPLES = 256  # telemetry / centre-line subsample sizes for the coarse search
_REFINE_SAMPLES = 400  # telemetry subsample for refining each coarse candidate
_N_CANDIDATES = 4  # coarse local minima refined with ICP
_TRIM_FACTOR = 3.0  # ICP keeps correspondences with dist <= max(TRIM_FACTOR * median, TRIM_FLOOR)
_TRIM_FLOOR_M = 10.0
_GAP_FACTOR = 10.0  # telemetry steps longer than this x median step count as gaps for weighting
_POLISH_SAMPLES = 1500  # telemetry subsample for the corridor polish
_CORRIDOR_SOFTNESS_M = 0.5  # softplus width of the edge penalty
_CENTRE_WEIGHT = 0.003  # weight of the squared centre-line distance in the corridor polish
_POLISH_HUBER_M = 2.0  # residuals above this count linearly (robust to stray samples)
_POLISH_MIN_GAIN = 0.002  # the polish must raise inside_ratio by more than this to be kept


@dataclass(frozen=True)
class Alignment:
    """Similarity transform from the telemetry frame to the track frame, plus fit statistics.

    ``median_dist_m`` is the median absolute distance of the aligned samples to the centre line
    and ``inside_ratio`` the share of samples with ``-w_right <= n <= w_left``.
    """

    rotation_rad: float
    tx_m: float
    ty_m: float
    scale: float
    mirrored: bool
    median_dist_m: float
    inside_ratio: float


def apply_alignment(a: Alignment, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Map telemetry-frame points into the track frame.

    The steps run in this order: mirror, then scale about the origin, then rotate about the
    origin, then translate. With ``m = -1`` if ``a.mirrored`` else ``1``,
    ``s = a.scale``, ``c = cos(a.rotation_rad)`` and ``k = sin(a.rotation_rad)``::

        x' = s * (c * x - k * m * y) + a.tx_m
        y' = s * (k * x + c * m * y) + a.ty_m

    In matrix form, ``p' = s * R(theta) @ diag(1, m) @ p + t``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if a.mirrored:
        y = -y
    c, k = np.cos(a.rotation_rad), np.sin(a.rotation_rad)
    xs, ys = a.scale * x, a.scale * y
    return c * xs - k * ys + a.tx_m, k * xs + c * ys + a.ty_m


def align_to_centreline(
    px: ArrayLike,
    py: ArrayLike,
    cx: ArrayLike,
    cy: ArrayLike,
    w_left: ArrayLike,
    w_right: ArrayLike,
    *,
    fit_scale: bool = False,
) -> Alignment:
    """Find the transform that puts the telemetry path ``(px, py)`` onto the centre line.

    ``(cx, cy)`` is a closed centre line (no repeated end point) with per-point widths
    ``w_left`` and ``w_right``. Scale stays exactly 1 unless ``fit_scale`` is True, in which case
    it is fitted and clamped to ``1 +/- SCALE_LIMIT``. Non-finite telemetry samples are ignored.

    The result is a best fit, not a verdict: callers decide from the statistics whether to
    accept it. For reference, a correct fit of real FastF1 telemetry (2024 Silverstone) to the
    TUM layout gives ``median_dist_m ~ 3.9`` and ``inside_ratio ~ 0.96`` (cars use the kerbs:
    99.8% of samples are within 1 m of the edges), while a wrong circuit gives ``< 0.3``.
    """
    p = _telemetry_points(px, py)
    cl = _Centreline(cx, cy, w_left, w_right)

    w_full = _arc_weights(p)
    sub_coarse = _even_subsample(w_full, _COARSE_SAMPLES)
    sub_refine = _even_subsample(w_full, _REFINE_SAMPLES)
    centroid_p = (w_full[:, None] * p).sum(axis=0) / w_full.sum()
    c_sub = cl.dense[np.linspace(0, len(cl.dense), _COARSE_SAMPLES, endpoint=False).astype(int)]

    # 1. Coarse search over rotation and mirroring, centroids matched.
    candidates = _coarse_candidates(p[sub_coarse] - centroid_p, c_sub - cl.centroid, cl)

    # 2. Refine the best few candidates on a subsample, keep the best two-way fit.
    best: tuple[float, bool, float, float, np.ndarray] | None = None
    q_ref = p[sub_refine]
    w_ref = np.ones(len(q_ref))
    for mirrored, theta in candidates:
        qm = _mirror(q_ref, mirrored)
        t0 = cl.centroid - _rot(theta) @ _mirror(centroid_p[None, :], mirrored)[0]
        th, sc, t = _icp(qm, w_ref, cl, theta, 1.0, t0, fit_scale, max_iter=40, tol_m=0.05)
        score = _two_way_score(sc * qm @ _rot(th).T + t, c_sub, cl)
        if best is None or score < best[0]:
            best = (score, mirrored, th, sc, t)
    assert best is not None
    _, mirrored, theta, scale, t = best

    # 3. Final ICP on all samples with arc-length weights.
    qm = _mirror(p, mirrored)
    theta, scale, t = _icp(qm, w_full, cl, theta, scale, t, fit_scale, max_iter=200, tol_m=1e-4)

    if not fit_scale:
        scale = 1.0
    median_dist, inside_ratio = _fit_stats(cl, scale * qm @ _rot(theta).T + t)

    # 4. Corridor polish: a racing line uses the track width, so a pure centre-line fit drags
    #    it inwards and pushes samples over the edges. Only when that happens, refit keeping
    #    samples inside the edges; keep the result only if it clearly helps. (When every sample
    #    is already inside, the edges don't pin the transform and the ICP answer is better.)
    if inside_ratio < 1.0 - _POLISH_MIN_GAIN:
        sub_polish = _even_subsample(w_full, _POLISH_SAMPLES)
        th_p, sc_p, t_p = _corridor_polish(qm[sub_polish], cl, theta, scale, t, fit_scale)
        med_p, inside_p = _fit_stats(cl, sc_p * qm @ _rot(th_p).T + t_p)
        if inside_p > inside_ratio + _POLISH_MIN_GAIN:
            theta, scale, t, median_dist, inside_ratio = th_p, sc_p, t_p, med_p, inside_p

    return Alignment(
        rotation_rad=float(np.arctan2(np.sin(theta), np.cos(theta))),
        tx_m=float(t[0]),
        ty_m=float(t[1]),
        scale=float(scale),
        mirrored=bool(mirrored),
        median_dist_m=median_dist,
        inside_ratio=inside_ratio,
    )


# --------------------------------------------------------------------------------------------
# Internals


@dataclass(frozen=True)
class _Projection:
    dist: np.ndarray  # unsigned distance to the nearest segment
    n: np.ndarray  # signed lateral offset, positive to the left, |n| == dist
    seg: np.ndarray  # segment index i (from point i to point i+1, wrapping)
    u: np.ndarray  # fraction along the segment, [0, 1]
    foot: np.ndarray  # (N, 2) closest point on the centre line


class _Centreline:
    """Closed polyline with widths, exact point-to-segment projection via a dense KD tree."""

    def __init__(self, cx: ArrayLike, cy: ArrayLike, w_left: ArrayLike, w_right: ArrayLike):
        pts = np.column_stack(
            [np.asarray(cx, dtype=np.float64).ravel(), np.asarray(cy, dtype=np.float64).ravel()]
        )
        wl = np.asarray(w_left, dtype=np.float64).ravel()
        wr = np.asarray(w_right, dtype=np.float64).ravel()
        if not (len(pts) == len(wl) == len(wr)):
            raise ValueError("cx, cy, w_left and w_right must have the same length")
        if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(wl)) and np.all(np.isfinite(wr))):
            raise ValueError("centre line and widths must be finite")
        # Drop zero-length segments (repeated points, including a repeated closing point).
        keep = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T) > 1e-9
        pts, wl, wr = pts[keep], wl[keep], wr[keep]
        if len(pts) < 3:
            raise ValueError("centre line needs at least 3 distinct points")

        d = np.roll(pts, -1, axis=0) - pts
        self.a = pts
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.tan = d / self.seg_len[:, None]
        self.w_left = wl
        self.w_right = wr

        mid = pts + 0.5 * d
        self.centroid = (self.seg_len[:, None] * mid).sum(axis=0) / self.seg_len.sum()

        n_sub = np.maximum(1, np.ceil(self.seg_len / _DENSE_STEP_M)).astype(np.int64)
        seg_idx = np.repeat(np.arange(len(pts)), n_sub)
        start = np.repeat(np.cumsum(n_sub) - n_sub, n_sub)
        u = (np.arange(len(seg_idx)) - start) / np.repeat(n_sub, n_sub)
        self.dense = pts[seg_idx] + u[:, None] * d[seg_idx]
        self.dense_seg = seg_idx
        self.tree = cKDTree(self.dense)

    def project(self, p: np.ndarray, k: int = 4) -> _Projection:
        """Exact projection onto the nearest of the segments owning the k nearest dense samples."""
        k = min(k, len(self.dense))
        _, idx = self.tree.query(p, k=k)
        idx = np.asarray(idx).reshape(len(p), k)
        cand = self.dense_seg[idx]  # (N, k)
        a = self.a[cand]
        tan = self.tan[cand]
        rel = p[:, None, :] - a
        along = np.clip((rel * tan).sum(axis=-1), 0.0, self.seg_len[cand])
        foot = a + along[..., None] * tan
        diff = p[:, None, :] - foot
        dist = np.hypot(diff[..., 0], diff[..., 1])
        j = np.argmin(dist, axis=1)
        rows = np.arange(len(p))
        seg = cand[rows, j]
        dist_best = dist[rows, j]
        cross = tan[rows, j, 0] * rel[rows, j, 1] - tan[rows, j, 1] * rel[rows, j, 0]
        n = np.where(cross >= 0.0, dist_best, -dist_best)
        return _Projection(
            dist=dist_best,
            n=n,
            seg=seg,
            u=along[rows, j] / self.seg_len[seg],
            foot=foot[rows, j],
        )

    def widths(self, seg: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Widths linearly interpolated along each segment."""
        nxt = (seg + 1) % len(self.a)
        wl = (1.0 - u) * self.w_left[seg] + u * self.w_left[nxt]
        wr = (1.0 - u) * self.w_right[seg] + u * self.w_right[nxt]
        return wl, wr


def _fit_stats(cl: _Centreline, aligned: np.ndarray) -> tuple[float, float]:
    """(median distance to the centre line, share of samples inside the edges)."""
    proj = cl.project(aligned, k=8)
    w_l, w_r = cl.widths(proj.seg, proj.u)
    inside = (proj.n >= -w_r) & (proj.n <= w_l)
    return float(np.median(proj.dist)), float(np.mean(inside))


def _telemetry_points(px: ArrayLike, py: ArrayLike) -> np.ndarray:
    x = np.asarray(px, dtype=np.float64).ravel()
    y = np.asarray(py, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("px and py must have the same length")
    p = np.column_stack([x, y])
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 10:
        raise ValueError("telemetry path needs at least 10 finite points")
    return p


def _arc_weights(p: np.ndarray) -> np.ndarray:
    """Per-sample arc-length weights of the open path, so uneven sampling doesn't bias the fit.

    Steps much longer than typical (gaps, the start/finish join) are capped.
    """
    step = np.hypot(*np.diff(p, axis=0).T)
    positive = step[step > 0]
    typical = float(np.median(positive)) if len(positive) else 1.0
    step = np.minimum(step, _GAP_FACTOR * typical)
    w = np.zeros(len(p))
    w[:-1] += 0.5 * step
    w[1:] += 0.5 * step
    w = np.maximum(w, 1e-3 * typical)  # duplicate points still count a little
    return w


def _even_subsample(w: np.ndarray, count: int) -> np.ndarray:
    """Indices of samples spread evenly in (weighted) arc length."""
    if len(w) <= count:
        return np.arange(len(w))
    cum = np.cumsum(w)
    targets = np.linspace(0.0, cum[-1], count, endpoint=False) + 0.5 * cum[-1] / count
    return np.unique(np.minimum(np.searchsorted(cum, targets), len(w) - 1))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _mirror(p: np.ndarray, mirrored: bool) -> np.ndarray:
    return p * np.array([1.0, -1.0]) if mirrored else p


def _coarse_candidates(
    p_centred: np.ndarray, c_centred: np.ndarray, cl: _Centreline
) -> list[tuple[bool, float]]:
    """Best local minima of the two-way median distance over rotations and mirroring."""
    angles = np.deg2rad(np.arange(0.0, 360.0, COARSE_STEP_DEG))
    cos, sin = np.cos(angles), np.sin(angles)
    scored: list[tuple[float, bool, float]] = []
    for mirrored in (False, True):
        q = _mirror(p_centred, mirrored)
        # Telemetry -> centre line: rotate telemetry by each angle, query the centre-line tree.
        qx = cos[:, None] * q[None, :, 0] - sin[:, None] * q[None, :, 1]
        qy = sin[:, None] * q[None, :, 0] + cos[:, None] * q[None, :, 1]
        fwd, _ = cl.tree.query(np.column_stack([qx.ravel(), qy.ravel()]) + cl.centroid)
        fwd = np.median(fwd.reshape(len(angles), -1), axis=1)
        # Centre line -> telemetry: rotate the centre line by -angle into the telemetry frame.
        c = c_centred
        cxr = cos[:, None] * c[None, :, 0] + sin[:, None] * c[None, :, 1]
        cyr = -sin[:, None] * c[None, :, 0] + cos[:, None] * c[None, :, 1]
        back, _ = cKDTree(q).query(np.column_stack([cxr.ravel(), cyr.ravel()]))
        back = np.median(back.reshape(len(angles), -1), axis=1)
        score = fwd + back
        is_min = (score <= np.roll(score, 1)) & (score <= np.roll(score, -1))
        scored.extend((float(score[i]), mirrored, float(angles[i])) for i in np.flatnonzero(is_min))
    scored.sort(key=lambda item: item[0])
    return [(m, th) for _, m, th in scored[:_N_CANDIDATES]]


def _two_way_score(aligned: np.ndarray, c_sub: np.ndarray, cl: _Centreline) -> float:
    fwd = cl.project(aligned, k=4).dist
    back, _ = cKDTree(aligned).query(c_sub)
    return float(np.median(fwd) + np.median(back))


def _procrustes(
    q: np.ndarray, f: np.ndarray, w: np.ndarray, fit_scale: bool, scale: float
) -> tuple[float, float, np.ndarray]:
    """Weighted closed-form 2D fit of ``f ~ s * R(theta) @ q + t``."""
    wsum = w.sum()
    qb = (w[:, None] * q).sum(axis=0) / wsum
    fb = (w[:, None] * f).sum(axis=0) / wsum
    qc, fc = q - qb, f - fb
    sxx = np.sum(w * (qc[:, 0] * fc[:, 0] + qc[:, 1] * fc[:, 1]))
    sxy = np.sum(w * (qc[:, 0] * fc[:, 1] - qc[:, 1] * fc[:, 0]))
    theta = float(np.arctan2(sxy, sxx))
    if fit_scale:
        var = np.sum(w * (qc[:, 0] ** 2 + qc[:, 1] ** 2))
        s_opt = np.hypot(sxx, sxy) / var if var > 0 else 1.0
        scale = float(np.clip(s_opt, 1.0 - SCALE_LIMIT, 1.0 + SCALE_LIMIT))
    t = fb - scale * (_rot(theta) @ qb)
    return theta, scale, t


def _icp(
    q: np.ndarray,
    w: np.ndarray,
    cl: _Centreline,
    theta: float,
    scale: float,
    t: np.ndarray,
    fit_scale: bool,
    *,
    max_iter: int,
    tol_m: float,
) -> tuple[float, float, np.ndarray]:
    """Trimmed point-to-segment ICP. ``q`` is already mirrored. Returns (theta, scale, t)."""
    t = np.asarray(t, dtype=np.float64)
    if not fit_scale:
        scale = 1.0
    x = scale * q @ _rot(theta).T + t
    for _ in range(max_iter):
        proj = cl.project(x, k=3)
        med = float(np.median(proj.dist))
        keep = proj.dist <= max(_TRIM_FACTOR * med, _TRIM_FLOOR_M)
        if np.count_nonzero(keep) < 3:
            break
        theta, scale, t = _procrustes(q, proj.foot, w * keep, fit_scale, scale)
        x_new = scale * q @ _rot(theta).T + t
        moved = float(np.max(np.hypot(*(x_new - x).T)))
        x = x_new
        if moved < tol_m:
            break
    return theta, scale, t


def _corridor_polish(
    q: np.ndarray,
    cl: _Centreline,
    theta: float,
    scale: float,
    t: np.ndarray,
    fit_scale: bool,
) -> tuple[float, float, np.ndarray]:
    """Minimise soft edge excess plus a weak centre-line term, starting from an ICP fit.

    Per sample, with margin ``m = min(w_left - n, n + w_right)`` (negative outside), the
    residuals are ``tau * softplus(-m / tau)`` and ``sqrt(CENTRE_WEIGHT) * dist``, under a Huber
    loss. Parameters are in metres about the path centroid so the problem is well conditioned.
    """
    q_bar = q.mean(axis=0)
    qc = q - q_bar
    r_g = float(np.sqrt(np.mean(np.sum(qc**2, axis=1)))) or 1.0
    c0 = scale * _rot(theta) @ q_bar + t  # where the centroid lands
    tau = _CORRIDOR_SOFTNESS_M
    sqrt_eps = np.sqrt(_CENTRE_WEIGHT)

    def unpack(v: np.ndarray) -> tuple[float, float, np.ndarray]:
        th = theta + v[0] / r_g
        sc = scale * (1.0 + v[3] / r_g) if fit_scale else scale
        return th, sc, c0 + v[1:3]

    def residuals(v: np.ndarray) -> np.ndarray:
        th, sc, c = unpack(v)
        proj = cl.project(sc * qc @ _rot(th).T + c, k=3)
        w_l, w_r = cl.widths(proj.seg, proj.u)
        margin = np.minimum(w_l - proj.n, proj.n + w_r)
        return np.concatenate([tau * np.logaddexp(0.0, -margin / tau), sqrt_eps * proj.dist])

    n_par = 4 if fit_scale else 3
    lo = np.full(n_par, -np.inf)
    hi = np.full(n_par, np.inf)
    if fit_scale:
        lo[3] = r_g * ((1.0 - SCALE_LIMIT) / scale - 1.0)
        hi[3] = r_g * ((1.0 + SCALE_LIMIT) / scale - 1.0)
    result = least_squares(
        residuals,
        np.zeros(n_par),
        bounds=(lo, hi),
        loss="huber",
        f_scale=_POLISH_HUBER_M,
        diff_step=1e-3,  # parameters start at 0, so this is a 1 mm step
        x_scale=1.0,
        max_nfev=200,
    )
    th, sc, c = unpack(result.x)
    return th, sc, c - sc * _rot(th) @ q_bar
