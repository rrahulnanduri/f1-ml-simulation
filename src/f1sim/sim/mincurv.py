"""A minimum-curvature racing line for circuits without a TUM raceline (D37).

The line is the centre line moved sideways by ``alpha_i`` along each point's normal,
``p_i = c_i + alpha_i * n_i``, with ``alpha`` chosen to minimise the summed squared second
difference of the points (``sum |p_{i+1} - 2 p_i + p_{i-1}|^2``). On points evenly spaced
along the centre line that is the summed squared curvature up to a constant, and it is a
linear least-squares problem in ``alpha`` with box bounds: the line keeps
``CAR_HALF_WIDTH_M + INSIDE_MARGIN_M + EDGE_MARGIN_M`` inside each edge of the track it is
given (the limits, for the track the cars drive on). It is solved once, then
``ITERATIONS`` passes re-weight each row by the current line's spacing so the cost follows
the line's own curvature rather than the centre line's spacing (at Silverstone that is
91.6 s -> 87.2 s for the scripted lap). Each solve is ``scipy.optimize.lsq_linear`` on the
sparse system; the whole line takes under a second. (ADMM and a primal-dual active-set
method were tried and were slower: the fourth-difference normal matrix is badly
conditioned.)

The same formulation (without the re-weighting) is the core of TUM's
``global_racetrajectory_optimization`` min-curvature line, which their published racelines
start from. Minimum curvature is not minimum time: it ignores where the car is slow, so it
cuts long, fast corners a little too much. At Silverstone the scripted lap on it is
87.15 s, against 86.43 s on the TUM line and 108.96 s on the centre line.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from f1sim.sim.scripted import CAR_HALF_WIDTH_M, INSIDE_MARGIN_M
from f1sim.track.track import Track

STEP_M = 5.0
"""Spacing of the centre-line points the line is solved on, m (3 m: 3.5x the time, same lap)."""
EDGE_MARGIN_M = 0.2
"""Room beyond the scripted driver's own inside margin, so ``keep_inside`` has nothing to do."""
ITERATIONS = 2
"""Re-weighted passes after the first solve."""
SOLVER_TOL = 1e-6
"""``lsq_linear`` tolerance: 1e-10 (its default) takes 6x longer for no faster lap."""


def _second_difference(m: int) -> sparse.csr_matrix:
    """Cyclic ``(m, m)`` second-difference operator: row i is ``x[i-1] - 2 x[i] + x[i+1]``."""
    i = np.arange(m)
    rows = np.concatenate([i, i, i])
    cols = np.concatenate([(i - 1) % m, i, (i + 1) % m])
    vals = np.concatenate([np.ones(m), -2.0 * np.ones(m), np.ones(m)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, m))


def min_curvature_line(track: Track, step_m: float = STEP_M) -> tuple[np.ndarray, np.ndarray]:
    """The closed minimum-curvature line ``(x, y)`` inside ``track``; see the module doc.

    Point 0 is at ``s = 0``, and the line runs in the track's direction. Where the track is
    narrower than twice the margin the bounds meet on the centre line.
    """
    length = track.length
    m = max(int(round(length / step_m)), 16)
    s = np.arange(m) * (length / m)
    c = track.point_at(s)
    h = track.heading_at(s)
    nx, ny = -np.sin(h), np.cos(h)
    w_left, w_right = track.widths_at(s)
    keep = CAR_HALF_WIDTH_M + INSIDE_MARGIN_M + EDGE_MARGIN_M
    hi = np.maximum(w_left - keep, 0.0)
    lo = -np.maximum(w_right - keep, 0.0)
    lo = np.minimum(lo, hi - 1e-9)  # lsq_linear needs lo < hi

    d2 = _second_difference(m)
    a_x = d2 @ sparse.diags(nx)
    a_y = d2 @ sparse.diags(ny)
    b_x = -(d2 @ c[:, 0])
    b_y = -(d2 @ c[:, 1])
    weights = np.ones(m)
    alpha = np.zeros(m)
    for _ in range(1 + ITERATIONS):
        w = sparse.diags(weights)
        a = sparse.vstack([w @ a_x, w @ a_y]).tocsr()
        b = np.concatenate([weights * b_x, weights * b_y])
        alpha = lsq_linear(a, b, bounds=(lo, hi), method="trf", lsmr_tol="auto", tol=SOLVER_TOL).x
        px, py = c[:, 0] + alpha * nx, c[:, 1] + alpha * ny
        # Curvature ~ second difference / spacing^2: weight each row by the line's own spacing
        seg = np.hypot(np.diff(np.append(px, px[0])), np.diff(np.append(py, py[0])))
        spacing = 0.5 * (seg + np.roll(seg, 1))
        weights = (length / m / np.maximum(spacing, 1e-6)) ** 2
    return c[:, 0] + alpha * nx, c[:, 1] + alpha * ny
