"""Scripted reference driver: the physics calibration tool (docs/m2-interfaces.md section G).

:func:`speed_profile` computes the fastest speed the car model allows at each point of a path
(quasi-steady-state lap simulation), and :func:`drive_reference_lap` actually drives
:func:`f1sim.sim.car_model.step` along that path with a pure-pursuit steering controller and a
model-inverting speed controller, so the lap time reflects the real integrator.

The limits mirror ``car_model.step`` exactly:

* friction circle ``a_max(v) = grip_limit(v)``, with the lateral demand ``v^2 * kappa`` served
  first and the longitudinal tyre force trimmed to ``sqrt(a_max^2 - a_lat^2)``;
* drive ``min(power / (mass * max(v, V_FLOOR)), max_drive_g * g)``;
* brake ``min(max_brake_g * g, a_max)``;
* drag ``drag_coeff * v^2`` on top of the tyre force (it helps braking, costs drive).

The steady-state cornering speed holds speed against drag inside the friction circle:
``(v^2 kappa)^2 + (drag * v^2)^2 <= a_max(v)^2``. Because downforce grows with ``v^2`` as fast
as the lateral demand does, bends gentler than about 150 m radius (default parameters) have no
grip limit at all; there the speed is capped at the power/drag top speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

from f1sim.sim.car_model import V_FLOOR, CarParams, CarState, G, step
from f1sim.track.track import Track, TrackData

PATH_SPACING_M = 2.0
"""Paths sparser than ``RESAMPLE_ABOVE_M`` (e.g. TUM racelines, ~5 m) are resampled to this."""
RESAMPLE_ABOVE_M = 2.5
CURVATURE_SIGMA_M = 2.0
"""Light Gaussian smoothing of the discrete path curvature (standard deviation, metres)."""

CAR_HALF_WIDTH_M = 1.0
"""Off track when ``|n| > w_side - CAR_HALF_WIDTH_M`` (revamp-plan section 6)."""
LAP_MIN_FRACTION = 0.98
INSIDE_MARGIN_M = 0.3
"""Room a raceline keeps inside the off-track limit, for the controller's tracking error."""
# keep_inside spreads each correction over +-INSIDE_SPREAD_M and smooths it with this sigma;
# wide, because a narrow bump of 0.5 m adds ~0.5 / sigma^2 of curvature.
INSIDE_SPREAD_M = 40.0
INSIDE_SIGMA_M = 30.0

# Pure pursuit: lookahead distance along the path, metres, as a function of speed.
LOOKAHEAD_BASE_M = 3.0
LOOKAHEAD_TIME_S = 0.12
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 25.0
# Speed controller: target the profile speed this many steps ahead of the car.
SPEED_LEAD_STEPS = 1.5
DRIVE_CURVATURE_MARGIN = 0.01
"""The driver follows a profile computed with 1% more curvature than the path. Braking while
cornering at the exact friction-circle limit is unstable (a car 0.1 m/s over the profile has
almost no longitudinal budget left, falls further behind and slides); 1% costs ~0.4% lap time."""

_PROFILE_MAX_ITERS = 50
_PROFILE_TOL = 1e-9


@dataclass(frozen=True)
class ReferenceLap:
    """Result of :func:`drive_reference_lap`; the trace arrays have one entry per physics step
    (plus the initial state at ``t = 0``)."""

    lap_time_s: float  # nan if the car left the track or didn't finish
    completed: bool
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    s: np.ndarray
    n: np.ndarray
    max_abs_n_excess_m: float  # worst |n| beyond (w_side - 1.0); <= 0 means always on track


# ------------------------------------------------------------------------------- limits


def top_speed(params: CarParams) -> float:
    """Speed where full drive equals drag: ``min(cbrt(P / (m c)), sqrt(max_drive_g g / c))``."""
    c = params.drag_coeff
    if c <= 0.0:
        return math.inf
    v_power = (params.power_w / (params.mass_kg * c)) ** (1.0 / 3.0)
    v_traction = math.sqrt(params.max_drive_g * G / c)
    return min(v_power, v_traction)


def lateral_speed_limit(kappa, params: CarParams) -> np.ndarray:
    """Highest steady speed on curvature ``kappa`` while holding speed against drag.

    Solves ``(v^2 k)^2 + (c v^2)^2 <= (g (base + aero v^2))^2`` for ``v^2``:
    ``v^2 = g base / (sqrt(k^2 + c^2) - g aero)``, unlimited when the denominator is <= 0.
    The result is capped at :func:`top_speed`.
    """
    k = np.abs(np.asarray(kappa, dtype=np.float64))
    denom = np.hypot(k, params.drag_coeff) - G * params.grip_aero_coeff
    with np.errstate(divide="ignore"):
        v2 = np.where(denom > 0.0, G * params.grip_base_g / np.where(denom > 0, denom, 1.0), np.inf)
    return np.minimum(np.sqrt(v2), top_speed(params))


def _long_limits(v: float, k: float, p: CarParams) -> tuple[float, float]:
    """``(max net acceleration, max net deceleration)`` at speed ``v`` on curvature ``k``.

    Exactly ``car_model.step`` at full throttle / full brake with lateral demand ``v^2 k``:
    the tyre force is trimmed to the friction-circle budget and drag is added on top.
    """
    a_max = G * (p.grip_base_g + p.grip_aero_coeff * v * v)
    a_lat = v * v * k
    budget = math.sqrt(max(a_max * a_max - a_lat * a_lat, 0.0))
    drive = min(p.power_w / (p.mass_kg * max(v, V_FLOOR)), p.max_drive_g * G)
    brake = min(p.max_brake_g * G, a_max)
    drag = p.drag_coeff * v * v
    return min(drive, budget) - drag, min(brake, budget) + drag


# --------------------------------------------------------------------------------- path


@dataclass(frozen=True)
class _Path:
    """A path evaluated at (possibly resampled) points, with spline arc length and curvature."""

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray  # arc length of each point from point 0
    ds: np.ndarray  # arc length to the next point (closed: includes the closing segment)
    heading: np.ndarray
    curvature: np.ndarray  # smoothed, > 0 = turning left
    length: float  # total arc length (closed: full loop)
    s_input: np.ndarray  # arc length of each input point (to map results back)
    closed: bool


def _clean_points(x, y, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("path_x and path_y must have the same length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("path contains non-finite values")
    if closed and x.size > 1 and math.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-9:
        x, y = x[:-1], y[:-1]
    if x.size < 4:
        raise ValueError(f"path needs at least 4 points, got {x.size}")
    return x, y


def _prepare_path(path_x, path_y, *, closed: bool) -> _Path:
    """Fit a cubic spline through the points (periodic when closed) and evaluate it at the
    input points, or at ~PATH_SPACING_M when the input is sparser than RESAMPLE_ABOVE_M."""
    x, y = _clean_points(path_x, path_y, closed)
    pts = np.column_stack([x, y])
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    if np.any(seg <= 0.0):
        raise ValueError("path has repeated consecutive points")
    u = np.concatenate([[0.0], np.cumsum(seg)])
    spline = CubicSpline(u, pts, bc_type="periodic" if closed else "natural")

    # Arc length of the spline as a function of its (chord-length) parameter.
    ud = np.linspace(0.0, u[-1], 20 * seg.size + 1)
    pd = spline(ud)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pd[:, 0]), np.diff(pd[:, 1])))])
    length = float(arc[-1])
    u_input = u[:-1] if closed else u
    s_input = np.interp(u_input, ud, arc)

    if seg.max() > RESAMPLE_ABOVE_M:
        m = max(int(math.ceil(length / PATH_SPACING_M)), 4)
        s_pts = np.arange(m) * (length / m) if closed else np.linspace(0.0, length, m + 1)
        u_pts = np.interp(s_pts, arc, ud)
    else:
        s_pts, u_pts = s_input, u_input

    p = spline(u_pts)
    d1 = spline(u_pts, 1)
    d2 = spline(u_pts, 2)
    speed = np.hypot(d1[:, 0], d1[:, 1])
    heading = np.arctan2(d1[:, 1], d1[:, 0])
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3
    ds = np.diff(np.append(s_pts, length)) if closed else np.diff(s_pts)
    sigma = CURVATURE_SIGMA_M / float(np.mean(ds))
    if sigma > 0.1:
        kappa = gaussian_filter1d(kappa, sigma, mode="wrap" if closed else "nearest")
    return _Path(
        x=p[:, 0].copy(),
        y=p[:, 1].copy(),
        s=s_pts,
        ds=ds,
        heading=heading,
        curvature=kappa,
        length=length,
        s_input=s_input,
        closed=closed,
    )


# -------------------------------------------------------------------------- speed profile


def _backward_update(v: np.ndarray, i: int, j: int, ds: float, kap, p: CarParams) -> None:
    """Braking: lower ``v[i]`` so the car can brake from it to ``v[j]`` over ``ds``."""
    vj = v[j]
    d_j = _long_limits(vj, kap[j], p)[1]
    cand = math.sqrt(vj * vj + 2.0 * ds * d_j)
    d_i = _long_limits(cand, kap[i], p)[1]
    if d_i < d_j:
        cand = math.sqrt(vj * vj + 2.0 * ds * d_i)
    if cand < v[i]:
        v[i] = cand


def _forward_update(v: np.ndarray, i: int, j: int, ds: float, kap, p: CarParams) -> None:
    """Traction/power: lower ``v[j]`` to what the car can reach from ``v[i]`` over ``ds``."""
    vi = v[i]
    a_i = _long_limits(vi, kap[i], p)[0]
    cand = math.sqrt(max(vi * vi + 2.0 * ds * a_i, 0.0))
    a_j = _long_limits(cand, kap[j], p)[0]
    if a_j < a_i:
        cand = math.sqrt(max(vi * vi + 2.0 * ds * a_j, 0.0))
    if cand < v[j]:
        v[j] = cand


def _profile_on_path(path: _Path, params: CarParams, curvature_margin: float = 0.0) -> np.ndarray:
    """Speed profile at the path's own points (see :func:`speed_profile`).

    ``curvature_margin`` inflates the curvature by that fraction (0 = exact limits)."""
    kap = np.abs(path.curvature) * (1.0 + curvature_margin)
    v = lateral_speed_limit(kap, params)
    finite_cap = 1e3  # only reached with zero drag and no grip limit anywhere
    v = np.minimum(v, finite_cap)
    kap_l = kap.tolist()
    ds = path.ds.tolist()
    m = v.size
    vv = v.tolist()
    if not path.closed:
        for i in range(m - 2, -1, -1):
            _backward_update(vv, i, i + 1, ds[i], kap_l, params)
        for i in range(m - 1):
            _forward_update(vv, i, i + 1, ds[i], kap_l, params)
        return np.asarray(vv)

    # Closed loop: start both passes at the slowest point, repeat until periodic.
    for _ in range(_PROFILE_MAX_ITERS):
        old = list(vv)
        start = int(np.argmin(vv))
        for k in range(m):
            i = (start - 1 - k) % m
            _backward_update(vv, i, (i + 1) % m, ds[i], kap_l, params)
        for k in range(m):
            i = (start + k) % m
            _forward_update(vv, i, (i + 1) % m, ds[i], kap_l, params)
        if max(abs(a - b) for a, b in zip(vv, old, strict=True)) < _PROFILE_TOL:
            break
    return np.asarray(vv)


def _interp_periodic(s_query, s_pts: np.ndarray, values: np.ndarray, length: float):
    s_ext = np.append(s_pts, length)
    v_ext = np.append(values, values[0])
    return np.interp(np.mod(s_query, length), s_ext, v_ext)


def speed_profile(path_x, path_y, params: CarParams, *, closed: bool = True) -> np.ndarray:
    """Max speed (m/s) at each path point.

    The path is fitted with a cubic spline (periodic when ``closed``) and, when sparser than
    2.5 m, resampled to ~2 m; curvature comes from the spline and is lightly smoothed. The
    lateral limit per point is :func:`lateral_speed_limit` (friction circle including the drag
    the tyres must hold, capped at the top speed). Then a backward pass applies the braking
    limit and a forward pass the drive/power limit, both exactly as in ``car_model.step``
    (tyre force trimmed to the friction-circle budget left by the lateral demand, drag on top).
    A closed loop repeats both passes until the profile is periodic. Results computed on a
    resampled path are interpolated back to the input points.
    """
    path = _prepare_path(path_x, path_y, closed=closed)
    v = _profile_on_path(path, params)
    if path.s.size == path.s_input.size and np.array_equal(path.s, path.s_input):
        return v
    if closed:
        return _interp_periodic(path.s_input, path.s, v, path.length)
    return np.interp(path.s_input, path.s, v)


def profile_lap_time(path_x, path_y, v, *, closed: bool = True) -> float:
    """``sum(ds / v)`` over the path (trapezoidal in ``1/v`` per segment, i.e. constant
    acceleration between points); the closed loop includes the closing segment."""
    x, y = _clean_points(path_x, path_y, closed)
    v = np.asarray(v, dtype=np.float64)
    if closed:
        x, y, v = np.append(x, x[0]), np.append(y, y[0]), np.append(v, v[0])
    ds = np.hypot(np.diff(x), np.diff(y))
    return float(np.sum(2.0 * ds / (v[:-1] + v[1:])))


# ------------------------------------------------------------------------------ raceline


def _wrap_ds(ds: np.ndarray, length: float) -> np.ndarray:
    return (ds + 0.5 * length) % length - 0.5 * length


def orient_raceline(track: Track, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Raceline in the track's direction of travel, starting at the point nearest after s=0.

    The points are projected onto the track; if the mean wrap-aware ``ds`` is negative the
    raceline runs backwards and is reversed. It is then rolled so that point 0 has the
    smallest ``s``.
    """
    x, y = _clean_points(x, y, closed=True)
    s, _, _ = track.project(x, y)
    if np.mean(_wrap_ds(np.diff(np.append(s, s[0])), track.length)) < 0.0:
        x, y, s = x[::-1], y[::-1], s[::-1]
    k = int(np.argmin(s))
    return np.roll(x, -k), np.roll(y, -k)


def keep_inside(track: Track, x, y, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Pull a closed line in, smoothly, wherever ``|n| > w_side - CAR_HALF_WIDTH_M - margin_m``.

    The lateral correction is spread over ``+-INSIDE_SPREAD_M`` (max filter) and
    Gaussian-smoothed (``INSIDE_SIGMA_M``), repeated until the line is inside, so it adds
    little curvature. If no point is outside, the line is returned unchanged.
    """
    x, y = _clean_points(x, y, closed=True)
    s, n, _ = track.project(x, y)
    w_left, w_right = track.widths_at(s)
    hi = np.maximum(w_left - CAR_HALF_WIDTH_M - margin_m, 0.0)
    lo = -np.maximum(w_right - CAR_HALF_WIDTH_M - margin_m, 0.0)
    spacing = float(np.mean(np.hypot(np.diff(x), np.diff(y))))
    size = 2 * max(int(round(INSIDE_SPREAD_M / spacing)), 1) + 1
    sigma = INSIDE_SIGMA_M / spacing
    total = np.zeros_like(n)
    for _ in range(50):
        cur = n + total
        corr = np.clip(cur, lo, hi) - cur
        if np.max(np.abs(corr)) < 1e-3:
            break
        spread = maximum_filter1d(np.maximum(corr, 0.0), size, mode="wrap") + minimum_filter1d(
            np.minimum(corr, 0.0), size, mode="wrap"
        )
        total += gaussian_filter1d(spread, sigma, mode="wrap")
    if not np.any(total):
        return x, y
    shift = np.clip(n + total, lo, hi) - n
    h = track.heading_at(s)
    return x - np.sin(h) * shift, y + np.cos(h) * shift


def _path_as_track(path: _Path) -> Track:
    """Wrap a closed path in a :class:`Track` for its projection and interpolation."""
    ones = np.ones_like(path.x)
    data = TrackData(
        id="scripted-path",
        name="scripted path",
        x=path.x,
        y=path.y,
        s=path.s - path.s[0],
        w_left=ones,
        w_right=ones,
        heading=path.heading,
        curvature=path.curvature,
        length_m=path.length,
        resolution_m=float(np.mean(path.ds)),
    )
    return Track(data)


def _start_offset(track: Track, path: _Path) -> float:
    """Path arc length where the path crosses the track's s=0 line going forward."""
    s, _, _ = track.project(path.x, path.y)
    s_next = np.roll(s, -1)
    crossing = np.nonzero(s_next < s - 0.5 * track.length)[0]
    if crossing.size == 0:
        return 0.0
    i = int(crossing[np.argmin(np.abs(s[crossing] - track.length))])
    before = track.length - s[i]
    frac = before / (before + s_next[i])
    return float((path.s[i] + frac * path.ds[i]) % path.length)


# -------------------------------------------------------------------------------- driver


def _reference_path(track: Track, raceline_xy, inside_margin_m: float | None) -> _Path:
    """The path the reference driver follows (see :func:`drive_reference_lap`)."""
    if raceline_xy is None:
        return _prepare_path(track.data.x, track.data.y, closed=True)
    rx, ry = raceline_xy
    px, py = orient_raceline(track, rx, ry)
    path = _prepare_path(px, py, closed=True)
    if inside_margin_m is None:
        return path
    px, py = keep_inside(track, path.x, path.y, inside_margin_m)
    return _prepare_path(px, py, closed=True)


def drive_reference_lap(
    track: Track,
    params: CarParams,
    *,
    raceline_xy=None,
    dt: float = 1.0 / 60.0,
    flying: bool = True,
    inside_margin_m: float | None = INSIDE_MARGIN_M,
) -> ReferenceLap:
    """Drive ``car_model.step`` around ``track`` along ``raceline_xy`` (centre line if None).

    Steering is pure pursuit on a lookahead point ``LOOKAHEAD_*`` metres ahead along the path
    (the chord form ``k = 2 y_local / d^2``, exact on a circle), converted to
    ``steer_cmd = clip(atan(L k) / max_steer, -1, 1)``. The speed controller inverts the car
    model's longitudinal law towards the :func:`speed_profile` speed (with
    ``DRIVE_CURVATURE_MARGIN``) sampled ``SPEED_LEAD_STEPS`` steps ahead, so braking starts on
    time. ``s`` and ``n`` come from ``Track.project`` with hints, and ``max_abs_n_excess_m``
    is the worst ``|n| - (w_side - 1.0)``. ``flying=True`` starts on the s=0 line at the
    profile speed; ``flying=False`` starts there from rest. The lap ends at the first forward
    crossing of s=0 after ``LAP_MIN_FRACTION`` of the track length (time interpolated within
    the step). ``lap_time_s`` is nan unless the lap completed and the car stayed on track.

    ``inside_margin_m`` (not in the M2 contract; ``None`` disables it): a raceline is first
    resampled and then pulled in with :func:`keep_inside` so that it keeps this much room
    inside the off-track limit for tracking error. TUM racelines run closer than
    ``CAR_HALF_WIDTH_M`` to the built track's edges, so without this the reference car is
    off track by the section-E rule even when it follows the line exactly.
    """
    if not dt > 0.0:
        raise ValueError(f"dt must be > 0, got {dt!r}")
    path = _reference_path(track, raceline_xy, inside_margin_m)
    v_prof = _profile_on_path(path, params, curvature_margin=DRIVE_CURVATURE_MARGIN)
    ptrack = _path_as_track(path)
    p_len = path.length

    def profile_at(sp):
        return float(_interp_periodic(sp, path.s, v_prof, p_len))

    sp0 = _start_offset(track, path)
    x0, y0 = ptrack.point_at(sp0)
    v0 = profile_at(sp0) if flying else 0.0
    state = CarState.at(1, float(x0), float(y0), float(ptrack.heading_at(sp0)), v0)
    k0 = float(ptrack.curvature_at(sp0))
    delta0 = math.atan(params.wheelbase_m * k0)
    state.delta[:] = max(-params.max_steer_rad, min(params.max_steer_rad, delta0))

    s_t, n_t, idx_t = track.project(state.x, state.y)
    _, _, idx_p = ptrack.project(state.x, state.y)

    def n_excess(s: float, n: float) -> float:
        w_left, w_right = track.widths_at(s)
        w_side = float(w_left) if n >= 0.0 else float(w_right)
        return abs(n) - (w_side - CAR_HALF_WIDTH_M)

    profile_time = profile_lap_time(path.x, path.y, v_prof)
    t_cap = 2.0 * profile_time + 60.0
    max_steps = int(math.ceil(t_cap / dt))
    drag = params.drag_coeff
    a_brake_cap = params.max_brake_g * G

    ts, xs, ys, vs = [0.0], [float(x0)], [float(y0)], [v0]
    ss, ns = [float(s_t[0])], [float(n_t[0])]
    worst = n_excess(ss[0], ns[0])
    progress = 0.0
    lap_time = math.nan
    completed = False
    t = 0.0
    for _ in range(max_steps):
        v = float(state.v[0])
        x, y, yaw = float(state.x[0]), float(state.y[0]), float(state.yaw[0])

        # Pure pursuit steering.
        sp, e_path, idx_p = ptrack.project(state.x, state.y, hint=idx_p)
        sp = float(sp[0])
        if abs(float(e_path[0])) > 30.0:
            break  # lost the path: give up
        ld = min(max(LOOKAHEAD_BASE_M + LOOKAHEAD_TIME_S * v, LOOKAHEAD_MIN_M), LOOKAHEAD_MAX_M)
        tx, ty = ptrack.point_at(sp + ld)
        dx, dy = float(tx) - x, float(ty) - y
        y_loc = -math.sin(yaw) * dx + math.cos(yaw) * dy
        k_cmd = 2.0 * y_loc / max(dx * dx + dy * dy, 1e-6)
        steer = max(-1.0, min(1.0, math.atan(params.wheelbase_m * k_cmd) / params.max_steer_rad))

        # Speed: invert the longitudinal law towards the profile speed a little ahead.
        v_tgt = profile_at(sp + SPEED_LEAD_STEPS * max(v, V_FLOOR) * dt)
        tyre = (v_tgt - v) / dt + drag * v * v
        if tyre >= 0.0:
            a_drive = min(
                params.power_w / (params.mass_kg * max(v, V_FLOOR)), params.max_drive_g * G
            )
            pedal = tyre / a_drive
        else:
            a_max = G * (params.grip_base_g + params.grip_aero_coeff * v * v)
            pedal = tyre / min(a_brake_cap, a_max)
        pedal = max(-1.0, min(1.0, pedal))

        step(state, steer, pedal, params, dt)
        t += dt

        s_prev = ss[-1]
        s_new, n_new, idx_t = track.project(state.x, state.y, hint=idx_t)
        s_new, n_new = float(s_new[0]), float(n_new[0])
        d_s = float(_wrap_ds(np.array(s_new - s_prev), track.length))
        progress += d_s
        worst = max(worst, n_excess(s_new, n_new))

        ts.append(t)
        xs.append(float(state.x[0]))
        ys.append(float(state.y[0]))
        vs.append(float(state.v[0]))
        ss.append(s_new)
        ns.append(n_new)

        wrapped = d_s > 0.0 and s_new < s_prev
        if wrapped and progress >= LAP_MIN_FRACTION * track.length:
            before = track.length - s_prev
            frac = before / (before + s_new) if before + s_new > 0.0 else 1.0
            lap_time = t - dt + frac * dt
            completed = True
            break

    on_track = worst <= 0.0
    return ReferenceLap(
        lap_time_s=lap_time if (completed and on_track) else math.nan,
        completed=completed,
        t=np.asarray(ts),
        x=np.asarray(xs),
        y=np.asarray(ys),
        v=np.asarray(vs),
        s=np.asarray(ss),
        n=np.asarray(ns),
        max_abs_n_excess_m=float(worst),
    )
