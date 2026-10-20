"""Scripted reference driver: speed profile limits and driven laps on synthetic tracks."""

import math

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data.track_builder import build_track
from f1sim.sim.car_model import CarParams, G
from f1sim.sim.scripted import (
    ReferenceLap,
    drive_reference_lap,
    keep_inside,
    lateral_speed_limit,
    orient_raceline,
    speed_profile,
    top_speed,
)
from f1sim.track.track import Track

P = CarParams()
STRAIGHT = 400.0
RADIUS = 40.0
HALF_WIDTH = 7.0


@pytest.fixture(scope="module")
def track() -> Track:
    x, y = stadium(STRAIGHT, RADIUS, 2.0)
    n = x.size
    w = np.full(n, HALF_WIDTH)
    return Track(build_track(x, y, w, w, track_id="stadium", name="Stadium"))


@pytest.fixture(scope="module")
def profile(track):
    return speed_profile(track.data.x, track.data.y, P)


@pytest.fixture(scope="module")
def flying_lap(track) -> ReferenceLap:
    return drive_reference_lap(track, P)


def closed_ds(x, y):
    return np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))


def integrate_profile(x, y, v) -> float:
    """sum(ds / v) around the loop, independent of src (trapezoid in 1/v per segment)."""
    ds = closed_ds(x, y)
    return float(np.sum(2.0 * ds / (v + np.roll(v, -1))))


# ------------------------------------------------------------------------ speed profile


def test_lateral_limit_matches_closed_form():
    k = 1.0 / 40.0
    expected = math.sqrt(G * P.grip_base_g / (math.hypot(k, P.drag_coeff) - G * P.grip_aero_coeff))
    assert lateral_speed_limit(k, P) == pytest.approx(expected, rel=1e-12)
    # Above ~150 m radius downforce outgrows the lateral demand: only the top speed caps it.
    assert lateral_speed_limit(1.0 / 160.0, P) == pytest.approx(top_speed(P))
    assert lateral_speed_limit(0.0, P) == pytest.approx(top_speed(P))
    assert top_speed(P) == pytest.approx((P.power_w / (P.mass_kg * P.drag_coeff)) ** (1 / 3))


def test_profile_is_periodic(track, profile):
    x, y = track.data.x, track.data.y
    k = 137
    rolled = speed_profile(np.roll(x, -k), np.roll(y, -k), P)
    np.testing.assert_allclose(rolled, np.roll(profile, -k), rtol=1e-6, atol=1e-6)
    # A repeated closing point is accepted and changes nothing.
    np.testing.assert_allclose(
        speed_profile(np.append(x, x[0]), np.append(y, y[0]), P), profile, atol=1e-9
    )


def test_profile_never_exceeds_lateral_limit(track, profile):
    # Independent curvature: the stadium's analytic 1/R on the bends, 0 on the straights. The
    # curvature step at the joins is smoothed (by build_track and lightly by speed_profile),
    # so the first 4 m of each bend get a looser bound.
    x = np.abs(track.data.x)
    k_true = np.where(x > STRAIGHT / 2, 1.0 / RADIUS, 0.0)
    tol = np.where((x > STRAIGHT / 2) & (x < STRAIGHT / 2 + 4.0), 1.03, 1.01)
    assert np.all(profile <= lateral_speed_limit(k_true, P) * tol)
    # ... and the bends are taken at (close to) that limit, not conservatively.
    v_bend = lateral_speed_limit(1.0 / RADIUS, P)
    mid_bend = np.abs(track.data.x) > STRAIGHT / 2 + 10.0
    np.testing.assert_allclose(profile[mid_bend], v_bend, rtol=0.02)
    assert profile.max() < top_speed(P)  # the straights are too short to reach it


def test_profile_respects_longitudinal_limits(track, profile):
    """Between consecutive points, the speed change fits car_model's best-case limits."""
    v = profile
    v_next = np.roll(v, -1)
    ds = closed_ds(track.data.x, track.data.y)
    accel = (v_next**2 - v**2) / (2.0 * ds)

    # Upper bounds from car_model with the whole friction circle available (no lateral demand).
    v_lo, v_hi = np.minimum(v, v_next), np.maximum(v, v_next)
    a_max_hi = G * (P.grip_base_g + P.grip_aero_coeff * v_hi**2)
    drive_lo = np.minimum(P.power_w / (P.mass_kg * np.maximum(v_lo, 1.0)), P.max_drive_g * G)
    max_accel = drive_lo - P.drag_coeff * v_lo**2
    max_decel = np.minimum(P.max_brake_g * G, a_max_hi) + P.drag_coeff * v_hi**2
    assert np.all(accel <= max_accel * 1.01 + 1e-6)
    assert np.all(-accel <= max_decel * 1.01 + 1e-6)
    # Both limits are actually used somewhere (full throttle and full braking on the straights).
    assert np.any(accel > 0.9 * max_accel)
    assert np.any(-accel > 0.9 * max_decel)


def test_open_path_profile():
    # A straight then a quarter circle: the car accelerates from the start's lateral limit and
    # brakes for the bend; the end isn't wrapped onto the start.
    s = np.arange(0.0, 300.0, 2.0)
    a = np.linspace(0.0, np.pi / 2, 40)[1:]
    x = np.concatenate([s, 300.0 + 30.0 * np.sin(a)])
    y = np.concatenate([np.zeros_like(s), 30.0 * (1.0 - np.cos(a))])
    v = speed_profile(x, y, P, closed=False)
    assert v.shape == x.shape
    assert v[0] == pytest.approx(top_speed(P), rel=1e-6)  # unconstrained start
    assert v[-5] == pytest.approx(lateral_speed_limit(1.0 / 30.0, P), rel=0.03)


# ------------------------------------------------------------------------------ driving


def test_flying_lap_completes_on_track_close_to_profile(track, profile, flying_lap):
    lap = flying_lap
    assert lap.completed
    assert lap.max_abs_n_excess_m <= 0.0
    assert math.isfinite(lap.lap_time_s)
    expected = integrate_profile(track.data.x, track.data.y, profile)
    assert lap.lap_time_s == pytest.approx(expected, rel=0.02)
    # Trace: starts on the s=0 line at the profile speed, ends just past it.
    assert lap.t[0] == 0.0 and lap.t.shape == lap.v.shape == lap.s.shape == lap.n.shape
    assert min(lap.s[0], track.length - lap.s[0]) < 0.5
    assert lap.v[0] == pytest.approx(profile[0], rel=0.01)  # driven with a 1% curvature margin
    assert lap.t[-1] - 1.0 / 60.0 <= lap.lap_time_s <= lap.t[-1]
    assert lap.s[-1] < 10.0
    assert np.abs(lap.n).max() < 0.5  # it follows the centre line closely
    assert lap.v.max() <= profile.max() * 1.01


def test_standing_start_is_slower(track, flying_lap):
    standing = drive_reference_lap(track, P, flying=False)
    assert standing.completed and standing.max_abs_n_excess_m <= 0.0
    assert standing.v[0] == 0.0
    assert standing.lap_time_s > flying_lap.lap_time_s + 0.5


def test_leaving_the_track_gives_nan_lap_time():
    # A raceline the car can't keep inside: with the inside margin disabled, it is driven as is.
    x, y = stadium(STRAIGHT, RADIUS + 6.5, 2.0)
    n = x.size
    xc, yc = stadium(STRAIGHT, RADIUS, 2.0)
    narrow = Track(
        build_track(xc, yc, np.full(xc.size, 7.0), np.full(xc.size, 7.0), track_id="s", name="s")
    )
    assert n > 4
    lap = drive_reference_lap(narrow, P, raceline_xy=(x, y), inside_margin_m=None)
    assert lap.completed
    assert lap.max_abs_n_excess_m > 0.0
    assert math.isnan(lap.lap_time_s)
    # With the default margin the line is pulled in and the lap counts.
    lap_in = drive_reference_lap(narrow, P, raceline_xy=(x, y))
    assert lap_in.completed and lap_in.max_abs_n_excess_m <= 0.0
    assert math.isfinite(lap_in.lap_time_s)


# ---------------------------------------------------------------------------- raceline


def test_raceline_orientation_reversed_and_rolled(track):
    # A line 2 m inside the centre line (a concentric stadium), 5 m spaced like TUM's.
    rx, ry = stadium(STRAIGHT, RADIUS - 2.0, 5.0)
    s_fwd, _, _ = track.project(rx, ry)
    k = 50
    bx, by = np.roll(rx[::-1], k), np.roll(ry[::-1], k)

    ox, oy = orient_raceline(track, bx, by)
    s, n, _ = track.project(ox, oy)
    ds = np.diff(s)
    ds = (ds + track.length / 2) % track.length - track.length / 2
    assert np.all(ds > 0.0)  # runs with the track
    assert s[0] == pytest.approx(s_fwd.min(), abs=1e-6)  # starts nearest after s=0
    np.testing.assert_allclose(np.abs(n), 2.0, atol=0.1)

    lap_ref = drive_reference_lap(track, P, raceline_xy=(rx, ry))
    lap_bad = drive_reference_lap(track, P, raceline_xy=(bx, by))
    for lap in (lap_ref, lap_bad):
        assert lap.completed and lap.max_abs_n_excess_m <= 0.0
    assert lap_bad.lap_time_s == pytest.approx(lap_ref.lap_time_s, abs=0.05)
    assert min(lap_bad.s[0], track.length - lap_bad.s[0]) < 0.5


def test_keep_inside_only_moves_offending_points(track):
    # Inside the corridor already: unchanged.
    rx, ry = stadium(STRAIGHT, RADIUS - 2.0, 2.0)
    qx, qy = keep_inside(track, rx, ry, 0.3)
    np.testing.assert_allclose(qx, rx, atol=1e-9)
    np.testing.assert_allclose(qy, ry, atol=1e-9)
    # 6.5 m out, limit is 7 - 1 - 0.3 = 5.7 m: pulled in, smoothly.
    rx, ry = stadium(STRAIGHT, RADIUS + 6.5, 2.0)
    qx, qy = keep_inside(track, rx, ry, 0.3)
    _, n, _ = track.project(qx, qy)
    assert np.abs(n).max() <= HALF_WIDTH - 1.0 - 0.3 + 1e-6
    assert np.abs(n).max() > HALF_WIDTH - 1.0 - 0.3 - 0.05
