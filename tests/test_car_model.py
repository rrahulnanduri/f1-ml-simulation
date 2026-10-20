"""Car model sanity tests: analytic / independently integrated targets (revamp-plan section 6)."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from f1sim.sim.car_model import (
    CarParams,
    CarState,
    G,
    StepInfo,
    grip_limit,
    load_car_params,
    step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAR_TOML = REPO_ROOT / "configs" / "car_f1.toml"
DT = 1.0 / 60.0
P = CarParams()
KMH = 1.0 / 3.6


# ---------------------------------------------------------------------------- helpers


def drive_accel(v: float, p: CarParams = P) -> float:
    """Full-throttle tyre acceleration (power- or traction-limited), independent of src."""
    return min(p.power_w / (p.mass_kg * max(v, 1e-9)), p.max_drive_g * G)


def hold_speed_pedal(v: np.ndarray, v_target: float, p: CarParams = P) -> np.ndarray:
    """Feed-forward drag compensation plus a P term, so v_target is a fixed point."""
    ff = p.drag_coeff * v_target**2 / drive_accel(v_target, p)
    return np.clip(ff + 0.5 * (v_target - v), -1.0, 1.0)


def max_steady_speed(radius_m: float, p: CarParams = P) -> float:
    """Largest constant speed on a circle: solve v^2 / R = g * (base + aero * v^2) for v.

    v^2 = g * base / (1/R - g * aero); infinite when downforce grip outgrows v^2 / R.
    """
    denom = 1.0 / radius_m - G * p.grip_aero_coeff
    return math.inf if denom <= 0.0 else math.sqrt(G * p.grip_base_g / denom)


def circle_lap_time(radius_m: float, p: CarParams = P) -> float:
    """Lap time of a circle of the given radius at the maximum steady speed."""
    return 2.0 * math.pi * radius_m / max_steady_speed(radius_m, p)


def fit_circle_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Algebraic (Kasa) least-squares circle fit."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    return math.sqrt(d * d / 4.0 + e * e / 4.0 - f)


def cornering_state(n: int, v: float, radius_m: float) -> tuple[CarState, float]:
    """Cars at speed v with the steering already at the kinematic angle for radius_m."""
    delta = math.atan(P.wheelbase_m / radius_m)
    state = CarState.at(n, 0.0, 0.0, 0.0, v)
    state.delta[:] = delta
    return state, delta / P.max_steer_rad


# ---------------------------------------------------------------------------- longitudinal


def test_top_speed_matches_power_drag_balance():
    v_analytic = (P.power_w / (P.mass_kg * P.drag_coeff)) ** (1.0 / 3.0)
    assert v_analytic == pytest.approx(92.1, abs=0.1)
    state = CarState.at(1, 0.0, 0.0, 0.0)
    for _ in range(int(60.0 / DT)):
        step(state, 0.0, 1.0, P, DT)
    v_top = state.v[0]
    assert abs(v_top - v_analytic) < 2.0
    # The discrete fixed point is exactly the analytic balance; 60 s is ~20 time constants.
    assert abs(v_top - v_analytic) < 1e-3


def test_zero_to_100_kmh():
    a, c = P.max_drive_g * G, P.drag_coeff
    target = 100.0 * KMH
    assert drive_accel(target) == a  # traction-limited the whole way
    # dv/dt = a - c v^2  =>  t = atanh(v sqrt(c/a)) / sqrt(a c)
    t_analytic = math.atanh(target * math.sqrt(c / a)) / math.sqrt(a * c)

    state = CarState.at(1, 0.0, 0.0, 0.0)
    v_prev, t = 0.0, 0.0
    while state.v[0] < target:
        v_prev = state.v[0]
        step(state, 0.0, 1.0, P, DT)
        t += DT
    t_sim = t - DT * (state.v[0] - target) / (state.v[0] - v_prev)
    assert 2.0 <= t_sim <= 3.0
    assert t_sim == pytest.approx(t_analytic, abs=DT)


def test_braking_300_to_80_matches_independent_integral():
    v0, v1 = 300.0 * KMH, 80.0 * KMH

    def decel(v):
        tyre = np.minimum(P.max_brake_g * G, G * (P.grip_base_g + P.grip_aero_coeff * v * v))
        return tyre + P.drag_coeff * v * v

    # dx = v dv / decel(v), integrated on a fine grid.
    v_grid = np.linspace(v1, v0, 200_001)
    d_ref = float(np.trapezoid(v_grid / decel(v_grid), v_grid))

    state = CarState.at(1, 0.0, 0.0, 0.0, v0)
    info = step(state, 0.0, -1.0, P, DT)
    assert info.a_long[0] == pytest.approx(-decel(v0), rel=1e-12)
    x_prev, v_prev = state.x[0], state.v[0]
    while state.v[0] > v1:
        x_prev, v_prev = state.x[0], state.v[0]
        info = step(state, 0.0, -1.0, P, DT)
        assert not info.sliding[0]
    d_sim = x_prev + (state.x[0] - x_prev) * (v_prev - v1) / (v_prev - state.v[0])
    assert d_sim == pytest.approx(d_ref, rel=0.02)
    assert 50.0 < d_sim < 150.0
    assert state.y[0] == 0.0


def test_no_reverse():
    state = CarState.at(2, 0.0, 0.0, 0.0, 0.0)
    state.v[1] = 5.0
    for _ in range(120):
        info = step(state, 0.0, -1.0, P, DT)
        assert np.all(state.v >= 0.0)
        assert np.all(info.a_long <= 0.0)
    assert np.all(state.v == 0.0)
    assert state.x[0] == 0.0
    x_stop = state.x[1]
    step(state, 0.5, -1.0, P, DT)
    assert state.x[1] == x_stop and state.yaw[1] == 0.0


def test_standing_start_moves():
    state = CarState.at(1, 0.0, 0.0, 0.0)
    info = step(state, 0.0, 1.0, P, DT)
    assert info.a_long[0] == pytest.approx(P.max_drive_g * G, rel=1e-12)
    assert state.v[0] > 0.0 and state.x[0] > 0.0


# ---------------------------------------------------------------------------- lateral


def test_grip_limit_values():
    assert grip_limit(0.0, P) == pytest.approx(1.6 * G)
    assert grip_limit(250.0 * KMH, P) / G == pytest.approx(4.93, abs=0.01)  # "about 5 g"
    v = np.array([0.0, 30.0, 70.0])
    assert grip_limit(v, P).shape == (3,)


@pytest.mark.parametrize("v", [30.0, 50.0, 70.0])
def test_steady_state_cornering_below_limit(v):
    r_min = v * v / grip_limit(v, P)
    radius = 1.25 * r_min  # lateral demand = 80% of the limit
    state, steer = cornering_state(1, v, radius)
    xs, ys = [], []
    for _ in range(int(3.0 / DT)):
        info = step(state, steer, hold_speed_pedal(state.v, v), P, DT)
        assert not info.sliding[0]
        xs.append(state.x[0])
        ys.append(state.y[0])
    assert state.v[0] == pytest.approx(v, abs=1e-6)
    assert info.a_lat[0] == pytest.approx(v * v / radius, rel=1e-6)
    assert info.a_lat[0] < grip_limit(v, P)
    assert fit_circle_radius(np.array(xs), np.array(ys)) == pytest.approx(radius, rel=1e-3)


def test_overspeed_into_tight_radius_slides_and_scrubs():
    v0, radius = 70.0, 30.0
    straight = CarState.at(1, 0.0, 0.0, 0.0, v0)
    state, steer = cornering_state(1, v0, radius)
    sliding, a_lat, v_after, v_straight = [], [], [], []
    for _ in range(int(5.0 / DT)):
        info = step(state, steer, 0.0, P, DT)
        step(straight, 0.0, 0.0, P, DT)
        sliding.append(info.sliding[0])
        a_lat.append(info.a_lat[0])
        v_after.append(state.v[0])
        v_straight.append(straight.v[0])
    sliding, a_lat, v_after = np.array(sliding), np.array(a_lat), np.array(v_after)

    assert sliding[0]
    assert np.all(np.abs(a_lat) <= grip_limit(v_after, P) * (1.0 + 1e-12))
    # Scrub: much slower than coasting in a straight line for the same time.
    one_s = int(1.0 / DT) - 1
    assert v_after[one_s] < v_straight[one_s] - 20.0
    assert np.all(np.diff(v_after) < 0.0)
    # Once slow enough to hold the radius it stops sliding and follows v^2 / R.
    v_hold = max_steady_speed(radius)
    assert v_after[-1] <= v_hold
    assert not sliding[-1]
    r_kin = P.wheelbase_m / math.tan(state.delta[0])
    assert a_lat[-1] == pytest.approx(v_after[-1] ** 2 / r_kin, rel=1e-9)


def test_throttle_overrun_in_corner_is_trimmed():
    v = 50.0
    radius = v * v / (0.98 * grip_limit(v, P))  # lateral demand = 98% of the limit
    state, steer = cornering_state(1, v, radius)
    info = step(state, steer, 1.0, P, DT)
    assert info.sliding[0]  # combined request exceeded the circle
    budget = math.sqrt(grip_limit(v, P) ** 2 - (v * v / radius) ** 2)
    expected = budget - P.drag_coeff * v * v
    assert info.a_long[0] == pytest.approx(expected, rel=1e-9)


def test_circle_r100_max_speed():
    radius = 100.0
    v_max = max_steady_speed(radius)
    assert v_max * v_max / radius == pytest.approx(grip_limit(v_max, P), rel=1e-12)
    assert v_max == pytest.approx(69.7, abs=0.1)
    assert circle_lap_time(radius) == pytest.approx(2 * math.pi * radius / v_max)
    # Holding speed also needs tyre force against drag inside the friction circle:
    # (v^2/R)^2 + (c v^2)^2 = a_max(v)^2, so the drag-aware steady speed is a bit lower.
    v_hold = math.sqrt(
        G * P.grip_base_g / (math.hypot(1.0 / radius, P.drag_coeff) - G * P.grip_aero_coeff)
    )
    assert v_hold == pytest.approx(68.9, abs=0.1)
    below, s_below = cornering_state(1, 0.995 * v_hold, radius)
    for _ in range(int(2.0 / DT)):
        info = step(below, s_below, hold_speed_pedal(below.v, 0.995 * v_hold), P, DT)
        assert not info.sliding[0]
    assert below.v[0] == pytest.approx(0.995 * v_hold, abs=1e-6)
    # At v_max the lateral demand alone fits, but holding speed overruns the circle.
    at_max, s_max = cornering_state(1, v_max, radius)
    info = step(at_max, s_max, hold_speed_pedal(at_max.v, v_max), P, DT)
    assert info.sliding[0] and at_max.v[0] < v_max
    # Just above v_max the lateral demand alone slides, even coasting.
    above, s_above = cornering_state(1, 1.01 * v_max, radius)
    assert step(above, s_above, 0.0, P, DT).sliding[0]
    # Above ~148 m radius downforce grip outgrows v^2 / R: no speed limit at all.
    assert max_steady_speed(1.0 / (G * P.grip_aero_coeff) + 1.0) == math.inf


# ---------------------------------------------------------------------------- steering


def test_steering_rate_limit():
    state = CarState.at(1, 0.0, 0.0, 0.0, 30.0)
    max_step = P.steer_rate_rad_s * DT
    deltas = [0.0]
    cmds = [1.0] * 20 + [-1.0] * 40
    for cmd in cmds:
        step(state, cmd, 0.0, P, DT)
        deltas.append(state.delta[0])
    deltas = np.array(deltas)
    assert np.all(np.abs(np.diff(deltas)) <= max_step * (1.0 + 1e-9))
    assert np.all(np.abs(deltas) <= P.max_steer_rad + 1e-15)
    up = round(P.max_steer_rad / max_step)  # 12 steps to full lock
    assert deltas[up - 1] < P.max_steer_rad
    assert deltas[up] == pytest.approx(P.max_steer_rad, abs=1e-12)
    assert deltas[20 + 2 * up] == pytest.approx(-P.max_steer_rad, abs=1e-12)
    assert deltas[20 + 2 * up - 1] > -P.max_steer_rad + 1e-9


def test_positive_steer_turns_left():
    state = CarState.at(1, 0.0, 0.0, 0.0, 20.0)
    for _ in range(30):
        info = step(state, 0.5, 0.0, P, DT)
    assert state.yaw[0] > 0.0 and state.y[0] > 0.0 and info.a_lat[0] > 0.0


# ---------------------------------------------------------------------------- vectorisation


def _random_run(n: int, seed: int, n_steps: int = 90):
    rng = np.random.default_rng(seed)
    init = {
        "x": rng.uniform(-100, 100, n),
        "y": rng.uniform(-100, 100, n),
        "yaw": rng.uniform(-np.pi, np.pi, n),
        "v": rng.uniform(0, 90, n),
        "delta": rng.uniform(-P.max_steer_rad, P.max_steer_rad, n),
    }
    steer = rng.uniform(-1.2, 1.2, (n_steps, n))
    pedal = rng.uniform(-1.2, 1.2, (n_steps, n))
    return init, steer, pedal


def _simulate(init, steer, pedal) -> tuple[CarState, list[StepInfo]]:
    state = CarState(**{k: v.copy() for k, v in init.items()})
    infos = [step(state, s, p, P, DT) for s, p in zip(steer, pedal, strict=True)]
    return state, infos


def test_vectorised_equals_per_car():
    n = 1000
    init, steer, pedal = _random_run(n, seed=1234)
    vec_state, vec_infos = _simulate(init, steer, pedal)
    assert any(info.sliding.any() for info in vec_infos)  # the batch exercises the slide path
    picks = np.unique(np.r_[0, n - 1, np.random.default_rng(7).choice(n, 30, replace=False)])
    for i in picks:
        one_init = {k: v[i : i + 1] for k, v in init.items()}
        one_state, one_infos = _simulate(one_init, steer[:, i : i + 1], pedal[:, i : i + 1])
        for f in dataclasses.fields(CarState):
            np.testing.assert_allclose(
                getattr(one_state, f.name), getattr(vec_state, f.name)[i : i + 1], 1e-12, 1e-12
            )
        for one, vec in zip(one_infos, vec_infos, strict=True):
            np.testing.assert_allclose(one.a_long, vec.a_long[i : i + 1], 1e-12, 1e-9)
            np.testing.assert_allclose(one.a_lat, vec.a_lat[i : i + 1], 1e-12, 1e-9)
            assert one.sliding[0] == vec.sliding[i]


def test_scalar_commands_broadcast_and_update_in_place():
    state = CarState.at(3, 1.0, 2.0, 0.3, 10.0)
    names = [f.name for f in dataclasses.fields(CarState)]
    before = [getattr(state, name) for name in names]
    info = step(state, 0.2, 0.5, P, DT)
    assert all(getattr(state, name) is arr for name, arr in zip(names, before, strict=True))
    assert info.a_long.shape == info.a_lat.shape == info.sliding.shape == (3,)
    assert info.sliding.dtype == bool
    assert np.all(state.x == state.x[0]) and state.x[0] != 1.0


def test_deterministic():
    init, steer, pedal = _random_run(1000, seed=99)
    s1, i1 = _simulate(init, steer, pedal)
    s2, i2 = _simulate(init, steer, pedal)
    for f in dataclasses.fields(CarState):
        assert np.array_equal(getattr(s1, f.name), getattr(s2, f.name))
    for a, b in zip(i1, i2, strict=True):
        assert np.array_equal(a.a_long, b.a_long)
        assert np.array_equal(a.a_lat, b.a_lat)
        assert np.array_equal(a.sliding, b.sliding)


def test_invalid_inputs():
    state = CarState.at(1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        step(state, 0.0, 0.0, P, 0.0)
    with pytest.raises(ValueError):
        CarState(x=np.zeros(2), y=np.zeros(2), yaw=np.zeros(2), v=np.zeros(3), delta=np.zeros(2))
    with pytest.raises(ValueError):
        CarParams(mass_kg=0.0)


# ---------------------------------------------------------------------------- config


def test_config_round_trip():
    assert load_car_params(CAR_TOML) == CarParams()
    text = CAR_TOML.read_text(encoding="utf-8")
    for f in dataclasses.fields(CarParams):
        assert f"\n{f.name} = " in text


def test_config_partial_file_keeps_defaults(tmp_path):
    path = tmp_path / "car.toml"
    path.write_text("mass_kg = 798\npower_w = 7.6e5\n", encoding="utf-8")
    params = load_car_params(path)
    assert params == dataclasses.replace(CarParams(), mass_kg=798.0, power_w=760e3)
    assert isinstance(params.mass_kg, float)


@pytest.mark.parametrize(
    "body", ["mass_kg = 800.0\ntop_speed = 92.0\n", "[car]\nmass_kg = 800.0\n", 'mass_kg = "x"\n']
)
def test_config_rejects_bad_keys_and_values(tmp_path, body):
    path = tmp_path / "car.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_car_params(path)
