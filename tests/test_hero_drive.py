"""The driving hero's logic (f1sim.app.ui.hero_drive): offline, no window."""

import math

import numpy as np
import pytest

from f1sim.app.ui import hero_drive as hd


def _circle_circuit(radius=100.0, speed=50.0) -> hd.Circuit:
    """A counter-clockwise circle, one point per metre: a constant left-hander."""
    n = int(round(2 * math.pi * radius))
    a = np.arange(n) / radius
    line = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    heading = a + math.pi / 2
    edge = lambda r: np.column_stack([r * np.cos(a), r * np.sin(a)])  # noqa: E731
    return hd.Circuit(
        line=line,
        speed=np.full(n, speed),
        heading=heading,
        curvature=np.full(n, 1.0 / radius),
        left=edge(radius - 6),
        right=edge(radius + 6),
        kerb=np.zeros((0, 3, 2)),
        kerb_red=np.zeros(0, bool),
        name="Circle",
        lap="",
    )


def test_the_bundled_circuit_is_a_real_singapore_lap():
    c = hd.load_circuit()
    assert c is not None and "Marina Bay" in c.name
    assert 4700 < c.length < 5100  # metres
    assert 85 < c.lap_time < 95  # a pole lap, about 1:29
    assert c.speed.min() > 15 and c.speed.max() < 95  # m/s
    assert len(c.kerb) == len(c.kerb_red) > 100
    assert hd.load_circuit("not-a-circuit") is None


def test_drive_loops_forever_at_the_lap_speed():
    c = _circle_circuit(speed=50.0)
    d = hd.Drive(c)
    for _ in range(int(2.5 * c.length / 50.0 / 0.05)):
        d.update(0.05)
    assert d.laps == 2  # kept going after each lap
    assert 0.0 <= d.s < c.length
    assert d.speed == pytest.approx(50.0)


def test_steering_follows_the_bend_left_positive_and_limited():
    d = hd.Drive(_circle_circuit(radius=100.0))
    expected = math.atan(d.wheelbase_m / 100.0) * hd.STEER_GAIN
    assert d.steer == pytest.approx(expected)  # a left-hander: positive
    right = _circle_circuit()
    right = hd.Circuit(**{**right.__dict__, "curvature": -right.curvature})
    assert hd.Drive(right).steer == pytest.approx(-expected)
    hairpin = _circle_circuit(radius=6.0)
    assert hd.Drive(hairpin).steer == pytest.approx(hd.MAX_STEER_RAD)


def test_pose_heading_is_continuous_across_the_line():
    c = _circle_circuit()
    d = hd.Drive(c)
    d.s = c.length - 0.5
    h_before = d.pose()[2]
    d.update(0.02)  # crosses the line
    h_after = d.pose()[2]
    assert abs(math.remainder(h_after - h_before, 2 * math.pi)) < 0.05


def test_world_to_screen_puts_the_heading_up():
    pose = (10.0, 20.0, math.radians(30.0))
    ahead = (10.0 + 5 * math.cos(pose[2]), 20.0 + 5 * math.sin(pose[2]))
    left = (10.0 - 2 * math.sin(pose[2]), 20.0 + 2 * math.cos(pose[2]))
    pts = hd.world_to_screen([pose[:2], ahead, left], pose, ppm=10.0, anchor=(300.0, 200.0))
    assert pts[0] == pytest.approx((300.0, 200.0))  # the rear axle at the anchor
    assert pts[1] == pytest.approx((300.0, 250.0))  # 5 m ahead: 50 px straight up
    assert pts[2] == pytest.approx((280.0, 200.0))  # 2 m left: 20 px to the left


def test_edge_quads_cover_a_band_around_the_line():
    c = _circle_circuit()
    q = hd.edge_quads(c.left, 0.1)
    assert q.shape == (2 * len(c.left), 3, 2)
    r = np.hypot(q[..., 0], q[..., 1])
    assert r.min() == pytest.approx(94.0 - 0.1, abs=1e-3)
    assert r.max() == pytest.approx(94.0 + 0.1, abs=1e-3)
