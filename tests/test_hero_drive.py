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


def test_world_to_screen_is_a_fixed_top_down_view():
    # The view never turns with the car: whatever its heading, world +y is up and +x right
    for heading in (0.0, math.radians(30.0), math.radians(-120.0)):
        pose = (10.0, 20.0, heading)
        pts = hd.world_to_screen(
            [pose[:2], (15.0, 20.0), (10.0, 25.0)], pose, ppm=10.0, anchor=(300.0, 200.0)
        )
        assert pts[0] == pytest.approx((300.0, 200.0))  # the rear axle at the anchor
        assert pts[1] == pytest.approx((350.0, 200.0))  # 5 m east: 50 px right
        assert pts[2] == pytest.approx((300.0, 250.0))  # 5 m north: 50 px up


def test_edge_quads_cover_a_band_around_the_line():
    c = _circle_circuit()
    q = hd.edge_quads(c.left, 0.1)
    assert q.shape == (2 * len(c.left), 3, 2)
    r = np.hypot(q[..., 0], q[..., 1])
    assert r.min() == pytest.approx(94.0 - 0.1, abs=1e-3)
    assert r.max() == pytest.approx(94.0 + 0.1, abs=1e-3)


def _straight_circuit(length=2000, speed=50.0, half_width=6.0) -> hd.Circuit:
    """A long straight, closed by a jump back (only the straight part is used)."""
    x = np.arange(length, dtype=float)
    line = np.column_stack([x, np.zeros(length)])
    return hd.Circuit(
        line=line,
        speed=np.full(length, speed),
        heading=np.zeros(length),
        curvature=np.zeros(length),
        left=line + (0.0, half_width),
        right=line - (0.0, half_width),
        kerb=np.zeros((0, 3, 2)),
        kerb_red=np.zeros(0, bool),
        name="Straight",
        lap="",
    )


def test_track_room_measures_both_limits():
    left, right = hd.track_room(_circle_circuit())
    assert left == pytest.approx(np.full_like(left, 6.0), abs=0.01)
    assert right == pytest.approx(np.full_like(right, 6.0), abs=0.01)
    left, right = hd.track_room(hd.load_circuit())
    assert left.min() > 0 and right.min() > 0
    half = 0.5 * (left + right)  # the 12 m road, wherever the line runs on it
    assert np.mean((half > 5.0) & (half < 7.0)) > 0.98


def test_an_offset_moves_the_car_and_turns_it_into_the_move():
    d = hd.Drive(_straight_circuit())
    d.s = 100.0
    d.d_target = 2.5
    steers, headings = [], []
    for _ in range(int(3.0 / 0.01)):
        d.update(0.01)
        steers.append(d.steer)
        headings.append(d.pose()[2])
    x, y, h = d.pose()
    assert y == pytest.approx(2.5, abs=0.02)  # settled 2.5 m left of the line
    assert h == pytest.approx(0.0, abs=1e-3)  # straight again
    assert max(headings) > 0.01  # it turned left to get there...
    assert max(steers) > 0.01 and min(steers) < -0.005  # ...steering in, then out
    assert np.argmax(steers) < np.argmin(steers)


def test_an_offset_inside_a_bend_tightens_the_steering():
    c = _circle_circuit(radius=100.0)
    d = hd.Drive(c)
    d.place(0.0, 2.0)  # 2 m to the inside of a left-hander
    x, y, _ = d.pose()
    assert math.hypot(x, y) == pytest.approx(98.0, abs=1e-6)
    expected = math.atan(d.wheelbase_m / 98.0) * hd.STEER_GAIN
    assert d.steer == pytest.approx(expected, rel=1e-6)


@pytest.fixture(scope="module")
def traffic_run():
    """Ten laps of Marina Bay with the hero and the slower cars, checked every step."""
    c = hd.load_circuit()
    hero = hd.Drive(c)
    traffic = hd.Traffic(hero, seed=3)
    dt = 1 / 30
    overlaps = outside = 0
    for _ in range(int(10 * c.lap_time / dt)):
        traffic.update(dt)
        cars = [hero, *(r.drive for r in traffic.rivals)]
        for car in cars:
            i = int(car.s) % len(c.line)
            room = (traffic.room_left if car.d > 0 else traffic.room_right)[i]
            if abs(car.d) > 0.15 and abs(car.d) + hd.CAR_HALF_WIDTH_M > room + 0.05:
                outside += 1
        prints = [hd.footprint(car.pose()) for car in cars]
        for i in range(len(cars)):
            for j in range(i + 1, len(cars)):
                if hd.overlap(prints[i], prints[j]):
                    overlaps += 1
    return traffic, overlaps, outside


def test_offset_paths_stay_inside_the_track_limits(traffic_run):
    traffic, _, outside = traffic_run
    assert outside == 0
    assert max(abs(r.drive.d) for r in traffic.rivals) <= hd.GIVE_MAX_M + 1e-6


def test_no_two_cars_ever_overlap(traffic_run):
    assert traffic_run[1] == 0


def test_every_slower_car_is_passed_and_comes_back_ahead_out_of_view(traffic_run):
    traffic = traffic_run[0]
    assert traffic.hero.laps >= 8  # it hardly ever waits behind a slower car
    assert all(r.respawns >= 20 for r in traffic.rivals)
    log = np.array(traffic.respawn_log)
    assert log[:, 1].min() > traffic.view_m  # left behind out of view...
    assert log[:, 2].min() > traffic.view_m  # ...and back ahead out of view
    assert len(log) / (10 * traffic.circuit.lap_time) > 1 / 5.0  # an overtake every few seconds


def test_traffic_repeats_with_its_seed():
    c = hd.load_circuit()
    room = hd.track_room(c)
    runs = []
    for _ in range(2):
        t = hd.Traffic(hd.Drive(c), seed=7, room=room)
        for _ in range(600):
            t.update(1 / 60)
        runs.append([r.drive.pose() for r in t.rivals] + [t.hero.pose()])
    assert runs[0] == runs[1]


def test_footprint_overlap():
    a = hd.footprint((0.0, 0.0, 0.0))
    assert hd.overlap(a, hd.footprint((3.0, 1.5, 0.0)))  # alongside, too close
    assert not hd.overlap(a, hd.footprint((0.0, 2.2, 0.0)))  # alongside, clear
    assert not hd.overlap(a, hd.footprint((5.8, 0.0, 0.0)))  # nose to tail, clear
    assert hd.overlap(a, hd.footprint((5.0, 0.0, 0.3)))


def test_trackside_lamps_sit_outside_the_limits_in_red_and_blue_pairs():
    c = hd.load_circuit()
    room = hd.track_room(c)
    lamps = hd.trackside_lamps(c, room=room)
    units = len(lamps) // 2
    assert c.length / hd.LAMP_GAP_M[1] < units < c.length / hd.LAMP_GAP_M[0]
    assert [lp.colour for lp in lamps[:8]] == ["red", "blue"] * 2 + ["blue", "red"] * 2
    assert lamps[0].phase != lamps[2].phase  # neighbours flash out of step
    assert hd.trackside_lamps(c, room=room) == lamps  # seeded
    assert hd.trackside_lamps(c, room=room, seed=1) != lamps
    import shapely

    road = shapely.Polygon(c.left).symmetric_difference(shapely.Polygon(c.right))
    for lp in lamps:
        assert not road.contains(shapely.Point(lp.x, lp.y))  # off the track
        d = np.min(np.hypot(c.line[:, 0] - lp.x, c.line[:, 1] - lp.y))
        assert 1.0 < d < 17.0  # but beside it


def test_lamps_flash_alternately():
    red = hd.Lamp(0.0, 0.0, "red", 0.0)
    blue = hd.Lamp(0.0, 0.0, "blue", 0.0)
    ts = np.linspace(0.0, hd.LAMP_CYCLE_S, 700, endpoint=False)
    r = np.array([hd.lamp_brightness(red, t) for t in ts])
    b = np.array([hd.lamp_brightness(blue, t) for t in ts])
    assert r.max() == pytest.approx(1.0) and b.max() == pytest.approx(1.0)
    assert (r > 0.5).sum() > 0 and (b > 0.5).sum() > 0
    assert not np.any((r > 0.2) & (b > 0.2))  # never both at once
    assert np.mean(r < 0.05) > 0.3  # flashes: dark between them


def test_the_bundled_driving_stage_ships_with_its_maps():
    st = hd.load_stage()
    assert st is not None
    cells = len(st.xs) * len(st.ys)
    assert st.xs[0] <= -1.8 and st.xs[-1] >= 1.8  # photographed beside the car too
    assert st.ys[0] < 0.0 and st.ys[-1] > st.size_m[1]  # and past both ends
    for layer in (st.body, st.wheels):
        assert layer["light"].shape[0] == cells and layer["light"].ndim == 3  # grey photos
        h, w = layer["albedo"].shape[:2]
        assert layer["light"].shape[1:] == ((h + 1) // 2, (w + 1) // 2)  # at half size
        for m in hd.STAGE_MAPS:
            assert layer[m].shape == (h, w, 4)
    assert st.white_floor > 0 and st.ambient_gain >= 1
    assert 5.0 < st.size_m[1] < 6.5 and 3.0 < st.wheelbase < 4.0


def test_the_driving_stage_is_cached_once_decoded(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    cache = hd.stage_cache_path(hd.DRIVE_DIR)
    assert str(cache).startswith(str(tmp_path)) and not cache.exists()
    first = hd.load_stage()
    assert cache.is_file()  # decoded once...
    again = hd.load_stage()  # ...then read back
    for layer in ("body", "wheel"):
        for k in ("light", *hd.STAGE_MAPS):
            assert np.array_equal(
                getattr(first, layer if layer == "body" else "wheels")[k],
                getattr(again, layer if layer == "body" else "wheels")[k],
            )
