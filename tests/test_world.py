import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data.track_builder import build_track
from f1sim.sim.progress import DoneReason
from f1sim.sim.sensors import OBS_SIZE
from f1sim.sim.world import World, WorldConfig
from f1sim.track.track import Track

STRAIGHT, RADIUS, HALF_WIDTH = 600.0, 120.0, 7.0


@pytest.fixture(scope="module")
def track():
    x, y = stadium(STRAIGHT, RADIUS)
    w = np.full(x.size, HALF_WIDTH)
    return Track(build_track(x, y, w, w.copy(), track_id="stadium", name="Stadium"))


def _hold_line(obs, target_v=30.0):
    """A crude driver: steer against heading error and offset, hold a modest speed."""
    heading, lateral, curvature = obs[:, 10], obs[:, 11], obs[:, 12]
    # Feed-forward: steer for curvature κ is atan(3.6 κ) / 0.30 ≈ 12 κ, and the feature is 50 κ
    steer = np.clip(-3.0 * heading - 0.6 * lateral + 0.24 * curvature, -1, 1)
    v = obs[:, 9] * 92.0
    pedal = np.clip((target_v - v) / 5.0, -1, 1)
    return np.stack([steer, pedal], axis=1)


def test_reset_places_cars_on_the_centre_line(track):
    world = World(track, WorldConfig(n_cars=3))
    obs = world.reset(start_s=np.array([0.0, 100.0, 1000.0]), start_v=20.0)
    assert obs.shape == (3, OBS_SIZE) and obs.dtype == np.float32
    np.testing.assert_allclose(world.progress.state.s, [0.0, 100.0, 1000.0], atol=1e-6)
    np.testing.assert_allclose(world.progress.state.n, 0.0, atol=1e-6)
    np.testing.assert_allclose(world.car.v, 20.0)
    np.testing.assert_allclose(obs[:, 10], 0.0, atol=1e-6)  # heading error


def test_driving_makes_progress_and_completes_a_lap(track):
    world = World(track, WorldConfig(n_cars=2, episode_time_s=400.0))
    obs = world.reset(start_s=np.array([5.0, 600.0]), start_v=30.0)
    laps, total = np.zeros(2, int), np.zeros(2)
    for _ in range(int(150 * 30)):  # 150 s of decisions at 30 Hz
        res = world.step(_hold_line(obs))
        obs = res.obs
        total += res.progress_delta
        laps += res.lap_completed
        assert not res.done.any(), DoneReason(int(res.done_reason.max()))
    assert (laps >= 1).all()
    np.testing.assert_allclose(total, world.progress.state.progress - [0.0, 0.0], rtol=1e-12)
    assert world.progress.state.best_lap_time[0] == pytest.approx(track.length / 30.0, rel=0.05)


def test_done_cars_are_frozen_and_time_limit_applies(track):
    world = World(track, WorldConfig(n_cars=2, episode_time_s=2.0))
    obs = world.reset(start_s=10.0, start_v=30.0)
    actions = _hold_line(obs)
    actions[1] = [1.0, 1.0]  # car 1 turns hard left into the wall
    res = world.step(actions)
    for _ in range(60):
        actions = _hold_line(res.obs)
        actions[1] = [1.0, 1.0]
        res = world.step(actions)
        if res.done[1]:
            break
    assert res.done[1] and res.done_reason[1] == DoneReason.OFF_TRACK
    frozen_at = (world.car.x[1], world.car.y[1])
    for _ in range(10):
        res = world.step(_hold_line(res.obs))
    assert (world.car.x[1], world.car.y[1]) == frozen_at
    while not res.done[0]:
        res = world.step(_hold_line(res.obs))
    assert res.done_reason[0] == DoneReason.TIME_LIMIT
    assert world.progress.state.t[0] == pytest.approx(2.0, abs=world.config.dt)


def test_masked_reset_respawns_only_selected_cars(track):
    world = World(track, WorldConfig(n_cars=3))
    obs = world.reset(start_s=50.0, start_v=25.0)
    for _ in range(30):
        obs = world.step(_hold_line(obs)).obs
    before = world.car.x.copy()
    world.reset(start_s=200.0, start_v=0.0, mask=np.array([False, True, False]))
    assert world.car.x[0] == before[0] and world.car.x[2] == before[2]
    assert world.progress.state.s[1] == pytest.approx(200.0, abs=1e-6)
    assert world.car.v[1] == 0.0


def test_same_inputs_give_identical_runs(track):
    def run():
        world = World(track, WorldConfig(n_cars=8))
        obs = world.reset(start_s=np.linspace(0, 1500, 8), start_v=25.0)
        rng = np.random.default_rng(3)
        for _ in range(200):
            obs = world.step(_hold_line(obs) + rng.normal(0, 0.1, (8, 2))).obs
        return obs, world.progress.state.progress.copy()

    (o1, p1), (o2, p2) = run(), run()
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(p1, p2)
