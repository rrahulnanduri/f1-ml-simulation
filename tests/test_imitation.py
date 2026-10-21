"""Behaviour cloning (f1sim.agents.imitation) on a synthetic oval, offline."""

import math

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.agents import imitation
from f1sim.agents import policy as pol
from f1sim.data.track_builder import build_track
from f1sim.sim.car_model import CarParams
from f1sim.sim.sensors import OBS_SIZE
from f1sim.sim.world import World, WorldConfig
from f1sim.track.track import Track

STRAIGHT, RADIUS, HALF_WIDTH = 200.0, 60.0, 7.0
CAR = CarParams()


@pytest.fixture(scope="module")
def track():
    x, y = stadium(STRAIGHT, RADIUS)
    w = np.full(x.size, HALF_WIDTH)
    return Track(build_track(x, y, w, w.copy(), track_id="oval", name="Oval"))


def _world(track, n, episode_s=60.0):
    return World(track, WorldConfig(n_cars=n, use_track_limits=False, episode_time_s=episode_s))


def test_teacher_laps_the_oval_and_merges_from_off_the_line(track):
    world = _world(track, 2)
    teacher = imitation.Teacher(track, CAR, None, world.config.dt * world.config.action_repeat)
    world.reset(start_s=[10.0, 150.0], start_v=[20.0, 30.0])
    world.car.y[1] += 3.0  # 3 m off the centre line, on a straight
    st, car = world.progress.state, world.car
    laps = np.zeros(2, dtype=bool)
    for _ in range(int(60.0 / (world.config.dt * world.config.action_repeat))):
        res = world.step(teacher.act(car.x, car.y, car.yaw, car.v))
        laps |= res.lap_completed
        if laps.all():
            break
    assert laps.all()
    assert not st.done.any()  # nobody crashed


def test_collect_labels_every_live_state_with_the_teacher(track):
    world = _world(track, 4)
    teacher = imitation.Teacher(track, CAR, None, world.config.dt * world.config.action_repeat)
    s0 = np.array([0.0, 100.0, 300.0, 500.0])
    r = imitation.collect(world, teacher, s0, np.full(4, 25.0), 2.0)
    decisions = math.ceil(2.0 / (world.config.dt * world.config.action_repeat))
    assert r.obs.shape == (4 * decisions, OBS_SIZE)
    assert r.actions.shape == (4 * decisions, 2)
    assert np.all(np.abs(r.actions) <= 1.0)
    assert r.crash_rate == 0.0 and r.mean_progress_m > 40.0
    # The labels are the teacher's actions in the recorded states (same first decision).
    world.reset(start_s=s0, start_v=np.full(4, 25.0))
    teacher.reset_hints()
    first = teacher.act(world.car.x, world.car.y, world.car.yaw, world.car.v)
    np.testing.assert_allclose(r.actions[:4], first)


def test_fit_recovers_a_policy_from_its_own_actions():
    spec = pol.PolicySpec(hidden=(8, 8))
    rng = np.random.default_rng(3)
    target = pol.init_params(spec, 1, rng)[0]
    obs = rng.uniform(-1.0, 1.0, (4000, spec.obs_size)).astype(np.float32)
    actions = pol.act(spec, target, obs)
    start = pol.init_params(spec, 1, rng)[0]
    before = imitation.loss(spec, start, obs, actions)
    fitted = imitation.fit(spec, obs, actions, start, epochs=40, batch=256, lr=1e-2, rng=rng)
    after = imitation.loss(spec, fitted, obs, actions)
    assert fitted.dtype == np.float32 and fitted.shape == (spec.n_params,)
    assert after < 0.02 * before


def test_fit_rejects_relu():
    spec = pol.PolicySpec(hidden=(4,), activation="relu")
    with pytest.raises(ValueError, match="tanh"):
        imitation.fit(spec, np.zeros((2, spec.obs_size)), np.zeros((2, 2)), np.zeros(spec.n_params))
