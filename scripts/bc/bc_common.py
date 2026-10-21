"""Shared setup for the behaviour-cloning experiments."""

import math

import numpy as np

from f1sim.agents import imitation
from f1sim.agents import policy as pol
from f1sim.data import catalog, tum_source
from f1sim.data.track_pipeline import load_track
from f1sim.sim.car_model import CarParams
from f1sim.sim.world import World, WorldConfig
from f1sim.train.trainer import stage2_start

CAR = CarParams()
TRACK = load_track("silverstone").sim_track()
RACELINE = tum_source.parse_tum_raceline(
    tum_source.fetch_tum_raceline(catalog.get("silverstone").tum_name)
)
CFG = WorldConfig(n_cars=1, use_track_limits=False, car=CAR, episode_time_s=300.0)
DEC_DT = CFG.dt * CFG.action_repeat
S2_S, S2_V = stage2_start(TRACK, CAR, 80.0)
SPEC = pol.PolicySpec(hidden=(16, 16))


def teacher() -> imitation.Teacher:
    return imitation.Teacher(TRACK, CAR, RACELINE, DEC_DT)


def lap(controller, n_cars=1, cap_s=300.0, obs_fn=None):
    """Drive from the stage-2 start; controller(world, obs) -> actions. Returns (lap_time,
    done_reason, progress)."""
    world = World(
        TRACK, WorldConfig(n_cars=n_cars, use_track_limits=False, car=CAR, episode_time_s=cap_s)
    )
    obs_fn = obs_fn or (lambda _w, o: o)
    obs = obs_fn(world, world.reset(start_s=S2_S, start_v=S2_V))
    st = world.progress.state
    lap_t = np.full(n_cars, np.nan)
    for _ in range(int(math.ceil(cap_s / DEC_DT))):
        res = world.step(controller(world, obs))
        obs = obs_fn(world, res.obs)
        new = res.lap_completed & np.isnan(lap_t)
        lap_t[new] = res.lap_time[new]
        if np.all(st.done | ~np.isnan(lap_t)):
            break
    return lap_t, st.done_reason.copy(), st.progress.copy()


def teacher_controller(t: imitation.Teacher):
    def c(world, obs):
        return t.act(world.car.x, world.car.y, world.car.yaw, world.car.v)

    return c


def policy_controller(params):
    def c(world, obs):
        return pol.act(SPEC, params, obs)

    return c


AHEAD_M = (10.0, 30.0, 60.0, 100.0)
SPEED_AHEAD_M = (0.0, 50.0, 100.0, 200.0)
N_EXTRA = 2 + len(AHEAD_M) + len(SPEED_AHEAD_M)


def raceline_features(t: imitation.Teacher):
    """obs_fn adding raceline-relative inputs (computed from the teacher's path)."""
    pt = t.ptrack

    hints = {}

    def f(world, obs):
        c = world.car
        h0 = hints.get(id(world))
        fresh = world.progress.state.t <= 0.0  # just reset: search globally
        if h0 is None or h0.size != c.x.size or fresh.all():
            sp, e, idx = pt.project(c.x, c.y)
        else:
            sp, e, idx = pt.project(c.x, c.y, hint=h0)
            if fresh.any():
                g = pt.project(c.x[fresh], c.y[fresh])
                sp[fresh], e[fresh], idx[fresh] = g
        hints[id(world)] = idx
        h = pt.heading_at(sp)
        herr = (c.yaw - h + np.pi) % (2 * np.pi) - np.pi
        cols = [np.clip(e / 5.0, -2, 2), herr / 0.5]
        for d in AHEAD_M:
            q = np.atleast_2d(pt.point_at(sp + d))
            dx, dy = q[:, 0] - c.x, q[:, 1] - c.y
            cols.append(np.clip((-np.sin(c.yaw) * dx + np.cos(c.yaw) * dy) / 20.0, -2, 2))
        for d in SPEED_AHEAD_M:
            cols.append(t.profile_speed(sp + d) / 92.0)
        return np.hstack([obs, np.column_stack(cols).astype(np.float32)])

    return f
