"""Behaviour cloning + DAgger on Silverstone (experiment D16).

Usage: python bc_run.py <out.npz> <plain|aug> [hidden...]  (aug adds the raceline inputs)
"""

import sys
import time

import numpy as np
from bc_common import (
    CAR,
    S2_S,
    S2_V,
    TRACK,
    imitation,
    lap,
    pol,
    policy_controller,
    teacher,
)

from f1sim.sim.scripted import speed_profile
from f1sim.sim.world import World, WorldConfig

out = sys.argv[1]
aug = sys.argv[2] == "aug"
hidden = tuple(int(h) for h in sys.argv[3:]) or (16, 16)
import bc_common as bcc  # noqa: E402

spec = pol.PolicySpec(obs_size=19 + (bcc.N_EXTRA if aug else 0), hidden=hidden)
import bc_common  # noqa: E402

bc_common.SPEC = spec
rng = np.random.default_rng(0)
t0 = time.perf_counter()
N = 64
world = World(TRACK, WorldConfig(n_cars=N, use_track_limits=False, car=CAR, episode_time_s=40.0))
tch = teacher()
obs_fn = bcc.raceline_features(tch) if aug else None
d = TRACK.data
v_centre = speed_profile(d.x, d.y, CAR, closed=True)


def starts(n):
    s = rng.uniform(0.0, TRACK.length, n)
    v = np.interp(s, d.s, v_centre) * rng.uniform(0.6, 1.0, n)
    s[:4] = S2_S  # a few from the training start
    v[:4] = S2_V
    return s, v


def batch(params=None, beta=1.0, dur=20.0):
    s, v = starts(N)
    r = imitation.collect(
        world, tch, s, v, dur, spec=spec, params=params, beta=beta, rng=rng, obs_fn=obs_fn
    )
    return r


obs, act = [], []
for _ in range(4):
    r = batch()
    obs.append(r.obs)
    act.append(r.actions)
print(f"teacher data {sum(len(o) for o in obs)} samples, crash {r.crash_rate:.0%}")
params = pol.init_params(spec, 1, rng)[0]
history = []
for it in range(8):
    X, Y = np.concatenate(obs), np.concatenate(act)
    params = imitation.fit(spec, X, Y, params, epochs=40 if it == 0 else 20, rng=rng)
    lt, reason, prog = lap(policy_controller(params), obs_fn=obs_fn)
    # student rollouts from random starts: data for the next round, and a robustness check
    beta = 0.5 if it == 0 else 0.0
    r = batch(params, beta=beta)
    obs.append(r.obs)
    act.append(r.actions)
    msg = (
        f"iter {it}: n={len(X)} loss {imitation.loss(spec, params, X, Y):.4f} "
        f"lap {lt[0]:.2f}s reason {reason[0]} progress {prog[0]:.0f}m | "
        f"student crash {r.crash_rate:.0%} ({time.perf_counter() - t0:.0f}s)"
    )
    print(msg, flush=True)
    history.append(params.copy())
    np.savez(out, params=params, history=np.array(history), hidden=np.array(hidden))
