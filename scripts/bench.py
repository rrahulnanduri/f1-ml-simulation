"""Simulation speed benchmark (revamp-plan M2 target: >= 50x real time with 64 cars, headless).

    python scripts/bench.py [--track silverstone] [--cars 1 16 64 256] [--seconds 20]

For each car count, a simple line-holding driver runs every car for ``--seconds`` of simulated
time. Reports the real-time factor (simulated seconds per wall second), car-steps per second and
how the time splits between car physics, progress tracking and observations.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from f1sim.data.track_pipeline import load_track
from f1sim.sim import car_model
from f1sim.sim.world import World, WorldConfig


def hold_line(obs: np.ndarray, target_v: float = 45.0) -> np.ndarray:
    """Steer against heading error and offset with curvature feed-forward; hold ``target_v``."""
    steer = np.clip(-3.0 * obs[:, 10] - 0.6 * obs[:, 11] + 0.24 * obs[:, 12], -1, 1)
    pedal = np.clip((target_v - obs[:, 9] * 92.0) / 5.0, -1, 1)
    return np.stack([steer, pedal], axis=1)


def bench(track, n: int, sim_seconds: float) -> dict:
    cfg = WorldConfig(n_cars=n, episode_time_s=1e9)
    world = World(track, cfg)
    obs = world.reset(start_s=np.linspace(0, track.length, n, endpoint=False), start_v=40.0)
    decisions = int(round(sim_seconds / (cfg.dt * cfg.action_repeat)))

    # Component timings, accumulated while running the same loop World.step runs
    t_car = t_prog = t_obs = 0.0
    alive_end = 0
    wall0 = time.perf_counter()
    for _ in range(decisions):
        actions = hold_line(obs)
        for _ in range(cfg.action_repeat):
            a = time.perf_counter()
            car_model.step(world.car, actions[:, 0], actions[:, 1], cfg.car, cfg.dt)
            b = time.perf_counter()
            world.progress.update(world.car.x, world.car.y, cfg.dt)
            c = time.perf_counter()
            t_car += b - a
            t_prog += c - b
        world.prev_action[:] = actions
        a = time.perf_counter()
        obs = world._observe()
        t_obs += time.perf_counter() - a
    wall = time.perf_counter() - wall0
    alive_end = int((~world.progress.state.done).sum())

    # The real World.step (includes frozen-car bookkeeping and the time limit)
    world2 = World(track, cfg)
    obs = world2.reset(start_s=np.linspace(0, track.length, n, endpoint=False), start_v=40.0)
    w0 = time.perf_counter()
    for _ in range(decisions):
        obs = world2.step(hold_line(obs)).obs
    wall_step = time.perf_counter() - w0

    sim = decisions * cfg.action_repeat * cfg.dt
    return {
        "n": n,
        "rtf": sim / wall_step,
        "car_steps_per_s": n * decisions * cfg.action_repeat / wall_step,
        "ms_per_decision": 1000 * wall_step / decisions,
        "split": (t_car / wall, t_prog / wall, t_obs / wall),
        "alive": alive_end,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--track", default="silverstone")
    ap.add_argument("--cars", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--seconds", type=float, default=20.0, help="simulated seconds per run")
    args = ap.parse_args()

    track = load_track(args.track)
    print(f"{track.data.name}: {track.length:.0f} m, {len(track.data.x)} points")
    print(
        f"{'cars':>5} {'x real time':>11} {'car-steps/s':>12} {'ms/decision':>12} "
        f"{'physics':>8} {'progress':>9} {'obs':>6} {'alive':>6}"
    )
    for n in args.cars:
        r = bench(track, n, args.seconds)
        car, prog, obs = r["split"]
        print(
            f"{r['n']:>5} {r['rtf']:>10.1f}x {r['car_steps_per_s']:>12,.0f} "
            f"{r['ms_per_decision']:>12.2f} {car:>7.0%} {prog:>9.0%} {obs:>6.0%} "
            f"{r['alive']:>3}/{r['n']}"
        )


if __name__ == "__main__":
    main()
