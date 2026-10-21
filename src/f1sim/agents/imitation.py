"""Behaviour cloning of the scripted reference driver into a policy (D16 in the project log).

The GA plateaus at 101-116 s on Silverstone against the scripted driver's 87 s. Cloning
answers whether the small policy can drive like the scripted driver at all; if it can, the
clone is a much better starting point for the GA than random weights.

- :class:`Teacher` is the scripted driver of :func:`f1sim.sim.scripted.drive_reference_lap`
  (pure-pursuit steering on the TUM raceline and the model-inverting speed controller), made
  vectorised and callable at any state. It decides at the policy's rate (every
  ``action_repeat`` physics steps), so its laps are what a perfect clone could drive.
- :func:`collect` drives a :class:`~f1sim.sim.world.World` and records ``(obs, teacher
  action)`` pairs. The executed action is the teacher's or, for DAgger, the student policy's:
  labelling the states the *student* reaches teaches it to recover from its own mistakes,
  which a clean teacher lap never shows.
- :func:`fit` trains the policy's flat parameter vector by minibatch Adam on the mean squared
  action error (plain numpy backprop through :mod:`f1sim.agents.policy`'s layout).
- :func:`clone` runs the DAgger loop and returns the parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from f1sim.agents import policy as pol
from f1sim.sim import scripted
from f1sim.sim.car_model import V_FLOOR, CarParams, G
from f1sim.sim.world import World
from f1sim.track.track import Track

STEER_WEIGHT = 4.0
"""Loss weight of the steering error relative to the pedal's. Steering errors put the car off
track within a second; a pedal error only costs a little speed."""


MERGE_GAIN = 4.0
"""Extra pure-pursuit lookahead per metre off the raceline. The scripted lap starts on its
line; a car 3.6 m off it (the training start is on the centre line) steered back at full
lock and overshot off the track. A longer lookahead merges gently and is unchanged on the
line."""


class Teacher:
    """The scripted reference driver as a vectorised controller: ``act(x, y, yaw, v)``.

    ``track`` is the track the cars drive on (the World's, i.e. with track limits if it uses
    them): the raceline is pulled ``scripted.INSIDE_MARGIN_M`` inside its off-track limit.
    ``decision_dt`` is how long each action is held (``dt * action_repeat``).
    """

    def __init__(self, track: Track, car: CarParams, raceline_xy, decision_dt: float):
        self.car = car
        self.dt = float(decision_dt)
        self.path = scripted._reference_path(track, raceline_xy, scripted.INSIDE_MARGIN_M)
        self.v_prof = scripted._profile_on_path(
            self.path, car, curvature_margin=scripted.DRIVE_CURVATURE_MARGIN
        )
        self.ptrack = scripted._path_as_track(self.path)
        self.merge_gain = MERGE_GAIN
        self._hint: np.ndarray | None = None

    def reset_hints(self) -> None:
        """Forget the projection hints (after cars were reset somewhere else)."""
        self._hint = None

    def profile_speed(self, s_path) -> np.ndarray:
        return scripted._interp_periodic(s_path, self.path.s, self.v_prof, self.path.length)

    def act(self, x, y, yaw, v) -> np.ndarray:
        """Actions ``(N, 2)`` = (steer, pedal) in ``[-1, 1]`` for cars at these states."""
        p = self.car
        x, y, yaw, v = (np.asarray(a, dtype=np.float64) for a in (x, y, yaw, v))
        sp, e_path, idx = self.ptrack.project(x, y, hint=self._hint)
        self._hint = idx
        ld = np.clip(
            scripted.LOOKAHEAD_BASE_M + scripted.LOOKAHEAD_TIME_S * v,
            scripted.LOOKAHEAD_MIN_M,
            scripted.LOOKAHEAD_MAX_M,
        )
        ld = ld + self.merge_gain * np.abs(e_path)
        tgt = np.atleast_2d(self.ptrack.point_at(sp + ld))
        dx, dy = tgt[:, 0] - x, tgt[:, 1] - y
        y_loc = -np.sin(yaw) * dx + np.cos(yaw) * dy
        k_cmd = 2.0 * y_loc / np.maximum(dx * dx + dy * dy, 1e-6)
        steer = np.clip(np.arctan(p.wheelbase_m * k_cmd) / p.max_steer_rad, -1.0, 1.0)

        v_tgt = self.profile_speed(
            sp + scripted.SPEED_LEAD_STEPS * np.maximum(v, V_FLOOR) * self.dt
        )
        tyre = (v_tgt - v) / self.dt + p.drag_coeff * v * v
        a_drive = np.minimum(p.power_w / (p.mass_kg * np.maximum(v, V_FLOOR)), p.max_drive_g * G)
        a_brake = np.minimum(p.max_brake_g * G, G * (p.grip_base_g + p.grip_aero_coeff * v * v))
        pedal = np.clip(np.where(tyre >= 0.0, tyre / a_drive, tyre / a_brake), -1.0, 1.0)
        return np.column_stack([steer, pedal])


@dataclass(frozen=True)
class Rollout:
    obs: np.ndarray  # (k, OBS_SIZE) float32, every decision of every car still driving
    actions: np.ndarray  # (k, 2) the teacher's action in those states
    crash_rate: float  # share of cars that went off track, wrong way or stalled
    mean_progress_m: float


def collect(
    world: World,
    teacher: Teacher,
    start_s: np.ndarray,
    start_v: np.ndarray,
    duration_s: float,
    *,
    spec: pol.PolicySpec | None = None,
    params: np.ndarray | None = None,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    obs_fn: Callable[[World, np.ndarray], np.ndarray] | None = None,
) -> Rollout:
    """Drive every car of ``world`` from ``start_s``/``start_v`` for ``duration_s`` and label
    each visited state with the teacher's action.

    Each car executes the teacher's action with probability ``beta`` per episode and the
    student's (``spec``/``params``) otherwise; with ``beta = 1`` (or no student) it's plain
    behaviour cloning. ``obs_fn(world, obs)`` replaces the World's observations (to test
    extra inputs).
    """
    n = world.n_cars
    obs_fn = obs_fn or (lambda _w, o: o)
    obs = obs_fn(world, world.reset(start_s=start_s, start_v=start_v))
    teacher.reset_hints()
    rng = rng if rng is not None else np.random.default_rng(0)
    use_teacher = np.ones(n, dtype=bool)
    if params is not None and beta < 1.0:
        use_teacher = rng.random(n) < beta
    decision_dt = world.config.dt * world.config.action_repeat
    obs_out, act_out = [], []
    st, car = world.progress.state, world.car
    for _ in range(int(math.ceil(duration_s / decision_dt))):
        live = ~st.done
        if not live.any():
            break
        a_teacher = teacher.act(car.x, car.y, car.yaw, car.v)
        obs_out.append(obs[live])
        act_out.append(a_teacher[live])
        a = a_teacher
        if not use_teacher.all():
            a = np.where(use_teacher[:, None], a_teacher, pol.act(spec, params, obs))
        obs = obs_fn(world, world.step(a).obs)
    crashed = np.isin(st.done_reason, [1, 2, 3])
    return Rollout(
        obs=np.concatenate(obs_out).astype(np.float32),
        actions=np.concatenate(act_out),
        crash_rate=float(crashed.mean()),
        mean_progress_m=float(st.progress.mean()),
    )


# ------------------------------------------------------------------------------- fitting


def _forward(spec: pol.PolicySpec, layers, x: np.ndarray):
    """Activations of every layer (inputs first) for a batch ``x``."""
    acts = [x]
    h = x
    for W, b in layers:
        h = np.tanh(h @ W + b)
        acts.append(h)
    return acts


def fit(
    spec: pol.PolicySpec,
    obs: np.ndarray,
    actions: np.ndarray,
    init: np.ndarray,
    *,
    epochs: int = 30,
    batch: int = 2048,
    lr: float = 3e-3,
    rng: np.random.Generator | None = None,
    log: Callable[[str], object] | None = None,
) -> np.ndarray:
    """Minibatch Adam on the weighted mean squared action error; returns ``(n_params,)``.

    Only ``tanh`` hidden layers are supported (the policy default). The steering error is
    weighted ``STEER_WEIGHT``. The learning rate decays linearly to 10% over the epochs.
    """
    if spec.activation != "tanh":
        raise ValueError("fit supports tanh hidden layers only")
    rng = rng if rng is not None else np.random.default_rng(0)
    theta = np.asarray(init, dtype=np.float64).copy()
    x_all = np.asarray(obs, dtype=np.float64)
    y_all = np.asarray(actions, dtype=np.float64)
    w = np.array([STEER_WEIGHT, 1.0]) / (STEER_WEIGHT + 1.0)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = x_all.shape[0]
    for epoch in range(epochs):
        step_lr = lr * (1.0 - 0.9 * epoch / max(epochs - 1, 1))
        order = rng.permutation(n)
        for k in range(0, n, batch):
            sel = order[k : k + batch]
            x, y = x_all[sel], y_all[sel]
            layers = pol.unflatten(spec, theta)
            acts = _forward(spec, layers, x)
            grad = np.zeros_like(theta)
            g_layers = pol.unflatten(spec, grad)  # views into grad
            delta = 2.0 * w * (acts[-1] - y) / x.shape[0]
            for li in range(len(layers) - 1, -1, -1):
                delta = delta * (1.0 - acts[li + 1] ** 2)  # through tanh
                g_layers[li][0][...] = acts[li].T @ delta
                g_layers[li][1][...] = delta.sum(axis=0)
                delta = delta @ layers[li][0].T
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            theta -= step_lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        if log is not None:
            log(f"  epoch {epoch + 1:3d}: loss {loss(spec, theta, x_all, y_all):.5f}")
    return theta.astype(np.float32)


def loss(spec: pol.PolicySpec, params: np.ndarray, obs: np.ndarray, actions: np.ndarray) -> float:
    """The weighted mean squared action error :func:`fit` minimises."""
    w = np.array([STEER_WEIGHT, 1.0]) / (STEER_WEIGHT + 1.0)
    err = pol.act(spec, np.asarray(params, dtype=np.float64), obs) - actions
    return float(np.mean(np.sum(w * err * err, axis=1)))
