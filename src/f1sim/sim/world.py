"""N cars on one track: car physics + progress/laps/termination + observations.

One :meth:`World.step` applies each car's action for ``action_repeat`` physics steps of ``dt``
(30 Hz decisions on 60 Hz physics by default) and returns the observations for the next
decision (docs/data-formats.md section 6) plus what happened during the step.

Cars that are done (off track, wrong way, stall, or out of time) are frozen in place until
they're reset with a mask. ``World`` holds no random state: start positions come from the
caller, so a seeded caller gives reproducible runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from f1sim.sim import car_model
from f1sim.sim.car_model import CarParams, CarState
from f1sim.sim.progress import DoneReason, ProgressParams, ProgressTracker
from f1sim.sim.sensors import Sensors
from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track


@dataclass(frozen=True)
class WorldConfig:
    n_cars: int = 64
    dt: float = 1.0 / 60.0
    action_repeat: int = 2
    episode_time_s: float = 180.0
    # Drive on the track limits (where real cars go) rather than the white line, when the
    # track has them (data/track_limits.py)
    use_track_limits: bool = True
    car: CarParams = field(default_factory=CarParams)
    progress: ProgressParams = field(default_factory=ProgressParams)


@dataclass(frozen=True)
class StepResult:
    """What one decision step produced, per car (shape (N,) unless noted)."""

    obs: np.ndarray  # (N, OBS_SIZE) float32, for the next decision
    progress_delta: np.ndarray  # metres along the track gained this step (negative = backwards)
    done: np.ndarray  # bool, done after this step
    newly_done: np.ndarray  # bool, became done during this step
    done_reason: np.ndarray  # int8 DoneReason
    lap_completed: np.ndarray  # bool
    lap_time: np.ndarray  # float, nan where no lap completed
    sliding: np.ndarray  # bool, grip limit exceeded on any physics step


class World:
    def __init__(self, track: Track, config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        self.track = track.sim_track() if self.config.use_track_limits else track
        n = self.config.n_cars
        if n < 1:
            raise ValueError(f"n_cars must be >= 1, got {n}")
        self.grid = SegmentGrid.from_track(self.track)
        self.sensors = Sensors(self.track, self.grid, car=self.config.car)
        self.progress = ProgressTracker(self.track, n, self.config.progress)
        self.car = CarState.at(n, 0.0, 0.0, 0.0)
        self.prev_action = np.zeros((n, 2), dtype=np.float64)

    @property
    def n_cars(self) -> int:
        return self.config.n_cars

    def reset(self, start_s=0.0, start_v=0.0, mask=None) -> np.ndarray:
        """Place cars on the centre line at ``start_s`` heading along the track at ``start_v``.

        ``start_s`` and ``start_v`` are scalars or (N,) arrays; ``mask`` selects which cars to
        reset (all if None). Returns the observations of all cars.
        """
        n = self.n_cars
        m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        s = np.broadcast_to(np.asarray(start_s, dtype=np.float64), (n,))[m] % self.track.length
        v = np.broadcast_to(np.asarray(start_v, dtype=np.float64), (n,))[m]
        p = self.track.point_at(s)
        # Write into the existing arrays so references to them stay valid
        self.car.x[m] = p[:, 0]
        self.car.y[m] = p[:, 1]
        self.car.yaw[m] = self.track.heading_at(s)
        self.car.v[m] = v
        self.car.delta[m] = 0.0
        self.prev_action[m] = 0.0
        self.progress.reset(self.car.x, self.car.y, mask=m)
        self.sensors.forget(None if mask is None else m)
        return self._observe()

    def step(self, actions) -> StepResult:
        """Apply ``actions`` (N, 2) = (steer, pedal) in [-1, 1] for ``action_repeat`` steps."""
        cfg = self.config
        n = self.n_cars
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(n, 2), -1.0, 1.0)
        state = self.progress.state
        progress_before = state.progress.copy()
        newly_done = np.zeros(n, dtype=bool)
        lap_completed = np.zeros(n, dtype=bool)
        lap_time = np.full(n, np.nan)
        sliding = np.zeros(n, dtype=bool)

        for _ in range(cfg.action_repeat):
            frozen = state.done.copy()
            if frozen.any():
                saved = [a[frozen].copy() for a in self._car_arrays()]
            info = car_model.step(self.car, actions[:, 0], actions[:, 1], cfg.car, cfg.dt)
            if frozen.any():
                for arr, keep in zip(self._car_arrays(), saved, strict=True):
                    arr[frozen] = keep
            sliding |= info.sliding & ~frozen

            events = self.progress.update(self.car.x, self.car.y, cfg.dt)
            newly_done |= events.newly_done
            lap_completed |= events.lap_completed
            lap_time = np.where(events.lap_completed, events.lap_time, lap_time)

            out_of_time = ~state.done & (state.t >= cfg.episode_time_s - 1e-9)
            if out_of_time.any():
                state.done[out_of_time] = True
                state.done_reason[out_of_time] = DoneReason.TIME_LIMIT
                newly_done |= out_of_time

        self.prev_action[:] = actions
        return StepResult(
            obs=self._observe(),
            progress_delta=state.progress - progress_before,
            done=state.done.copy(),
            newly_done=newly_done,
            done_reason=state.done_reason.copy(),
            lap_completed=lap_completed,
            lap_time=lap_time,
            sliding=sliding,
        )

    def _car_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.car.x, self.car.y, self.car.yaw, self.car.v, self.car.delta)

    def _observe(self) -> np.ndarray:
        st = self.progress.state
        return self.sensors.observe(
            self.car.x, self.car.y, self.car.yaw, self.car.v, st.s, st.n, self.prev_action
        )
