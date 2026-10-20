"""Headless neuroevolution training (docs/m3-interfaces.md section K, revamp-plan section 7).

One generation: the optimiser asks for ``population`` genomes, :meth:`Trainer.evaluate` drives
them in a vectorised :class:`~f1sim.sim.world.World`, the optimiser is told the fitnesses and
the run folder (:mod:`f1sim.train.metrics`) gets a metrics row, checkpoints and the best lap.

**Curriculum.**

- *Stage 1* (random starts). One World of ``N = population * episodes_per_genome`` cars; car
  ``i`` drives genome ``i // episodes_per_genome``. Starts are ``s0 ~ U(0, L)`` and
  ``v0 ~ U(20, 60)`` m/s, the episode lasts ``stage1_episode_s``. Car fitness is its progress
  in metres; a crash (OFF_TRACK, WRONG_WAY or STALL) costs 10% of the distance driven, i.e.
  ``progress - 0.1 * max(progress, 0)`` (a car that only went backwards keeps its negative
  progress). Genome fitness is the mean over its episodes.
- *Promotion* to stage 2 after the first generation whose best per-genome mean progress
  (``best_progress_m``) reaches ``stage1_promote_progress_m`` (0 = auto,
  ``0.8 * stage1_episode_s * 40``). The ``stage`` column of ``metrics.csv`` logs it.
- *Stage 2* (a full lap). Every car starts ``stage2_lead_in_m`` before the line, on the centre
  line, at ``min(60, v_profile)`` m/s where ``v_profile`` is :func:`scripted.speed_profile` of
  the centre line at that point (the car starts on the centre line, so that's the speed it can
  actually carry there; the 60 m/s cap gives an untrained policy a fair entry). The episode
  cap is ``stage2_time_factor * T_ref``, counted from the reset. The first line crossing only
  starts the timed lap (``ProgressTracker`` counts line-to-line laps). A car is *finished* at
  its first completed lap: its lap time is kept and later events are ignored (the world keeps
  it driving harmlessly). Fitness is progress (with the crash penalty) for unfinished cars and
  ``L + 1000 + 50 * (stage2_time_factor * T_ref - lap_time)`` for finished ones.

  Stage-2 starts have no randomness, so the ``episodes_per_genome`` copies of a genome would
  drive identical episodes. Stage 2 therefore runs a second World with one car per genome
  (car ``i`` = genome ``i``): the fitness is the same as the mean over identical copies, at
  ``1 / episodes_per_genome`` of the cost.

The world stops stepping as soon as every car is done (or finished, in stage 2).

**T_ref** is the lap time of :func:`scripted.drive_reference_lap` along the circuit's TUM
raceline (catalogue entry + :mod:`f1sim.data.tum_source`; downloaded once into the cache),
computed once. If the track isn't in the catalogue, has no TUM raceline, the raceline can't be
fetched, or the raceline lap doesn't complete cleanly, the centre line is used instead. The
``reference_lap_s`` argument overrides it. ``config.toml`` records which one was used.

**Determinism.** A single ``numpy.random.Generator(seed)`` draws the optimiser seed first and
then every stage-1 start, so the same seed gives the same ``metrics.csv`` apart from
``wall_s`` (and the number of rows when ``max_wall_s`` stops the run).

The policy, optimiser and checkpoint modules (``f1sim.agents.policy``, ``.optimisers``,
``.checkpoint``) are imported when a :class:`Trainer` is created, so this module imports
without them.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import logging
import math
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from f1sim import __version__
from f1sim.sim.car_model import CarParams
from f1sim.sim.progress import DoneReason
from f1sim.sim.sensors import OBS_SPEC_VERSION
from f1sim.sim.world import World, WorldConfig
from f1sim.track.track import Track
from f1sim.train.metrics import RunWriter

log = logging.getLogger(__name__)

STAGE1_V_RANGE = (20.0, 60.0)
"""Stage-1 start speeds, m/s (uniform)."""
AUTO_PROMOTE_SPEED_M_S = 40.0
AUTO_PROMOTE_FRACTION = 0.8
"""Auto promotion threshold: ``0.8 * stage1_episode_s * 40 m/s``."""
STAGE2_MAX_START_V = 60.0
CRASH_PENALTY = 0.1
LAP_BONUS = 1000.0
LAP_TIME_WEIGHT = 50.0
_CRASH_REASONS = np.array(
    [DoneReason.OFF_TRACK, DoneReason.WRONG_WAY, DoneReason.STALL], dtype=np.int8
)
RUN_ONLY_TABLES = ("resolved", "car", "run")
"""Tables a run folder's ``config.toml`` adds; :func:`load_train_config` skips them."""


# ---------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrainConfig:
    track_id: str = "silverstone"
    seed: int = 0
    optimiser: str = "ga"
    population: int = 64
    episodes_per_genome: int = 3
    hidden: tuple[int, ...] = (64, 64)
    sigma0: float = 0.1
    max_generations: int = 1000
    max_wall_s: float = 1800.0
    checkpoint_every: int = 10
    # curriculum
    stage1_episode_s: float = 30.0  # random starts, fitness = progress
    stage1_promote_progress_m: float = 0.0  # 0 = auto: 0.8 * (stage1_episode_s * 40 m/s)
    stage2_lead_in_m: float = 80.0  # start this far before the line at reference-profile speed
    stage2_time_factor: float = 1.5  # episode cap = factor * T_ref

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(int(h) for h in self.hidden))
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError(f"hidden must be one or more positive sizes, got {self.hidden}")
        if self.population < 1 or self.episodes_per_genome < 1:
            raise ValueError("population and episodes_per_genome must be >= 1")
        if self.max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0 (0 = only on improvement)")
        if not self.sigma0 > 0:
            raise ValueError("sigma0 must be > 0")
        if not self.max_wall_s >= 0:
            raise ValueError("max_wall_s must be >= 0")
        if not self.stage1_episode_s > 0:
            raise ValueError("stage1_episode_s must be > 0")
        if not self.stage1_promote_progress_m >= 0:
            raise ValueError("stage1_promote_progress_m must be >= 0 (0 = auto)")
        if not self.stage2_lead_in_m >= 0:
            raise ValueError("stage2_lead_in_m must be >= 0")
        if not self.stage2_time_factor > 0:
            raise ValueError("stage2_time_factor must be > 0")

    @property
    def promote_progress_m(self) -> float:
        """The stage-1 promotion threshold actually used (resolves 0 = auto)."""
        if self.stage1_promote_progress_m > 0:
            return float(self.stage1_promote_progress_m)
        return AUTO_PROMOTE_FRACTION * self.stage1_episode_s * AUTO_PROMOTE_SPEED_M_S


def _check_type(path: Path, key: str, value, default):
    """Coerce a TOML value to the type of the field's default, or raise ValueError."""
    if isinstance(default, tuple):
        if not isinstance(value, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise ValueError(f"{path}: {key} must be a list of integers, got {value!r}")
        return tuple(value)
    if isinstance(default, bool):  # (no bool fields today; keep bools out of int/float)
        if not isinstance(value, bool):
            raise ValueError(f"{path}: {key} must be true or false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: {key} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: {key} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string, got {value!r}")
        return value
    raise TypeError(f"unsupported config field type for {key}")  # pragma: no cover


def load_train_config(path: Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a flat TOML file (``configs/train_default.toml``).

    Keys are field names; missing keys keep their defaults. Unknown keys and values of the
    wrong type raise ``ValueError``; ``hidden`` is a list of integers. A run folder's
    ``config.toml`` loads too: its extra tables (:data:`RUN_ONLY_TABLES`) are skipped.
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    defaults = {f.name: f.default for f in fields(TrainConfig)}
    values = {}
    unknown = []
    for key, value in data.items():
        if key in RUN_ONLY_TABLES and isinstance(value, dict):
            continue
        if key not in defaults:
            unknown.append(key)
            continue
        values[key] = _check_type(path, key, value, defaults[key])
    if unknown:
        raise ValueError(f"{path}: unknown training option(s): {', '.join(sorted(unknown))}")
    return TrainConfig(**values)


# ---------------------------------------------------------------------------------- result


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    generations: int
    stage: int
    best_fitness: float
    best_lap_s: float | None
    first_lap_wall_s: float | None


# ---------------------------------------------------------------------------------- helpers


def _git_commit() -> str:
    """Commit of the working tree (``"unknown"`` outside git), for checkpoint metadata."""
    try:
        checkpoint = importlib.import_module("f1sim.agents.checkpoint")
        fn = getattr(checkpoint, "current_git_commit", None)
        if fn is not None:
            return str(fn())
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def reference_lap_time(track: Track, car: CarParams) -> tuple[float, str]:
    """``(T_ref, source)``: the scripted lap on the TUM raceline, else on the centre line.

    ``source`` is ``"tum-raceline"`` or ``"centre-line"``. Raises ``ValueError`` if neither
    lap completes cleanly.
    """
    from f1sim.data import catalog, tum_source
    from f1sim.sim.scripted import drive_reference_lap

    raceline = None
    try:
        tum_name = catalog.get(track.data.id).tum_name
        if tum_name is not None:
            raceline = tum_source.parse_tum_raceline(tum_source.fetch_tum_raceline(tum_name))
    except (KeyError, ValueError, OSError) as exc:
        log.info("no TUM raceline for %s (%s); using the centre line", track.data.id, exc)
        raceline = None
    if raceline is not None:
        lap = drive_reference_lap(track, car, raceline_xy=raceline)
        if lap.completed and math.isfinite(lap.lap_time_s):
            return float(lap.lap_time_s), "tum-raceline"
        log.warning("reference lap on the TUM raceline failed; using the centre line")
    lap = drive_reference_lap(track, car)
    if not (lap.completed and math.isfinite(lap.lap_time_s)):
        raise ValueError(
            f"{track.data.id}: the scripted reference lap didn't complete cleanly on the TUM "
            "raceline or the centre line; pass reference_lap_s"
        )
    return float(lap.lap_time_s), "centre-line"


def stage2_start(track: Track, car: CarParams, lead_in_m: float) -> tuple[float, float]:
    """``(s0, v0)`` of the stage-2 start: ``lead_in_m`` before the line, ``min(60, profile)``."""
    from f1sim.sim.scripted import speed_profile

    d = track.data
    length = track.length
    s0 = float((-lead_in_m) % length)
    v = speed_profile(d.x, d.y, car, closed=True)
    v_at = float(np.interp(s0, np.append(d.s, length), np.append(v, v[0])))
    return s0, min(STAGE2_MAX_START_V, v_at)


def _crash_mask(done_reason: np.ndarray) -> np.ndarray:
    return np.isin(done_reason, _CRASH_REASONS)


def _penalised(progress: np.ndarray, crashed: np.ndarray) -> np.ndarray:
    return np.where(crashed, progress - CRASH_PENALTY * np.maximum(progress, 0.0), progress)


# --------------------------------------------------------------------------------- trainer


class Trainer:
    """Curriculum neuroevolution on one track; see the module docstring.

    ``init`` (an addition to the contract) is passed to ``make_optimiser(..., init=init)``:
    rows (``(population, n_params)``) the optimiser's first population starts from, e.g. a
    hand-made policy or a checkpoint's parameters.
    """

    def __init__(
        self,
        track: Track,
        config: TrainConfig,
        run_dir: Path,
        *,
        car: CarParams = CarParams(),  # noqa: B008 (frozen dataclass, as in the contract)
        reference_lap_s: float | None = None,
        init: np.ndarray | None = None,
    ):
        self._policy = importlib.import_module("f1sim.agents.policy")
        optimisers = importlib.import_module("f1sim.agents.optimisers")
        self._checkpoint = importlib.import_module("f1sim.agents.checkpoint")

        self.track = track
        self.config = config
        self.car = car
        self.run_dir = Path(run_dir)
        self.spec = self._policy.PolicySpec(hidden=tuple(config.hidden))
        self.population = int(config.population)
        self.episodes = int(config.episodes_per_genome)

        if reference_lap_s is not None:
            if not (math.isfinite(reference_lap_s) and reference_lap_s > 0):
                raise ValueError(f"reference_lap_s must be > 0, got {reference_lap_s!r}")
            self.t_ref_s, self.t_ref_source = float(reference_lap_s), "override"
        else:
            self.t_ref_s, self.t_ref_source = reference_lap_time(track, car)
        self.stage2_cap_s = config.stage2_time_factor * self.t_ref_s
        self.stage2_start_s, self.stage2_start_v = stage2_start(track, car, config.stage2_lead_in_m)
        self.promote_progress_m = config.promote_progress_m

        self._rng = np.random.default_rng(config.seed)
        self.optimiser_seed = int(self._rng.integers(0, 2**31 - 1))
        kwargs = {
            "population": self.population,
            "sigma0": float(config.sigma0),
            "seed": self.optimiser_seed,
        }
        if init is not None:
            kwargs["init"] = np.asarray(init, dtype=np.float32)
        self.optimiser = optimisers.make_optimiser(config.optimiser, self.spec.n_params, **kwargs)

        base = WorldConfig()
        self._dt = base.dt
        self._action_repeat = base.action_repeat
        self._progress_params = base.progress
        self._world1 = World(
            track,
            WorldConfig(
                n_cars=self.population * self.episodes,
                episode_time_s=config.stage1_episode_s,
                car=car,
            ),
        )
        self._world2: World | None = None  # built on promotion

        self.stage = 1
        self.writer = RunWriter(self.run_dir)
        self.git_commit = _git_commit()
        self.best_lap_s: float | None = None
        self.first_lap_wall_s: float | None = None
        self._stage_best_fitness = -math.inf
        self._stage_best_params: np.ndarray | None = None
        self._prev_stage_best_fitness = -math.inf

    # ----------------------------------------------------------------------- evaluation

    def _stage2_world(self) -> World:
        if self._world2 is None:
            self._world2 = World(
                self.track,
                WorldConfig(n_cars=self.population, episode_time_s=self.stage2_cap_s, car=self.car),
            )
        return self._world2

    def evaluate(self, params: np.ndarray, stage: int) -> tuple[np.ndarray, dict]:
        """Fitness ``(P,)`` of ``params`` ``(P, n_params)`` in ``stage`` (1 or 2), and stats.

        ``stats``: ``progress_m`` (P,) mean raw progress, ``lap_time_s`` (P,) nan where no lap
        (stage 2 only), ``best_progress_m``, ``completion_rate`` (share of episodes with a
        completed lap), ``best_lap_s`` (nan if none; stage 2 only), ``crash_rate`` and
        ``sim_s`` (simulated seconds until every car was done).
        """
        params = np.asarray(params, dtype=np.float32)
        if params.ndim != 2 or params.shape[0] != self.population:
            raise ValueError(f"params must be ({self.population}, n_params), got {params.shape}")
        if stage == 1:
            return self._evaluate_stage1(params)
        if stage == 2:
            return self._evaluate_stage2(params)
        raise ValueError(f"stage must be 1 or 2, got {stage!r}")

    def _drive(self, world: World, params_per_car: np.ndarray, obs, finished=None):
        """Step until every car is done (or finished, if a mask is given); returns sim time."""
        n_steps = 0
        state = world.progress.state
        decision_dt = self._dt * self._action_repeat
        max_steps = int(math.ceil(world.config.episode_time_s / decision_dt)) + 2
        laps = []
        while n_steps < max_steps:
            stop = state.done if finished is None else (state.done | finished)
            if stop.all():
                break
            actions = self._policy.act(self.spec, params_per_car, obs)
            res = world.step(actions)
            obs = res.obs
            n_steps += 1
            if finished is not None and res.lap_completed.any():
                newly = res.lap_completed & ~finished
                if newly.any():
                    laps.append((newly, res.lap_time.copy(), state.progress.copy()))
                    finished |= newly
        return n_steps * decision_dt, laps

    def _evaluate_stage1(self, params: np.ndarray) -> tuple[np.ndarray, dict]:
        world = self._world1
        n, k, p = world.n_cars, self.episodes, self.population
        start_s = self._rng.uniform(0.0, self.track.length, n)
        start_v = self._rng.uniform(*STAGE1_V_RANGE, n)
        obs = world.reset(start_s=start_s, start_v=start_v)
        per_car = np.repeat(params, k, axis=0)  # car i drives genome i // k
        sim_s, _ = self._drive(world, per_car, obs)
        st = world.progress.state
        progress = st.progress.copy()
        crashed = _crash_mask(st.done_reason)
        fit = _penalised(progress, crashed).reshape(p, k).mean(axis=1)
        prog_g = progress.reshape(p, k).mean(axis=1)
        stats = {
            "progress_m": prog_g,
            "lap_time_s": np.full(p, np.nan),
            "best_progress_m": float(prog_g.max()),
            "completion_rate": float(np.mean(st.lap > 0)),
            "best_lap_s": math.nan,
            "crash_rate": float(crashed.mean()),
            "sim_s": sim_s,
        }
        return fit, stats

    def _evaluate_stage2(self, params: np.ndarray) -> tuple[np.ndarray, dict]:
        world = self._stage2_world()
        p = self.population
        obs = world.reset(start_s=self.stage2_start_s, start_v=self.stage2_start_v)
        finished = np.zeros(p, dtype=bool)
        sim_s, laps = self._drive(world, params, obs, finished)
        st = world.progress.state
        lap_time = np.full(p, np.nan)
        progress = st.progress.copy()
        for newly, lt, prog in laps:
            lap_time[newly] = lt[newly]
            progress[newly] = prog[newly]
        crashed = _crash_mask(st.done_reason) & ~finished
        fit = _penalised(progress, crashed)
        lap_fit = (
            self.track.length
            + LAP_BONUS
            + LAP_TIME_WEIGHT * (self.stage2_cap_s - np.where(finished, lap_time, 0.0))
        )
        fit = np.where(finished, lap_fit, fit)
        stats = {
            "progress_m": progress,
            "lap_time_s": lap_time,
            "best_progress_m": float(progress.max()),
            "completion_rate": float(finished.mean()),
            "best_lap_s": float(np.nanmin(lap_time)) if finished.any() else math.nan,
            "crash_rate": float(crashed.mean()),
            "sim_s": sim_s,
        }
        return fit, stats

    # ----------------------------------------------------------------------- best lap

    def record_lap(self, genome: np.ndarray) -> dict[str, np.ndarray] | None:
        """Re-run ``genome`` alone from the stage-2 start and return its timed lap's trace.

        The car is stepped one physics step at a time (``action_repeat = 1``) with each action
        held for the training ``action_repeat`` steps, which reproduces the batched episode
        exactly. Columns are :data:`metrics.TRACE_COLUMNS`: ``t`` from the line crossing that
        started the lap, ``s`` unwrapped from 0 to the track length, ``speed`` in m/s,
        ``throttle = max(pedal, 0)``, ``brake = pedal < 0`` and ``gear = 0``. The first and
        last samples are interpolated onto the line crossings, so ``t[-1] - t[0]`` is the lap
        time. Returns None if the re-run doesn't complete a lap.
        """
        world = World(
            self.track,
            WorldConfig(
                n_cars=1,
                dt=self._dt,
                action_repeat=1,
                episode_time_s=self.stage2_cap_s,
                car=self.car,
                progress=self._progress_params,
            ),
        )
        genome = np.asarray(genome, dtype=np.float32).reshape(1, -1)
        obs = world.reset(start_s=self.stage2_start_s, start_v=self.stage2_start_v)
        st, car = world.progress.state, world.car
        cols = {c: [] for c in ("t", "x", "y", "p", "v", "pedal")}

        def sample(pedal: float) -> None:
            cols["t"].append(float(st.t[0]))
            cols["x"].append(float(car.x[0]))
            cols["y"].append(float(car.y[0]))
            cols["p"].append(float(st.progress[0]))
            cols["v"].append(float(car.v[0]))
            cols["pedal"].append(pedal)

        action = None
        max_steps = int(math.ceil(self.stage2_cap_s / self._dt)) + 2
        lap = None
        for k in range(max_steps):
            if k % self._action_repeat == 0:
                action = np.asarray(self._policy.act(self.spec, genome, obs), dtype=np.float64)
                if k == 0:
                    sample(float(action[0, 1]))
            start_t, start_p = float(st.lap_start_t[0]), float(st.progress_at_lap_start[0])
            res = world.step(action)
            obs = res.obs
            sample(float(np.clip(action[0, 1], -1.0, 1.0)))
            if res.lap_completed[0]:
                end_t, end_p = float(st.lap_start_t[0]), float(st.progress_at_lap_start[0])
                lap = (start_t, start_p, end_t, end_p)
                break
            if res.done[0]:
                return None
        if lap is None:
            return None
        start_t, start_p, end_t, end_p = lap
        a = {k: np.asarray(v) for k, v in cols.items()}
        inside = (a["t"] > start_t) & (a["t"] < end_t)
        # Pedal on the steps that crossed the line: the first sample at or after each crossing.
        i_start = min(int(np.searchsorted(a["t"], start_t, side="left")), a["t"].size - 1)
        i_end = min(int(np.searchsorted(a["t"], end_t, side="left")), a["t"].size - 1)
        t_full = np.concatenate([[start_t], a["t"][inside], [end_t]])

        def edge(name: str) -> np.ndarray:
            return np.interp([start_t, end_t], a["t"], a[name])

        def col(name: str) -> np.ndarray:
            e = edge(name)
            return np.concatenate([[e[0]], a[name][inside], [e[1]]])

        pedal = np.concatenate([[a["pedal"][i_start]], a["pedal"][inside], [a["pedal"][i_end]]])
        s = np.concatenate([[0.0], a["p"][inside] - start_p, [end_p - start_p]])
        return {
            "t": t_full - start_t,
            "x": col("x"),
            "y": col("y"),
            "s": np.clip(s, 0.0, end_p - start_p),
            "speed": col("v"),
            "throttle": np.maximum(pedal, 0.0),
            "brake": (pedal < 0.0).astype(np.float64),
            "gear": np.zeros(t_full.size, dtype=np.int64),
        }

    # ----------------------------------------------------------------------------- run

    def resolved_config(self) -> dict:
        """The dict written to ``config.toml``: the config plus what was derived from it."""
        cfg = asdict(self.config)
        cfg["hidden"] = list(self.config.hidden)
        cfg["resolved"] = {
            "t_ref_s": self.t_ref_s,
            "t_ref_source": self.t_ref_source,
            "stage1_promote_progress_m": self.promote_progress_m,
            "stage2_start_s": self.stage2_start_s,
            "stage2_start_v": self.stage2_start_v,
            "stage2_episode_s": self.stage2_cap_s,
            "optimiser_seed": self.optimiser_seed,
            "n_params": int(self.spec.n_params),
            "dt": self._dt,
            "action_repeat": self._action_repeat,
        }
        cfg["car"] = asdict(self.car)
        cfg["run"] = {
            "f1sim": __version__,
            "git_commit": self.git_commit,
            "obs_spec_version": OBS_SPEC_VERSION,
            "track_name": self.track.data.name,
            "track_length_m": float(self.track.length),
        }
        return cfg

    def _meta(self, gen: int, fitness: float) -> dict:
        return {
            "obs_spec_version": OBS_SPEC_VERSION,
            "policy": self.spec.to_dict(),
            "car_params_hash": self._checkpoint.car_params_hash(self.car),
            "track_id": self.track.data.id,
            "seed": int(self.config.seed),
            "generation": int(gen),
            "fitness": float(fitness),
            "best_lap_s": self.best_lap_s,
            "git_commit": self.git_commit,
            "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "stage": int(self.stage),
        }

    def _save_checkpoint(self, gen: int) -> None:
        if self._stage_best_params is None:
            return
        self.writer.save_checkpoint(
            gen,
            self._stage_best_params,
            self._meta(gen, self._stage_best_fitness),
            self.optimiser.state_dict(),
        )

    def run(self, on_generation: Callable[[dict], object] | None = None) -> TrainResult:
        """Train until ``max_generations`` or ``max_wall_s``; see the module docstring.

        Each generation appends a ``metrics.csv`` row (per-generation values; ``best_lap_s``
        is the generation's fastest lap) and calls ``on_generation`` with that row as a dict.
        Checkpoints hold the best genome of the current stage: every ``checkpoint_every``
        generations, whenever it improves, and after the last generation.
        """
        cfg = self.config
        t0 = time.perf_counter()
        self.writer.write_config(self.resolved_config())
        gen = 0
        saved_gen = -1
        while gen < cfg.max_generations:
            stage = self.stage
            params = np.asarray(self.optimiser.ask(), dtype=np.float32)
            fitness, stats = self.evaluate(params, stage)
            self.optimiser.tell(fitness)
            wall = time.perf_counter() - t0

            if stage == 2 and math.isfinite(stats["best_lap_s"]):
                if self.first_lap_wall_s is None:
                    self.first_lap_wall_s = wall
                    log.info("gen %d: first completed lap after %.1f s", gen, wall)
                if self.best_lap_s is None or stats["best_lap_s"] < self.best_lap_s:
                    self.best_lap_s = float(stats["best_lap_s"])
                    trace = self.record_lap(params[int(np.nanargmin(stats["lap_time_s"]))])
                    if trace is not None:
                        self.writer.write_best_lap(trace)
                    else:
                        log.warning("gen %d: the best lap didn't reproduce on re-run", gen)

            i_best = int(np.argmax(fitness))
            improved = float(fitness[i_best]) > self._stage_best_fitness
            if improved:
                self._stage_best_fitness = float(fitness[i_best])
                self._stage_best_params = params[i_best].copy()

            row = {
                "gen": gen,
                "stage": stage,
                "wall_s": wall,
                "best_fitness": float(fitness[i_best]),
                "mean_fitness": float(np.mean(fitness)),
                "completion_rate": stats["completion_rate"],
                "best_lap_s": stats["best_lap_s"],
                "best_progress_m": stats["best_progress_m"],
                "sigma": float(self.optimiser.sigma),
            }
            self.writer.append_metrics(row)
            if improved or (cfg.checkpoint_every > 0 and gen % cfg.checkpoint_every == 0):
                self._save_checkpoint(gen)
                saved_gen = gen
            if on_generation is not None:
                on_generation(dict(row))

            if stage == 1 and stats["best_progress_m"] >= self.promote_progress_m:
                log.info(
                    "gen %d: best progress %.0f m >= %.0f m, promoting to stage 2",
                    gen,
                    stats["best_progress_m"],
                    self.promote_progress_m,
                )
                self.stage = 2
                self._prev_stage_best_fitness = self._stage_best_fitness
                self._stage_best_fitness = -math.inf
                self._stage_best_params = None
            gen += 1
            if time.perf_counter() - t0 >= cfg.max_wall_s:
                break

        if saved_gen != gen - 1 and self._stage_best_params is not None:
            self._save_checkpoint(gen - 1)
        return TrainResult(
            run_dir=self.run_dir,
            generations=gen,
            stage=self.stage,
            best_fitness=(
                self._stage_best_fitness
                if self._stage_best_params is not None
                else self._prev_stage_best_fitness
            ),
            best_lap_s=self.best_lap_s,
            first_lap_wall_s=self.first_lap_wall_s,
        )


__all__ = [
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "load_train_config",
    "reference_lap_time",
    "stage2_start",
]
