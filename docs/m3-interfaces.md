# M3 interface contract

M3 is headless neuroevolution training (revamp-plan section 7), plus the ghost and replay pieces
that make results visible. It's split into five owned pieces built in parallel. The main session
then wires `f1sim train`, `f1sim replay` and `f1sim ghost`, and runs the M3 acceptance check.

The rules from [m1-interfaces.md](m1-interfaces.md) and [m2-interfaces.md](m2-interfaces.md)
still apply:
- **Conventions:** SI units, `n` positive to the left.
- **Ownership:** don't edit another owner's files, and keep `__init__.py` files empty.
- **Tests:** offline by default; network tests marked `@pytest.mark.network`.
- **Lint:** ruff check and ruff format clean.

**Branch base.** Your worktree may start from an old commit. First run `git merge --ff-only main`,
and confirm this file exists.

**Cache location.** Processes started by the agent tools run inside the Claude desktop app's
package, so Windows redirects writes under `%LOCALAPPDATA%` to
`%LOCALAPPDATA%\Packages\<Claude package>\LocalCache\Local`. That's fine for your work:
`load_track("silverstone")` finds the built tracks there. Just don't hard-code cache paths; use
`f1sim.paths`.

## What already exists (read, don't modify unless your section says so)

- **`sim.world`:**
  - `WorldConfig(n_cars, dt=1/60, action_repeat=2, episode_time_s=180, car=CarParams(), progress=ProgressParams())`.
  - `World(track, config)` has `.reset(start_s, start_v, mask=None) -> obs` and
    `.step(actions (N,2)) -> StepResult(obs, progress_delta, done, newly_done, done_reason, lap_completed, lap_time, sliding)`.
  - Cars that are done are frozen until a masked reset.
  - `world.car` (`CarState`: x, y, yaw, v, delta) and `world.progress.state` (s, n, progress, lap, t, done, best_lap_time, …) are readable.
- **`sim.sensors`:** `OBS_SIZE = 19`, `OBS_SPEC_VERSION = 1`.
- **`sim.car_model.CarParams`:** tuned defaults, `max_brake_g=3.0`, `grip_base_g=1.9`.
- **`sim.scripted.drive_reference_lap(track, params, raceline_xy=...)`:** returns a `ReferenceLap` with `lap_time_s`, `t`, `x`, `y`, `v`, `s`, `n`. On Silverstone with the TUM raceline it gives 87.458 s.
- **`sim.progress.DoneReason`:** NONE 0, OFF_TRACK 1, WRONG_WAY 2, STALL 3, TIME_LIMIT 4.
- **`data.track_pipeline.load_track(id) -> Track`:** `track.data.fastf1_alignment` holds `rotation_rad`, `tx_m`, `ty_m`, `scale`, `mirrored`, `lap_source`, `reference_driver` and `reference_lap_s`.
- **`data.alignment`:** `Alignment(...)` and `apply_alignment(a, x, y)`.
- **`data.laps`:** `LapXY` and `reference_lap(track_id, year, *, kind, driver, source) -> (LapXY, source)`.
- **Data sources:** `data.openf1_source` (`find_session`, `fastest_lap_xy`, `_get`) and `data.fastf1_source`.
- **App:** `app.camera.Camera` and `app.render_track.build_track_shapes(track)`. See `app/drive.py` for how `DriveWindow` uses them with `arcade.Camera2D`.
- **Formats:** `docs/data-formats.md` sections 3 (ghost), 4 (checkpoint) and 5 (run folder).

## I. Policy and checkpoints: `agents/policy.py`, `agents/checkpoint.py`

```python
# f1sim/agents/policy.py
ACTION_SIZE = 2

@dataclass(frozen=True)
class PolicySpec:
    obs_size: int = OBS_SIZE
    hidden: tuple[int, ...] = (64, 64)
    action_size: int = ACTION_SIZE
    activation: str = "tanh"            # hidden activation; the output is always tanh
    @property
    def n_params(self) -> int: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "PolicySpec": ...

def init_params(spec: PolicySpec, n: int, rng: np.random.Generator) -> np.ndarray: ...
"""(n, n_params) float32. Biases are 0; weights ~ N(0, 1/fan_in)."""

def act(spec: PolicySpec, params: np.ndarray, obs: np.ndarray) -> np.ndarray: ...
"""params (N, n_params) or (n_params,) (broadcast), obs (N, obs_size) -> actions (N, 2) float64 in [-1, 1].
Each car uses its own row: batched per-car matmuls (einsum), with no Python loop over cars."""
```

**Flat layout**, row-major, in this order: `W1 (obs, h1)`, `b1`, `W2 (h1, h2)`, `b2`, …,
`Wout (h_last, 2)`, `bout`. Provide `unflatten(spec, params)` returning the list of
`(W, b)` pairs, for tests and debugging.

**Speed target:** `act` for N=192 cars with (64, 64) hidden layers in ≤ 0.5 ms. Report what you
measure.

```python
# f1sim/agents/checkpoint.py
class CheckpointMismatch(ValueError): ...

@dataclass(frozen=True)
class Checkpoint:
    params: np.ndarray                  # (n_params,) float32
    meta: dict
    optimiser_state: dict[str, np.ndarray]

def car_params_hash(params: CarParams) -> str: ...  # first 12 hex chars of sha1 over the sorted asdict JSON
def save_checkpoint(path: Path, params, meta: dict, optimiser_state: dict | None = None) -> None: ...
def load_checkpoint(path: Path, *, spec: PolicySpec | None = None) -> Checkpoint: ...
```

**`save_checkpoint`:**
- Writes `path` (`<gen>.npz`: `params` plus `opt__<key>` arrays) and the sidecar `path.with_suffix(".json")`.
- Both writes are atomic.
- The required meta keys are listed in data-formats section 4. Raise `ValueError` if any are
  missing; `obs_spec_version` and `policy` are filled in automatically when absent.

**`load_checkpoint`:** raises `CheckpointMismatch` when `obs_spec_version` differs from
`OBS_SPEC_VERSION`, or when `spec` is given and doesn't match `meta["policy"]`. The message says
what differs.

## J. Optimisers: `agents/optimisers.py`

```python
class Optimiser(Protocol):
    dim: int
    population: int
    generation: int                    # tells so far
    sigma: float                       # current step size, for logging
    best_params: np.ndarray            # (dim,) best ever told
    best_fitness: float
    def ask(self) -> np.ndarray: ...   # (population, dim) float32 to evaluate
    def tell(self, fitness: np.ndarray) -> None: ...   # (population,), higher = better, for the last ask
    def state_dict(self) -> dict: ...  # numpy arrays and scalars only, so it fits the NPZ
    def load_state_dict(self, d: dict) -> None: ...

class GA:     # implements Optimiser
    def __init__(self, dim, *, population=64, elites=4, tournament=4, sigma0=0.1,
                 sigma_min=0.01, sigma_max=0.5, patience=10, seed=0, init=None): ...
class CMAES:  # implements Optimiser via the `cma` package, diagonal covariance (dim ~5.6k)
    def __init__(self, dim, *, population=None, sigma0=0.1, seed=0, init=None): ...

def make_optimiser(name: str, dim: int, **kwargs) -> Optimiser: ...   # "ga" | "cmaes"
```

**GA rules:**
- Elites are copied unchanged and **re-evaluated each generation**. Fitness is noisy because
  starts are random, so a lucky elite can't stay on top forever.
- Parents are chosen by tournament.
- Children come from blend crossover (BLX-0.5) of two parents, plus Gaussian mutation with
  standard deviation `sigma`.
- **Sigma adaptation:** if the best fitness of a generation hasn't beaten the best of the last
  `patience` generations, `sigma *= 1.5`; on improvement, `sigma *= 0.9`. Clamp to
  `[sigma_min, sigma_max]`. Document it.
- **Initial population:** `init` gives rows to start from; otherwise rows are drawn with
  `policy.init_params`-like scaling. Accept a `scale` vector, or simply `N(0, sigma0)`; document
  your choice.
- **Determinism:** the same seed and the same told fitnesses give the same asks.

**`cma`:** an optional dependency. Add the extra `es = ["cma>=4.0"]` to `pyproject.toml`; that
one edit is allowed. Install it into the venv. If `cma` is missing, `make_optimiser("cmaes")`
raises an `ImportError` that says what to install.

**Tests:**
- Both optimisers improve a 50-D sphere and a shifted Rastrigin-lite noticeably within 100
  generations.
- GA sigma adapts as documented.
- `state_dict` round-trips mid-run, giving identical subsequent asks.
- Determinism.

## K. Trainer: `train/trainer.py`, `train/metrics.py`, `configs/train_default.toml`

```python
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
    stage1_episode_s: float = 30.0        # random starts, fitness = progress
    stage1_promote_progress_m: float = 0.0  # 0 = auto: 0.8 * (stage1_episode_s * 40 m/s)
    stage2_lead_in_m: float = 80.0        # start this far before the line at reference-profile speed
    stage2_time_factor: float = 1.5       # episode cap = factor * T_ref
def load_train_config(path: Path) -> TrainConfig: ...   # tomllib, unknown keys rejected

@dataclass(frozen=True)
class TrainResult:
    run_dir: Path; generations: int; stage: int
    best_fitness: float; best_lap_s: float | None; first_lap_wall_s: float | None

class Trainer:
    def __init__(self, track: Track, config: TrainConfig, run_dir: Path, *,
                 car: CarParams = CarParams(), reference_lap_s: float | None = None): ...
    def evaluate(self, params: np.ndarray, stage: int) -> tuple[np.ndarray, dict]: ...  # fitness (P,), stats
    def run(self, on_generation=None) -> TrainResult: ...
```

**Evaluation:**
- One `World` with `N = population · episodes_per_genome` cars. Car `i` uses genome
  `i // episodes_per_genome`, and actions come from `policy.act` once per decision.
- **Stage 1:**
  - Start positions are drawn uniformly over the lap, with `start_v ~ U(20, 60)` m/s.
  - The episode lasts `stage1_episode_s`.
  - Fitness = progress in metres; a car that crashed (OFF_TRACK, WRONG_WAY or STALL) loses 10%
    of its progress.
- **Stage 2:**
  - Start `stage2_lead_in_m` before the line, at the reference profile's speed there. Use
    `scripted.speed_profile`, or simply `min(60, profile)`; document the choice.
  - `ProgressTracker` only counts a line-to-line lap, so the first crossing just starts the timed
    lap.
  - A genome's episode ends at its first completed lap (treat it as done; the trainer may simply
    stop counting it), or when it's done, or at `stage2_time_factor · T_ref`.
  - Fitness = progress before a lap is completed; after a completed lap,
    `L + 1000 + 50 · (stage2_time_factor · T_ref − lap_time)`. The crash penalty is as in stage 1.
- **Promotion:** stage 1 → 2 once the generation's best mean progress reaches the promotion
  threshold. Log the stage.
- **`T_ref`:** the scripted reference lap's time, from `drive_reference_lap` with the TUM
  raceline, computed once. It can be overridden with `reference_lap_s`.
- **Seeding:** a single `numpy.random.Generator(seed)` drives the optimiser seed and all start
  positions.

**Run folder** (data-formats section 5), written by `metrics.py`:
- `config.toml`: the resolved config, plus `T_ref` and the car params.
- `metrics.csv`, one row per generation: `gen, stage, wall_s, best_fitness, mean_fitness,
  completion_rate, best_lap_s, best_progress_m, sigma`.
- `checkpoints/<gen:05d>.npz` every `checkpoint_every` generations and whenever the best improves.
- `best_lap.npz` whenever a faster completed lap appears. Re-run that genome alone from the
  stage-2 start, record at physics rate (every step) the columns
  `t, x, y, s, speed, throttle, brake, gear`, and write them with `gear = 0`.

**Stopping:** at `max_generations` or `max_wall_s`. Return a `TrainResult` with
`first_lap_wall_s`, the wall time when the first clean full lap happened.

**Tests (offline, small):** on a synthetic stadium track (`tests/_geometry_helpers.py` +
`build_track`) with a tiny population and short episodes, check:
- The run folder files exist with the right columns.
- The metrics are monotonic in `gen`.
- Reproducibility: same seed, identical `metrics.csv` apart from `wall_s`.
- Stage promotion happens.
- `best_lap.npz` appears once a lap is completed. Use a hand-made "policy" via `init`, or simply
  run enough generations on an easy oval.

The default config file holds the dataclass defaults.

**Report:**
- A real run on Silverstone from `load_track("silverstone")` with defaults for up to 30 minutes
  of wall time. **Run it in the background with a time cap** and report:
  - generations per minute,
  - when stage 2 started,
  - `first_lap_wall_s`,
  - the best lap time over time,
  - `completion_rate` at the end.
- If no lap is completed in 30 minutes, report what the fitness curve looks like and your best
  hypothesis. Don't tune for hours.

## L. Replay viewer: `app/replay.py`

```python
@dataclass(frozen=True)
class Trace:
    label: str; t: np.ndarray; x: np.ndarray; y: np.ndarray; s: np.ndarray; speed: np.ndarray
    lap_time_s: float

def load_trace(path: Path, label: str | None = None) -> Trace: ...
"""NPZ with at least t, x, y. s and speed are optional: compute s by projecting onto the track
lazily in run(), and speed from finite differences. lap_time_s = t[-1] - t[0]."""

def run(track_id: str, traces: list[Path], labels: list[str] | None = None) -> None: ...
```

**Behaviour:**
- Playback of one or more traces on the track, with up to about 4 cars in different colours and
  a legend.
- **Playback controls:** Space pause, Left/Right seek ±2 s, Up/Down change the playback speed
  (0.25× to 8×), `R` restart, `F` cycle which trace the camera follows, mouse-wheel zoom, right-drag
  pan.
- Car positions and headings are interpolated between samples. The heading comes from the
  trace's own tangent (central differences on the samples), smoothed over about 0.2 s so it
  never twitches.
- **HUD:** for each trace, label, current speed in km/h and lap time. Also the live delta of
  trace 2 relative to trace 1 at equal `s` (`t_2(s) − t_1(s)`), green or red.
- Reuse `Camera` and `build_track_shapes` the same way `drive.py` does.
- Keep testable logic (interpolation, heading smoothing, the delta at equal `s`, the playback
  clock) in pure functions or classes, and test those offline.
- **Smoke check:** a short run on Silverstone with two synthetic traces, e.g. the scripted
  reference lap saved as NPZ and the same lap time-scaled by 1.02. Report it as in M2-H.

## M. Real-driver ghost: `data/ghost_builder.py` (+ additive changes to `data/openf1_source.py`, `data/fastf1_source.py`, `data/laps.py`)

```python
@dataclass(frozen=True)
class GhostLap:
    t, x, y, s, speed, throttle, brake, gear: np.ndarray   # data-formats section 3, 20 Hz
    meta: dict            # driver, team (or ""), lap_time_s, year, session, lap_source, track_build

def build_ghost(track_id: str, year: int, *, driver: str | int | None = None, kind: str = "Q",
                source: str = "auto", rate_hz: float = 20.0) -> GhostLap: ...
def ghost_path(year: int, track_id: str, kind: str, driver: str) -> Path: ...   # in paths.ghosts_dir()
def save_ghost(g: GhostLap, path: Path) -> None: ...   # NPZ + JSON sidecar, atomic
def load_ghost(path: Path) -> GhostLap: ...
```

**Rules:**
- **Positions:** `laps.reference_lap`, then the track's stored `fastf1_alignment` transform via
  `apply_alignment`. Both sources share the raw frame, so any driver's lap maps with the same
  transform.
- **Smoothing (data-formats section 3):**
  - `scipy.interpolate.make_smoothing_spline` on `x(t)` and `y(t)`, with its automatic
    smoothing (GCV).
  - Evaluate at `rate_hz` over `[0, lap_time]`.
  - `speed` comes from the spline derivative.
  - `s` is projected with `Track.project` using hints, and unwrapped so it's monotonic over the lap.
- **Car data:** add a function to each source returning the lap's speed, throttle, brake and gear
  samples. That's OpenF1 `car_data` (`date`, `speed` km/h, `throttle` 0–100, `brake` 0/100,
  `n_gear`), and FastF1 `lap.get_car_data()`.
- **Channels:** interpolate throttle linearly, and brake and gear by nearest sample. Use the
  sensor speed only to cross-check: the median absolute difference from the spline speed goes
  into the meta as `speed_check_kmh`.
- **Validation:** raise `ValueError` if the smoothed ghost has `|a| > 8 g` anywhere, or less than
  95% of samples within 1 m of the track edges. That means bad data or the wrong layout.

**Tests:**
- **Offline:** a synthetic lap on a synthetic track with known speed: smoothing reduces jitter
  from noisy 3.7 Hz positions, `s` is monotonic, NPZ/JSON round-trip, and validation errors.
- **Network test (marked):** Silverstone 2024 HAM. Max |accel| < 6 g, ≥ 98% within 1 m of the
  edges, `speed_check_kmh` < 15, and the lap time is 85.99 s.

**Report** the network-test numbers.

## Integration (main session)

- The CLI: `f1sim train`, `f1sim replay <run_dir | npz…>` and `f1sim ghost <track> <year> [driver]`.
- Merging the branches and running the M3 acceptance check:
  - a first clean full lap on Silverstone within 30 minutes of wall time,
  - best lap time improving,
  - no flat best fitness for more than 20 generations without sigma adapting,
  - the same seed reproduces the run.
