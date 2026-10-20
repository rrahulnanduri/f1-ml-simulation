# M2 interface contract

M2 is the vectorised simulation core (revamp-plan section 6). It's split into four owned pieces
built in parallel. The main session then integrates them into `sim/world.py` and
`scripts/bench.py`.

The rules from [m1-interfaces.md](m1-interfaces.md) still apply:
- **Conventions:** SI units, closed loops without a repeated point, heading = travel direction,
  left normal `(-sin h, cos h)`, `n` positive to the left.
- **Ownership:** don't edit another owner's files, and keep `__init__.py` files empty.
- **Tests:** offline tests, with ruff check and ruff format clean.

**Branch base.** Your worktree may start from an old commit. Before anything else, run
`git merge --ff-only main` in your worktree and confirm `docs/m2-interfaces.md` exists.

## What already exists (read, don't modify)

- **`f1sim.track.track`:** `TrackData` and `Track`.
  - `Track` has `length`, `left_wall()` and `right_wall()` (M×2 arrays, closed implicitly).
  - It also has `point_at(s)`, `heading_at(s)`, `curvature_at(s)` and `widths_at(s)`, which
    return `(w_left, w_right)`.
  - `project(x, y, hint=None, window=10)` returns `(s, n, idx)`. It's vectorised; with a hint
    the cost is O(1) per point.
  - `data.sectors_s` is `(0.0, s1, s2)` when the track was aligned to FastF1/OpenF1, and empty
    otherwise.
- **`f1sim.sim.car_model`:** `CarParams`, `CarState` (with `.at(n, x, y, yaw, v)`),
  `step(state, steer_cmd, pedal_cmd, params, dt) -> StepInfo(a_long, a_lat, sliding)`, and
  `grip_limit(v, params)`.
- **`f1sim.data.track_pipeline.load_track(track_id) -> Track`:** reads the user cache.
- **`f1sim.data.tum_source`:**
  `parse_tum_raceline(fetch_tum_raceline(name)) -> (x, y)`, in the same frame as the built track,
  because the pipeline only reorders TUM points and never moves them.
- **Test fixtures:** `tests/_geometry_helpers.py` (`circle`, `stadium`, `teardrop`,
  `figure_eight`) builds synthetic tracks via
  `f1sim.data.track_builder.build_track(x, y, w_right, w_left, track_id=..., name=...)`.
- **Real tracks:** Silverstone is built in the user cache (`load_track("silverstone")`),
  and `f1sim build-track <id> --no-fastf1` builds any TUM circuit. Use real tracks only in
  tests marked `@pytest.mark.network`, or in reports.

## E. Progress, laps and termination: `sim/progress.py`

```python
class DoneReason(IntEnum):
    NONE = 0; OFF_TRACK = 1; WRONG_WAY = 2; STALL = 3; TIME_LIMIT = 4   # TIME_LIMIT is set by World, not here

@dataclass(frozen=True)
class ProgressParams:
    car_half_width_m: float = 1.0      # off track when |n| > w_side - car_half_width_m
    wrong_way_speed_m_s: float = -1.0  # ds/dt below this ...
    wrong_way_time_s: float = 1.0      # ... for this long -> WRONG_WAY
    stall_distance_m: float = 5.0      # less forward progress than this ...
    stall_time_s: float = 5.0          # ... within this window -> STALL
    lap_min_fraction: float = 0.98     # a lap counts only after this share of the length since the last lap start
    window: int = 10                   # projection hint window

@dataclass
class ProgressState:                   # all arrays shape (N,)
    idx: np.ndarray        # int64 segment index (projection hint)
    s: np.ndarray; n: np.ndarray
    progress: np.ndarray   # unwrapped distance along the track since reset (can go negative)
    lap: np.ndarray        # int64 completed laps
    lap_start_t: np.ndarray; last_lap_time: np.ndarray; best_lap_time: np.ndarray   # nan = none yet
    sector: np.ndarray     # int64 current sector index (0..len(sectors_s)-1), 0 if no sectors
    sector_start_t: np.ndarray
    t: np.ndarray          # seconds since reset
    done: np.ndarray       # bool
    done_reason: np.ndarray  # int8 DoneReason

@dataclass(frozen=True)
class ProgressEvents:                  # per update, shape (N,)
    lap_completed: np.ndarray          # bool
    lap_time: np.ndarray               # float, nan where no lap completed this step
    sector_completed: np.ndarray       # int64 index of the sector just finished, -1 if none
    sector_time: np.ndarray            # float, nan where none
    newly_done: np.ndarray             # bool, became done this step

class ProgressTracker:
    def __init__(self, track: Track, n_cars: int, params: ProgressParams = ProgressParams()): ...
    state: ProgressState
    def reset(self, x, y, mask=None) -> None: ...     # global projection; zero counters for masked cars (all if None)
    def update(self, x, y, dt: float) -> ProgressEvents: ...   # only cars with done == False change
```

**Rules:**
- **Projection:** use the hinted projection. If a car's `|Δs|` (wrap-aware) exceeds 50 m in one
  step, re-project that car globally.
- **Progress:** `progress` adds the wrap-aware `Δs`.
- **Laps:** a lap is completed when `s` wraps past 0 going forward **and**
  `progress - progress_at_lap_start ≥ lap_min_fraction · length`. Lap and sector times are
  interpolated within the step to the crossing, so they're sub-step accurate. The first crossing
  after a reset at the start line doesn't count unless a lap's worth of progress happened.
- **Sectors:** same crossing logic at each `sectors_s` boundary (forward only).
- **Termination:** off track, wrong way and stall as in `ProgressParams`. A car that's done
  stays done until `reset` with its mask.

## F. Ray grid and observations: `track/grid.py`, `sim/sensors.py`

```python
# f1sim/track/grid.py
class SegmentGrid:
    def __init__(self, segments: np.ndarray, cell_size_m: float = 25.0): ...   # segments (K, 2, 2): [[x0,y0],[x1,y1]]
    @classmethod
    def from_track(cls, track: Track, cell_size_m: float = 25.0) -> "SegmentGrid": ...  # both walls, closed
    def raycast(self, ox, oy, angles, max_dist: float) -> np.ndarray: ...
    """ox, oy (N,); angles (N, R) absolute radians -> distance to first wall hit (N, R), max_dist if none."""

# f1sim/sim/sensors.py
OBS_SPEC_VERSION = 1
RAY_ANGLES_DEG = (-90, -60, -35, -15, 0, 15, 35, 60, 90)   # relative to yaw; + = left
RAY_RANGE_M = 200.0
LOOKAHEAD_M = (20.0, 50.0, 100.0, 150.0, 250.0)
V_NORM = 92.0
CURVATURE_SCALE = 50.0
OBS_SIZE = 19

class Sensors:
    def __init__(self, track: Track, grid: SegmentGrid | None = None): ...
    def observe(self, x, y, yaw, v, s, n, prev_action) -> np.ndarray: ...
    """(N, OBS_SIZE) float32, layout exactly as docs/data-formats.md section 6:
    0-8 rays / RAY_RANGE_M; 9 v / V_NORM; 10 heading error (yaw - track heading at s, wrapped) / pi;
    11 n / (w_left if n >= 0 else w_right); 12-16 clip(curvature(s + d) * CURVATURE_SCALE, -1, 1);
    17-18 prev_action (steer, pedal). prev_action is (N, 2)."""
```

**Requirements:**
- **Ray accuracy:** within 1 cm of the analytic answer on a circular track.
- **Speed:** fully vectorised over N×R. Walk the grid cells along each ray (e.g. a DDA-style
  traversal done with numpy in batches), or gather the candidate segments per ray up to
  `max_dist`. Don't test every segment against every ray.
- **Benchmark target:** 64 cars × 9 rays on Silverstone in ≤ 2 ms. Report what you measure.

## G. Scripted reference driver (physics calibration): `sim/scripted.py`

```python
@dataclass(frozen=True)
class ReferenceLap:
    lap_time_s: float          # nan if the car left the track or didn't finish
    completed: bool
    t: np.ndarray; x: np.ndarray; y: np.ndarray; v: np.ndarray; s: np.ndarray; n: np.ndarray
    max_abs_n_excess_m: float  # worst |n| beyond (w_side - 1.0); <= 0 means always on track

def speed_profile(path_x, path_y, params: CarParams, *, closed: bool = True) -> np.ndarray: ...
"""Max speed at each path point. Lateral limit from curvature and grip_limit (including drag in
the friction circle, exactly like car_model.step), then backward (braking) and forward (drive)
passes with the same longitudinal limits as car_model. Closed loop: iterate until periodic."""

def drive_reference_lap(track: Track, params: CarParams, *, raceline_xy=None, dt: float = 1/60,
                        flying: bool = True) -> ReferenceLap: ...
"""Drive car_model.step around `track` along `raceline_xy` (centre line if None), following
speed_profile with a pure-pursuit (or Stanley) steering controller and a speed controller.
flying=True: start at s=0 at the profile speed (a flying lap, like qualifying).
Lap ends when the car crosses s=0 again after >= 98% of the length."""
```

**Requirements:**
- **Raceline direction:** may be either way relative to the track. Orient it to the track's
  direction yourself (project it and check the sign of `ds`), and roll it to start near `s = 0`.
- **Report:**
  - Silverstone lap times on the centre line and on the TUM raceline (`"Silverstone"`), with
    default `CarParams`, against the real 2024 pole of 85.819 s. The M2 target is within ±10%.
  - The speed trace's top speed and minimum corner speeds.
  - If the lap time is outside the ±10%, recommend specific `CarParams` changes with numbers.
    Don't edit `car_model.py` or `configs/car_f1.toml`.
- **Tests:**
  - A stadium track: lap time close to the analytic profile time.
  - The car stays on track.
  - Raceline orientation handling.

## H. Minimal drive viewer: `app/camera.py`, `app/render_track.py`, `app/drive.py`

```python
# f1sim/app/camera.py   (pure maths, no arcade import)
class Camera:
    def __init__(self, viewport_w: int, viewport_h: int): ...
    def fit(self, xmin, ymin, xmax, ymax, padding_px: int = 20) -> None: ...
    def zoom_at(self, factor: float, screen_x: float, screen_y: float) -> None: ...   # keeps that point fixed
    def pan(self, dx_px: float, dy_px: float) -> None: ...
    def follow(self, x: float, y: float) -> None: ...
    def resize(self, viewport_w: int, viewport_h: int) -> None: ...                     # keeps the centre
    def world_to_screen(self, x, y) -> tuple[np.ndarray, np.ndarray]: ...
    def screen_to_world(self, sx, sy) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def pixels_per_metre(self) -> float: ...

# f1sim/app/render_track.py   (geometry pure + one arcade function)
def road_triangles(track: Track) -> np.ndarray: ...       # (2M, 3, 2) triangles filling the road between walls
def boundary_lines(track: Track) -> dict[str, np.ndarray]: ...   # "start_finish" (2,2), "sectors" (k,2,2)
def build_track_shapes(track: Track): ...                 # arcade ShapeElementList in world metres, built once

# f1sim/app/drive.py
def run(track_id: str) -> None: ...   # opens the window; the main session wires `f1sim drive <id>` to this
```

**Requirements:**
- **Input:** keyboard driving with `car_model`. Up = throttle, down = brake (both ramped over
  about 0.15 s), left/right = steer (ramped). `R` resets to the start line, `F` toggles follow,
  the mouse wheel zooms at the cursor, a right-drag pans, and `Esc` quits.
- **Off track:** detect it with `Track.project`, using the same rule as section E. On leaving the
  track, show "OFF TRACK" and reset after 1 s.
- **Display:**
  - A simple lap timer with last and best laps (forward crossing of `s=0` after ≥ 98% of the
    length), plus speed in km/h and gear-less RPM-free text.
  - The car drawn as a 5.6 × 2.0 m rectangle.
  - The road as a filled polygon with walls, a red start line and blue sector lines.
- **Rendering cost:** cache the static track shapes, so the draw cost isn't per-frame geometry.
- **Tests:** offline, without opening a window. Cover camera maths (round trips, zoom keeps the
  cursor point fixed, fit), `road_triangles` covering the road area (the triangle areas sum to the
  road area within 1%), and boundary lines spanning the road width. Arcade is already a
  dependency.
- **Window check:** also run a short smoke script yourself that opens the window on Silverstone,
  steps about 120 frames with scripted inputs, and closes it. Report that it worked.

## Integration (main session)

- `sim/world.py`: car_model + progress + sensors for N cars, `step(actions) -> StepResult`.
- `scripts/bench.py`.
- The CLI commands `f1sim drive` and `f1sim reference-lap`.
- The M2 acceptance checks.
