# M1 (+ M2 car model) interface contract

Work is split across parallel branches that must fit together without edits. Each module below
has one owner. Don't create, edit or import another owner's module, except where "may import"
says so. Package `__init__.py` files already exist and must stay empty.

## Conventions (everyone)

- **Units:** SI everywhere (metres, seconds, radians). Arrays are `numpy.float64`, 1-D unless
  stated.
- **Closed loops:** stored without repeating the first point at the end. Index `i` goes to `i+1`,
  and the last point joins back to point 0.
- **Heading:** `atan2(dy, dx)` of the travel direction. The **left normal** is
  `(-sin h, cos h)`. The left wall is `p + nL * w_left` and the right wall is
  `p - nL * w_right`. The lateral offset `n` is **positive to the left**.
- **Arc length:** `s[0] = 0`, strictly increasing; `length_m` includes the closing segment.
- **Offline:** no network in tests. Network-touching tests go behind
  `@pytest.mark.network` and are skipped by default (`-m "not network"` is in addopts).
- **Python 3.11+, numpy/scipy/shapely only:** no new runtime deps, except `platformdirs`,
  which the data-sources owner adds to `pyproject.toml`.
- **Code style:** type hints on public functions, short docstrings, and `ruff check` +
  `ruff format` clean.
- **Running things:** use the repo venv,
  `C:\Users\r.nanduri\Desktop\Desktop\Desktop\Personal\f1-ml-simulation\.venv\Scripts\python.exe`,
  from your worktree root. pytest puts `src/` first on `sys.path`, so your worktree's code is the
  code under test.

## A. Track geometry: `track/track.py`, `data/track_builder.py`, `data/io.py`

```python
# f1sim/track/track.py
@dataclass(frozen=True)
class TrackData:
    id: str
    name: str
    x: np.ndarray; y: np.ndarray; s: np.ndarray
    w_left: np.ndarray; w_right: np.ndarray
    heading: np.ndarray; curvature: np.ndarray      # curvature > 0 = turning left
    length_m: float
    resolution_m: float
    sectors_s: tuple[float, ...] = ()
    fastf1_location: str | None = None
    direction: str = ""                              # "clockwise" | "counterclockwise"
    source: dict = field(default_factory=dict)
    fastf1_alignment: dict | None = None

class Track:
    def __init__(self, data: TrackData): ...
    data: TrackData; length: float
    def left_wall(self) -> np.ndarray: ...           # (M, 2)
    def right_wall(self) -> np.ndarray: ...          # (M, 2)
    def point_at(self, s) -> np.ndarray: ...         # s wraps modulo length; (..., 2)
    def heading_at(self, s) -> np.ndarray: ...
    def curvature_at(self, s) -> np.ndarray: ...
    def widths_at(self, s) -> tuple[np.ndarray, np.ndarray]: ...   # (w_left, w_right)
    def project(self, x, y, hint=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised. Returns (s, n, idx); idx = segment index.
        hint=None: global nearest segment. hint=array of idx: search only hint±window (O(1))."""

# f1sim/data/track_builder.py
class TrackBuildError(ValueError): ...
def build_track(x, y, w_right, w_left, *, track_id: str, name: str, resolution_m: float = 2.0,
                max_shift_m: float = 0.5, width_filter_m: float = 20.0,
                source: dict | None = None) -> TrackData: ...
def wall_self_crossings(data: TrackData) -> tuple[int, int]: ...   # (left, right) counts
```

`build_track` takes a raw closed loop, possibly with the first point repeated at the end, and
follows revamp-plan section 4.2:
1. Periodic smoothing spline that moves no point more than `max_shift_m`.
2. Arc-length resampling at `resolution_m`.
3. Widths interpolated, then median-filtered over `width_filter_m`.
4. Heading and curvature from the spline.
5. Wall repair: clip a side's width to `0.95/|κ|` where it exceeds that on the inside of the bend.
6. Raise `TrackBuildError` if either wall is still not simple.

```python
# f1sim/data/io.py
BUILDER_VERSION: int = 1
def write_track_json(data: TrackData, path: Path) -> None: ...
def read_track_json(path: Path) -> TrackData: ...    # format: docs/data-formats.md section 1
```
**May import:** nothing else from f1sim.

## B. Data sources: `paths.py`, `data/tum_source.py`, `data/catalog.py`, `data/fastf1_source.py`

```python
# f1sim/paths.py   (platformdirs user cache dir "f1sim"; env F1SIM_CACHE_DIR overrides; dirs created on access)
def cache_dir() -> Path: ...
def tum_dir() -> Path: ...; def tracks_dir() -> Path: ...; def ghosts_dir() -> Path: ...; def fastf1_dir() -> Path: ...

# f1sim/data/tum_source.py
TUM_REPO = "TUMFTM/racetrack-database"
TUM_COMMIT: str                                      # pinned full SHA (git ls-remote)
TUM_TRACKS: tuple[str, ...]                          # file stems, e.g. "Silverstone", "YasMarina"
class TumTrack(NamedTuple): x: np.ndarray; y: np.ndarray; w_right: np.ndarray; w_left: np.ndarray
def fetch_tum_csv(name: str) -> Path: ...            # downloads tracks/<name>.csv at TUM_COMMIT into tum_dir()/<commit>/, cached
def parse_tum_csv(path: Path) -> TumTrack: ...
def fetch_tum_raceline(name: str) -> Path: ...       # racelines/<name>.csv
def parse_tum_raceline(path: Path) -> tuple[np.ndarray, np.ndarray]: ...

# f1sim/data/catalog.py
@dataclass(frozen=True)
class CatalogEntry:
    id: str; name: str; tum_name: str | None; fastf1_locations: tuple[str, ...]; notes: str = ""
CATALOG: dict[str, CatalogEntry]                     # all current-calendar circuits; tum_name None where TUM lacks it
def get(track_id: str) -> CatalogEntry: ...
def entry_for_location(location: str) -> CatalogEntry | None: ...   # FastF1 Event "Location"

# f1sim/data/fastf1_source.py   (FastF1 cache in paths.fastf1_dir(); only module that imports fastf1)
@dataclass(frozen=True)
class EventInfo: year: int; round: int; name: str; location: str; country: str
@dataclass(frozen=True)
class LapXY:
    driver: str; lap_time_s: float
    t: np.ndarray; x_m: np.ndarray; y_m: np.ndarray  # FastF1 X/Y ÷ 10, t from lap start, duplicates removed
    sector_end_t: tuple[float, float]                # end of S1 and S2, seconds from lap start
def event_schedule(year: int) -> list[EventInfo]: ...   # excludes testing (round 0)
def load_session(year: int, location: str, kind: str = "Q"): ...
def fastest_lap_xy(session, driver: str | None = None) -> LapXY: ...   # driver = abbreviation or number
```
**May import:** `paths` (its own). Mark network tests; test parsing with small fixture CSVs in
`tests/fixtures/tum/`.

## C. Alignment: `data/alignment.py`

```python
@dataclass(frozen=True)
class Alignment:
    rotation_rad: float; tx_m: float; ty_m: float; scale: float; mirrored: bool
    median_dist_m: float; inside_ratio: float
def apply_alignment(a: Alignment, x, y) -> tuple[np.ndarray, np.ndarray]: ...
def align_to_centreline(px, py, cx, cy, w_left, w_right, *, fit_scale: bool = False) -> Alignment: ...
```
`(px, py)` is a telemetry path in metres in an unknown rigid frame. `(cx, cy, w_left, w_right)` is
a closed centre line with widths. The function follows revamp-plan section 4.3:
1. Coarse search over rotation (every 5°) and mirroring, starting from the centroids.
2. Point-to-segment ICP refinement.
3. Return the fit statistics: median distance, and the share of points inside the edges using the
   sign convention above.

The `mirrored` flag reflects the path across the x-axis **before** rotating it.
**May import:** nothing else from f1sim (it does its own projection).

## D. Car physics (M2, early): `sim/car_model.py`, `configs/car_f1.toml`

```python
@dataclass(frozen=True)
class CarParams:   # defaults = revamp-plan section 6 table
    length_m: float = 5.6; width_m: float = 2.0; wheelbase_m: float = 3.6
    max_steer_rad: float = 0.30; steer_rate_rad_s: float = 1.5
    power_w: float = 750e3; mass_kg: float = 800.0; max_drive_g: float = 1.3
    drag_coeff: float = 0.0012; grip_base_g: float = 1.6; grip_aero_coeff: float = 0.00069
    max_brake_g: float = 5.5
def load_car_params(path: Path) -> CarParams: ...    # tomllib, unknown keys rejected

@dataclass
class CarState:     # arrays of shape (N,)
    x: np.ndarray; y: np.ndarray; yaw: np.ndarray; v: np.ndarray; delta: np.ndarray
    @classmethod
    def at(cls, n: int, x: float, y: float, yaw: float, v: float = 0.0) -> "CarState": ...

@dataclass(frozen=True)
class StepInfo: a_long: np.ndarray; a_lat: np.ndarray; sliding: np.ndarray

def step(state: CarState, steer_cmd, pedal_cmd, params: CarParams, dt: float) -> StepInfo: ...
"""In-place, vectorised over N cars. steer_cmd and pedal_cmd in [-1, 1]; pedal > 0 = throttle, < 0 = brake."""
def grip_limit(v, params) -> np.ndarray: ...         # a_max(v) in m/s^2
```
Physics follows revamp-plan section 6: a kinematic bicycle with rate-limited steering, power- and
traction-limited drive, quadratic drag, and a friction circle. When the limit is exceeded the car
slides: yaw rate is reduced to the limit, speed is scrubbed, and `sliding` is True. There's no
reverse (`v >= 0`).
**May import:** nothing else from f1sim.

## Integration (main session, after merge)

`f1sim` CLI (`tracks`, `build-track <id>`), the catalogue→TUM→builder→io pipeline, alignment
against FastF1 laps, sector boundaries, and the M1 acceptance checks on real Silverstone.
