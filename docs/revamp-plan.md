# F1 ML Simulation: Revamp Plan

Status: approved 2026-10-20, with the section 12 decisions settled. M0 in progress.

## 1. Why revamp

These numbers come from measuring the current code: the Silverstone 2024 cache, the default 1500×1000 window, and a 4-minute headless training run with 21 generations.

| Area | What the code does today | Measured effect |
|---|---|---|
| Track shape | Uses George Russell's pole-lap telemetry (1:25.819) as the centre line and offsets two walls ±12.5 px from it | The "track" is one driver's racing line inside a tube, so there's no racing line left to learn. Hamilton's ghost sits a median 0.1 m from it |
| Track width | Fixed 25 px wall to wall | 29.7 m at Silverstone (real: about 12–18 m) and 33.5 m at Albert Park. The width changes with screen scale |
| Walls | A normal offset of a noisy line | 12 inner and 19 outer self-crossings, which make phantom walls at tight corners |
| Sampling | Raw telemetry points | Points are 0.15–34 m apart (median 7.9 m), with a 9.1 m start/finish gap |
| Units | Everything in screen pixels | The car is 11.9 × 7.1 m, the sensors reach 119 m, and the top speed is 2,563 km/h |
| Physics | Constant turn rate, no grip limit, one pedal, forced throttle | No braking behaviour to learn, and lap times mean nothing in real units |
| Fitness | Total distance driven, laps judged by distance | Zig-zagging is rewarded, and wrong-way driving is only caught near the start line |
| Generation end | Ends when the ghost finishes (86 s) | Every generation from gen 3 ends at exactly 86.0 s, so the 300 s timer is dead code |
| Evolution | Top-2 elitism, mutate-only, deterministic sim | Best fitness frozen at 4482.2 for 16 generations, and no lap ever completed |
| Performance | Per-car Python loop and shapely polygon tests | 395 physics steps/s = 6.6× real time with 30 cars, so the 20× slider can't be reached |
| Architecture | `renderer.py` holds UI, physics, game loop and training | Can't train without the window, and nothing is testable |
| Data and repo | 178 MB of cache, pickles and `.pyc` in git, no `.gitignore` | The repo is bloated, and cache naming duplicates tracks ("Silverstone" vs "British Grand Prix") |

## 2. Goals and non-goals

**Goals**
1. **A real track.** Use measured centre lines and left/right widths in metres, with no self-intersections, from a documented source.
2. **SI units everywhere** (metres, seconds, radians). The screen is only a camera.
3. **Believable F1-ish car physics.** A good driver model should lap within about 10% of real pole, so comparing against the ghost means something.
4. **A headless simulation core** that runs ≥ 50× real time with 64 cars and is deterministic for a given seed.
5. **Learning that works:** a full clean lap reliably, with lap time improving over generations.
6. **A thin, responsive app:** it never freezes while loading, has a camera, a real ghost with a live delta, and replays.
7. **A clean repo:** a package layout, tests, config files, and no generated data in git.

**Non-goals for now:** tyre wear, fuel, weather, multi-car racing or overtaking, 3D, elevation, and an online leaderboard.

## 3. Target architecture

```
f1-ml-simulation/
├── pyproject.toml              # package metadata, deps, extras [dev], [rl]; ruff + pytest config
├── README.md
├── LICENSE                     # + THIRD_PARTY.md (TUM racetrack-database is LGPL-3.0)
├── .gitignore                  # .venv/, __pycache__/, .DS_Store, runs/, cache/, *.egg-info
├── configs/
│   ├── car_f1.toml             # car parameters (section 6)
│   ├── sim.toml                # dt, episode limits, track-limit rule, sensors
│   └── train_default.toml      # optimiser, population, seed, curriculum
├── src/f1sim/
│   ├── __init__.py
│   ├── cli.py                  # `f1sim` entry point: tracks / build-track / train / replay / app / bench
│   ├── config.py               # frozen dataclasses + TOML loading/validation
│   ├── paths.py                # user cache/data dirs via platformdirs (never the repo)
│   ├── data/
│   │   ├── catalog.py          # track ids <-> FastF1 event Location names, availability
│   │   ├── tum_source.py       # download + parse TUM CSV (pinned commit)
│   │   ├── fastf1_source.py    # schedule, sessions, laps, telemetry, pos_data (one place for FastF1)
│   │   ├── track_builder.py    # resample, smooth, widths, wall repair -> TrackData
│   │   ├── width_estimator.py  # fallback: widths from spread of all drivers' positions
│   │   ├── alignment.py        # FastF1 frame -> track frame registration (ICP)
│   │   ├── ghost_builder.py    # FastF1 lap -> GhostLap in track frame, parameterised by s
│   │   └── io.py               # read/write track JSON, ghost NPZ, checkpoint NPZ (no pickle)
│   ├── track/
│   │   ├── track.py            # Track: centre line, s, widths, heading, curvature, walls, project()
│   │   └── grid.py             # uniform grid over wall segments for vectorised raycasts
│   ├── sim/
│   │   ├── car_model.py        # vectorised kinematic bicycle + friction circle
│   │   ├── sensors.py          # vectorised raycasts + look-ahead features -> obs
│   │   ├── progress.py         # Frenet (s, n) tracking, laps, sectors, wrong-way, stall
│   │   ├── world.py            # World: N cars as numpy arrays; step(actions) -> StepResult
│   │   ├── scripted.py         # reference driver (raceline follower) for physics calibration
│   │   └── env.py              # Gymnasium env wrapper (optional extra [rl])
│   ├── agents/
│   │   ├── policy.py           # MLP policy over a flat parameter vector (numpy)
│   │   ├── ga.py               # tournament selection, crossover, Gaussian mutation, elitism
│   │   ├── es.py               # CMA-ES via `cma` (optional)
│   │   └── checkpoint.py       # save/load agent + metadata
│   ├── train/
│   │   ├── trainer.py          # headless generations, evaluation, curriculum, logging
│   │   └── metrics.py          # metrics.csv + best-lap trace export
│   └── app/
│       ├── main.py             # arcade Window: modes Replay / Watch-training / Drive
│       ├── camera.py           # pan, zoom, follow
│       ├── loader.py           # background-thread loading with progress
│       ├── render_track.py     # road polygon, kerbs, start line, sectors (cached shapes)
│       ├── render_cars.py      # car + ghost sprites in metres
│       ├── hud.py              # lap, delta to ghost, speed, generation stats
│       └── panels.py           # year/event/driver pickers, speed control
├── tests/
│   ├── fixtures/               # tiny synthetic tracks (circle, oval, hairpin) + small Silverstone extract
│   └── test_*.py
├── scripts/bench.py            # steps/s benchmark
└── docs/
    ├── revamp-plan.md          # this file
    └── data-formats.md         # canonical file-format spec, kept in sync with code
```

**Dependency direction:** `app → train → agents → sim → track ← data`. `sim` and `track` never import `arcade` or `fastf1`, and `data` is the only module that touches the network.

**Training and viewing are separate processes.** `f1sim train` runs headless and writes checkpoints and metrics into `runs/<run_id>/`. The app's "Watch training" mode tails that folder and replays the latest best agent, so the UI frame rate never limits training.

### Old → new mapping

| Today | Becomes |
|---|---|
| `main.py` | `f1sim app` |
| `generate_track.py`, `src/constants/track_data.json` | Deleted (leftovers from the React version) |
| `src/track_service.py` | Split into `data/fastf1_source.py`, `data/ghost_builder.py` and `data/catalog.py` |
| `src/utils.py` (`fit_to_screen`, `generate_track_walls`) | `app/camera.py`, `data/track_builder.py` |
| `src/spatial_hash.py` | `track/grid.py` (vectorised) |
| `src/car.py` | `sim/car_model.py`, `sim/sensors.py`, `app/render_cars.py` |
| `src/neural_net.py`, `src/evolution_manager.py` | `agents/policy.py`, `agents/ga.py` |
| `src/ghost_car.py` | `data/ghost_builder.py` + `app/render_cars.py` |
| `src/renderer.py` | `app/*`, `sim/world.py`, `train/trainer.py` |

The old `src/` keeps running until M4 is done, then it gets deleted in one commit.

## 4. Track pipeline (the core fix)

### 4.1 Sources

1. **Primary:** the [TUM racetrack-database](https://github.com/TUMFTM/racetrack-database). Each file `tracks/<Name>.csv` has a `# x_m,y_m,w_tr_right_m,w_tr_left_m` header line, then about 5 m spaced centre-line points with widths. The centre lines come from OpenStreetMap and the widths from satellite images, so the centre line isn't perfectly centred and accuracy varies. It's licensed LGPL-3.0, so the download is pinned to a specific commit hash. It also has a `racelines/` folder with min-curvature racing lines, which section 6 uses for physics calibration.
2. **Fallback** for circuits TUM doesn't have: `width_estimator.py`.
   - Build a centre line from the median of every driver's qualifying and race laps (`session.pos_data`, about 4 Hz).
   - Take the 1st–99th percentile of each driver's sideways offset from that centre line, then add a 1.0 m margin to each side.
   - Smooth the result and tag the file `"widths": "fastf1-spread"` (lower quality, shown in the UI).

**Coverage.** The current calendar circuits TUM has are Sakhir, Melbourne, Suzuka, Shanghai, Catalunya, Montreal, Spielberg, Silverstone, Budapest, Spa, Zandvoort (DTM), Monza, Austin, Mexico City, São Paulo and Yas Marina. Monaco, Baku, Singapore, Jeddah, Miami, Imola, Las Vegas and Lusail need the fallback.

**Risk:** some TUM layouts may predate changes, such as Yas Marina 2021, Barcelona's chicane removal in 2023 and Albert Park 2022. The alignment check in 4.3 catches these and makes the code fall back.

### 4.2 Building the track (`track_builder.py`)

1. Parse the CSV and close the loop, removing a duplicate end point if there is one.
2. Fit a periodic smoothing spline through x and y, with the smoothing chosen so the spline moves points less than 0.5 m.
3. Resample by arc length every **2.0 m** (configurable). Interpolate the widths the same way, then median-filter them over 20 m.
4. Compute these per point: `s` (distance along the track), `heading`, signed `curvature` (from spline derivatives), and the normal vector.
5. Walls: `p ± n·w`.
6. **Repair self-crossings:** where a wall folds over itself (inner side of a tight corner, where `w > 1/|κ|`), clip the width to `0.95/|κ|` and recompute. Then check with `LineString.is_simple` and fail the build if it's still not simple.
7. Orientation: set the start/finish at `s = 0`, matching the FastF1 lap start (4.3), and set the direction to match F1 racing direction.
8. Sectors: take the sector boundaries in `s` from FastF1 sector times on the reference lap (the distance at each sector's end time).

### 4.3 Aligning FastF1 data to the track (`alignment.py`)

FastF1 X/Y is in 1/10 m, in its own frame. TUM uses local metres in a different frame.

1. Convert telemetry to metres (÷10).
2. Get an initial rotation by trying angles every 5° (and mirrored versions) and keeping the best centroid-aligned fit of the telemetry path to the centre line.
3. Refine with point-to-line ICP: rotation and translation, scale fixed at 1, with a ±2% scale check logged.
4. **Acceptance:** median distance from telemetry to the centre line ≤ 6 m, and ≥ 98% of samples within 1 m of the track edges. If not, reject the TUM layout for that year and use the fallback.
   - These were relaxed from ≤ 4 m and 98% strictly inside after measuring real data in M1. TUM's own optimal racing line sits a median 5.1 m off Silverstone's centre line. Russell's and Hamilton's 2024 laps are 95.7–95.8% strictly inside, because they use the kerbs at corner exits, and 99.7% within 1 m. A lap from the wrong circuit scores ≤ 23% inside.
5. Store the transform in the track file ([data-formats.md](data-formats.md), section 1) so ghosts and laps from any driver line up for free.

## 5. Data formats

Moved to [data-formats.md](data-formats.md), the canonical spec. It covers track JSON, the track catalogue, ghost-lap NPZ, agent checkpoints, run folders, and observation spec v1. All generated data lives in the user cache, never the repo, and nothing is pickled.

## 6. Simulation core

**Shared state.** All cars are numpy arrays of shape `(N,)`: `x, y, yaw, v, delta, s, n, idx, lap, lap_t, t, alive, done_reason`. One `World.step(actions)` advances every car at once.

**Timing.** Physics runs at `dt = 1/60` s, and the policy acts every 2 steps (30 Hz).

**Car model** (`car_model.py`): a kinematic bicycle with a friction circle that grows with downforce.

| Parameter | Start value | Note |
|---|---|---|
| length × width, wheelbase | 5.6 × 2.0 m, 3.6 m | |
| max steer, steer rate | 0.30 rad, 1.5 rad/s | rate-limited |
| power / mass | 750 kW / 800 kg | gives `a_eng = min(P/(m·v), 1.3 g)` |
| drag coefficient | `a_drag = 0.0012·v²` | top speed ≈ 92 m/s (330 km/h) |
| grip | `a_max(v) = g·(1.6 + 0.00069·v²)` | 1.6 g slow, about 5 g at 250 km/h |
| braking | limited by `a_max(v)`, cap 5.5 g | |
| limit rule | if `√(a_lat² + a_long²) > a_max` | the car slides: yaw rate is cut to the limit and speed is scrubbed |

**Calibration result (M2).** With the defaults, the scripted lap is 91.24 s (+6.3%) on the TUM raceline pulled 0.3 m inside the track limit. The raw TUM line comes within 1 m of our built edges at 353 of 1161 points, so it would count as off track. Known deviations from real F1:
- Braking from 300 to 80 km/h takes 81 m (real: 100–130 m).
- Corners above a radius of about 150 m are flat out.
- Medium-speed corners are too slow (Becketts 153 km/h, Stowe 163 km/h).

**Applied (2026-10-20):** `max_brake_g=3.0, grip_base_g=1.9` → 87.458 s (+1.9%), braking 300→80 km/h in 98 m.

**Calibration.** `scripted.py` follows TUM's racing line using a forward/backward speed-profile pass under these limits, and its lap time is compared with real pole: Silverstone 2024 Q was 1:25.819. Tune the parameters until it lands within ±10%. This number is the M2 acceptance test.

**Track limits and progress** (`progress.py`):
- Each car keeps its nearest centre-line index and only searches ±10 points per step, so finding its spot is O(1).
- `s` and `n` (sideways offset) come from projecting the car onto the local segment.
- **Off track:** `|n| > w_side − 1.0 m` (half the car's width), so no polygon tests are needed. The default is a crash; the config option `track_limits = "runoff"` slows the car down instead.
- **Laps:** a lap counts when `s` wraps past the start line **and** the progress since the last lap is at least 98% of the track length. Driving a long way can't fake a lap.
- **Wrong way:** `ds/dt < −1 m/s` for more than 1 s, checked anywhere on the track.
- **Stall:** less than 5 m of progress in 5 s.
- Sector times are recorded from `sectors_s`.

**Sensors.** Rays are tested against walls in a uniform grid (`track/grid.py`, 25 m cells). All cars and all rays are cast together as batched segment-intersection maths in numpy.

**Determinism.** Every random choice comes from one `numpy.random.Generator(seed)`, so the same seed and config produce the same metrics.

## 7. Learning

- **Policy:** an MLP 19 → 64 → 64 → 2 with tanh, about 5.6k parameters, stored as one flat vector.
- **Optimisers:**
  - **GA (default):** population 64, tournament of 4, blend crossover, Gaussian mutation (σ starts at 0.1, adaptive), 4 elites.
  - **CMA-ES:** the `cma` package, same interface.
  - **PPO:** later, through a Gymnasium env and stable-baselines3 (extra `[rl]`).
- **Breaking the deterministic-elite stall:** each genome is evaluated on K = 3 episodes with random start positions (`s₀`) and starting speeds. Fitness is the mean.
- **Fitness:**
  - Before a full lap: progress in metres. A crash costs 10% of the distance driven so far.
  - After a completed lap: `L + 1000 + 50·(T_ref·1.5 − lap_time)`.
  - `T_ref` comes from the scripted reference lap, never the ghost.
  - The episode cap is `1.5·T_ref`; the ghost no longer ends generations.
- **Curriculum:**
  1. Single sector, from random start points.
  2. Full lap from the grid.
  3. Several tracks (from M5).
- **Logging:** `metrics.csv` each generation, a checkpoint every 10 generations and whenever the best improves, and a best-lap trace.

## 8. App

- **Modes:**
  - **Replay:** a checkpoint or `best_lap.npz` against a ghost.
  - **Watch training:** follows `runs/<id>/` live.
  - **Drive:** keyboard driving, for checking how the physics feels.
- **Loading:** a background thread with a progress overlay, so the window never freezes. The year/event/driver pickers come from the catalogue and the FastF1 schedule, and the dropdown always shows the track that's loaded.
- **Camera:** fit to the whole track, mouse-wheel zoom, drag to pan, `F` to follow the best car, and refit on resize. The sidebar sits in its own viewport, not on top of the track.
- **Drawing:**
  - The road is a filled polygon between the walls, with kerbs on corner apexes, a proper start line across the track, and coloured sector boundaries from the data.
  - Cars are 5.6 × 2 m sprites; the ghost is semi-transparent with a driver label.
  - Sensor rays are toggled with `D` and drawn for the followed car only.
  - Static shapes are built once and cached.
- **HUD:** generation, cars alive, best lap and sectors, live delta to the ghost (green/red), speed and gear, plus a speed trace of the AI against the ghost over `s`.
- **Controls:** `Space` pause, `R` reset, `1–4` playback speed, `S` save a checkpoint (Watch mode), and `Esc` to go back to the menu.

## 9. Repo, tooling, testing

- **`pyproject.toml`:**
  - Python ≥ 3.11, tested on 3.13.
  - Runtime deps: `numpy`, `scipy`, `shapely`, `fastf1`, `arcade`, `platformdirs`.
  - Optional: `cma`. Extra `[rl]`: `gymnasium`, `stable-baselines3`. Extra `[dev]`: `pytest`, `ruff`.
  - Dropped from the explicit list: `pymunk`, `flake8`, and the pinned transitive deps.
- **Git clean-up:**
  - `git rm -r --cached cache src/__pycache__ src/.DS_Store`, plus the `.gitignore`.
  - Removing files from tracking doesn't shrink history. The initial commit still holds 178 MB, and dropping that means rewriting the single initial commit and force-pushing. Decided: no rewrite (section 12).
- **Tests** (`pytest`, all offline, using fixtures):

| Test | What it checks |
|---|---|
| track builder | closed loop; `is_simple` walls; Silverstone length within 1% of 5891 m; total widths 10–25 m; resampling spacing ±1% |
| projection | `s` round-trips along the centre line; `n` sign convention; correct across the wrap |
| raycasts | match analytic results on a circular track within 1 cm |
| car model | straight-line top speed 92 ± 2 m/s; 300→80 km/h braking distance in the expected range; steady-state cornering g at 2 speeds |
| laps | one lap per loop on an oval; zig-zag can't fake a lap; wrong-way is caught anywhere |
| determinism | same seed gives identical `metrics.csv` for 3 generations |
| checkpoints | round-trip, and refuses a mismatched `obs_spec_version` |
| smoke | 5 generations on the oval fixture finish in < 30 s |

- **Benchmark:** `scripts/bench.py` reports steps/s for N = 1, 16, 64 and 256 cars.

## 10. Milestones

Each milestone ends with every test passing, and is committed on its own.

| # | Milestone | Scope | Acceptance criteria | Size |
|---|---|---|---|---|
| **M0** | Repo cleanup and tooling | `.gitignore`, untrack generated files, `pyproject.toml`, ruff, pytest skeleton, real README, `docs/data-formats.md` | `git status` stays clean after running the app; `pip install -e .[dev]` works; `ruff check` and `pytest` pass | 0.5–1 day |
| **M1** | Track pipeline | `tum_source`, `track_builder`, `Track`, `alignment`, track JSON, `f1sim tracks`, `f1sim build-track silverstone`, debug plot | Silverstone: length 5891 ± 1%, walls simple (0 crossings, down from 31), widths 10–25 m; Russell's and Hamilton's 2024 laps are ≥ 98% within 1 m of the edges and median ≤ 6 m from the centre line; at least 5 TUM tracks build | 2–3 days |
| **M2** | Simulation core | `car_model`, `progress`, `sensors`, `grid`, `world`, `scripted`, a minimal Drive viewer, benchmark | All sim tests pass; scripted reference lap at Silverstone within ±10% of 1:25.8; ≥ 50× real time with 64 cars headless (today: 6.6× with 30) | 3–4 days |
| | **M2 result (2026-10-20)** | Done | All sim tests pass (303). Scripted lap on Silverstone's TUM raceline 91.24 s, +6.3% against 85.819 s, on track. 64 cars at 64.1× real time with numba (17.6× numpy only). Lap and sector timing are interpolated within the step | |
| **M3** | Training | `policy`, `ga`, `es`, `trainer`, checkpoints, metrics, `f1sim train` / `replay` | Silverstone: first clean full lap within 30 min of wall time; best lap time improves across the run; no flat best-fitness run longer than 20 generations without σ adapting; same seed reproduces the run | 3–4 days |
| **M4** | App rebuild | Replay, Watch-training and Drive modes, camera, HUD, ghost driver picker, live delta, background loading; delete old `src/`, `main.py`, `generate_track.py` | Loading any supported track never freezes the UI; ghost stays inside the track; 60 fps with 64 cars; the old code is gone | 3–5 days |
| **M5** | Breadth and polish | FastF1-spread fallback tracks, multi-track training and evaluation, PPO comparison, speed-trace charts, GIF/video export | The fallback builds Monaco and Baku within the M1 criteria; an agent trained on 3 tracks completes a lap on an unseen TUM track | ongoing |

Order: M0 → M1 → M2 → M3 → M4, with M5 items picked up afterwards. M2's minimal Drive viewer means the physics can be felt before M4.

## 11. Risks

| Risk | Mitigation |
|---|---|
| TUM layouts are outdated for some years | Alignment acceptance per year (4.3), with automatic fallback to FastF1-spread widths |
| Physics tuning drags on | Calibrate against one number (the scripted lap vs pole) and accept ±10% |
| Vectorised raycasts too slow | Uniform grid + batched maths; if still short of 50×, drop to 7 rays or move to numba (optional dep) |
| arcade draw cost with many cars | Cached static shapes, a sprite list for cars, sensors only on the followed car |
| FastF1 rate limits / offline | Cache under `paths.py`; the app works offline for already-built tracks |
| **F1 live timing needs sign-in (seen 2026-10-20):** `livetiming.formula1.com` returns HTTP 401 without an F1 TV subscription (FastF1 3.9 alpha adds the sign-in) | **Decided:** reference laps come from OpenF1 (free, same raw positions, 2023 onwards) via `data/laps.py`. FastF1 stays for older seasons and only works for cached sessions. FastF1 3.9 with F1 TV can be added later behind the same `LapXY` |
| LGPL data redistribution | Download at build time from a pinned commit; attribution in `THIRD_PARTY.md`; derived files aren't committed |

## 12. Decisions (settled 2026-10-20)

1. **Git history:** don't rewrite it. Stop tracking the cache from M0 onwards; the 178 MB stays in the initial commit.
2. **Learning approach:** neuroevolution (GA, then CMA-ES) is the main track. PPO is an optional comparison in M5.
3. **Leaving the track:** the car crashes. The `track_limits = "runoff"` option is deferred.
4. **Processed track files:** built on demand into the user cache and not committed. Only small test fixtures live in the repo.
5. **Python floor:** 3.11+, tested on 3.13.
