# Data formats

Canonical spec for every file f1sim reads or writes. Code and this doc change together: bump the relevant `version` / `obs_spec_version` when a format changes.

Status: specified, not yet implemented (tracks and ghosts arrive in M1, checkpoints and runs in M3). See [revamp-plan.md](revamp-plan.md).

All generated data goes in the user cache folder (`paths.py`), not the repo. The only exception is the small test fixtures. The formats are JSON and NPZ, loaded with `allow_pickle=False`, so there are no pickles.

## 1. Track file: `tracks/<track_id>.track.json`

```json
{
  "format": "f1sim.track",
  "version": 1,
  "id": "silverstone",
  "name": "Silverstone Circuit",
  "fastf1_location": "Silverstone",
  "source": {
    "geometry": "tum-racetrack-database",
    "commit": "<pinned sha>",
    "license": "LGPL-3.0",
    "widths": "tum"
  },
  "resolution_m": 2.0,
  "length_m": 5891.4,
  "direction": "clockwise",
  "sectors_s": [0.0, 1854.0, 3786.0],
  "points": {
    "x": [], "y": [], "s": [],
    "w_left": [], "w_right": [],
    "heading": [], "curvature": []
  },
  "fastf1_alignment": {
    "years_validated": [2024],
    "reference_driver": "RUS", "reference_lap_s": 85.819,
    "rotation_rad": 0.0, "tx_m": 0.0, "ty_m": 0.0, "scale": 1.0, "mirrored": false,
    "median_dist_m": 0.0, "inside_ratio": 0.0,
    "inside_ratio_tol": 0.0, "edge_tolerance_m": 4.0
  },
  "built_with": {"f1sim": "0.1.0", "builder_version": 1}
}
```

- Points are stored as columns (a few thousand per track, which is fine as JSON).
- Walls aren't stored; they're derived from the points when the track loads.
- **Version 2 adds track limits (optional).** The columns `points.limit_left` and `points.limit_right` hold how far cars may drive on each side, and a `limits` block records how they were made: `source`, `years`, `laps`, `lap_percentile`, `car_half_width_m`, `margin_m`, `max_extra_m`.
  - **What they are:** `f1sim.data.track_limits` builds them from every driver's fastest qualifying lap of those years, smoothed like the ghosts and projected onto the track.
  - **Bounds:** they are never inside the white line (`w_left`/`w_right`), and never more than `max_extra_m` beyond it.
  - **Who uses them:** the simulation drives on the limits (`Track.sim_track()`, `WorldConfig.use_track_limits`). Drawing uses `w_*` for the white line, and the band between the white line and the limit is the kerb and run-off.
  - **Older files:** version 1 files load without limits.
- **Tracks built from laps** (`f1sim.data.lap_geometry`, D40) have `source.geometry` `"openf1-laps"` and `source.widths` `"lap-envelope"`, plus `license`, `laps`, `years`, `min_width_m` and `approximate: true`. Their edges are where real cars drove plus a margin (no separate limits), and their `fastf1_alignment` is the identity, with `reference_year`.
- `fastf1_alignment.lap_source` says which source served the reference lap ("openf1" or "fastf1"). Both use the same raw frame.
- `fastf1_alignment` maps FastF1 metres (X/Y ÷ 10) into the track frame: `p' = scale · R(rotation_rad) · diag(1, −1 if mirrored else 1) · p + (tx_m, ty_m)` (`f1sim.data.alignment.apply_alignment`).
  - `inside_ratio` counts the reference lap's samples strictly inside the edges.
  - `inside_ratio_tol` counts them within `edge_tolerance_m` of the edges.
- `builder_version` bumps whenever the build algorithm changes, so stale cached files rebuild automatically.

## 2. Track catalogue: `data/catalog.py` (in code) plus `catalog.json` in the cache

This maps a `track_id` to the TUM file name, the FastF1 `Location` values (more stable than `EventName`), the layout years each source is valid for, and its availability (`tum`, `fastf1-spread`, `unsupported`). The UI's track picker reads this, so it never offers a track that can't be built.

## 3. Ghost lap: `ghosts/<year>_<track_id>_<session>_<driver>.npz` + `.json`

| Array | Unit | Notes |
|---|---|---|
| `t` | s | from lap start, 20 Hz |
| `x`, `y` | m | already in the track frame |
| `s` | m | projected onto the centre line, monotonic |
| `speed` | m/s | |
| `throttle` | 0–1 | |
| `brake` | 0/1 | |
| `gear` | int | |

The JSON sidecar holds `driver`, `team`, `lap_time_s`, `year`, `session`, `lap_source` ("openf1" or "fastf1") and `track_build`. The live delta is `t_ai(s) − t_ghost(s)`, interpolated on `s`.

**Smoothing is required.** Raw positions arrive at about 3.7 Hz. Joining them with straight lines (as the legacy app did) makes speed change in steps: Hamilton's 2024 Silverstone ghost hit 170 g at the 99th percentile and up to 470 km/h. Instead:
- Fit `x(t)` and `y(t)` with a smoothing spline (`scipy.interpolate.make_smoothing_spline`, which picks the amount of smoothing itself). On that lap this gives at most 4.2 g and follows the car's speed sensor to a median 7 km/h.
- Evaluate the spline at 20 Hz.
- Take `speed` from the spline's derivative, cross-checked against the speed sensor (`car_data` / OpenF1 `car_data`).
- Take the heading from the spline tangent, so the ghost doesn't twitch between samples.

## 4. Agent checkpoint: `runs/<run_id>/checkpoints/<gen>.npz` + `meta.json`

- **NPZ:** `params` (a flat float32 vector) plus optional optimiser state (CMA-ES mean, covariance and sigma).
- **`meta.json`:** `obs_spec_version`, `policy` (layer sizes, activations), `car_params_hash`, `track_id`, `seed`, `generation`, `fitness`, `best_lap_s`, `git_commit` and `created`.

When loading, the code checks that `obs_spec_version` and `policy` match the current code and refuses to load with a clear message if they don't.

## 5. Run folder: `runs/<run_id>/`

| File | Contents |
|---|---|
| `config.toml` | Fully resolved copy of the config, for reproducibility |
| `metrics.csv` | `gen, wall_s, best_fitness, mean_fitness, completion_rate, best_lap_s, best_progress_m, sigma` |
| `best_lap.npz` | Trace of the best lap (same columns as the ghost) for replay and overlays |
| `checkpoints/` | Agent checkpoints (section 4) |
| `generation.npz` | The latest generation, replaced atomically after each one (`write_generations`, default on) for the app's training view: `gen`, `stage`, `params` (P, n_params) float32, `fitness`, `start_s` and `start_v` (P,) (each genome's first episode start), `episode_s`, `lap_time_s` (nan = no lap), `progress_m` |
| `train.log` | Console output, when the app started the run |

## 6. Observation spec v2 (`sim/sensors.py`)

| # | Feature | Normalisation |
|---|---|---|
| 0–8 | 9 rays at −90, −60, −35, −15, 0, 15, 35, 60, 90° (range 200 m) | ÷ 200 |
| 9 | speed | ÷ 92 m/s |
| 10 | heading error (yaw − track heading at s) | ÷ π |
| 11 | lateral offset n | ÷ half-width on that side |
| 12–16 | curvature at s + 20, 50, 100, 150, 250 m | × 50, clipped ±1 |
| 17–18 | previous action (steer, pedal) | already in −1..1 |
| 19 | offset from the racing line (+ = car left of it) | ÷ 5 m, clipped ±2 |
| 20 | heading error to the racing line | ÷ 0.5 rad |
| 21–24 | racing line's lateral position 10, 30, 60, 100 m ahead along it, car frame (+ = left) | ÷ 20 m, clipped ±2 |
| 25–28 | racing line's profile speed 0, 50, 100, 200 m ahead | ÷ 92 m/s |

The racing line (`sim/raceline.py`) is the scripted driver's path. It's the TUM raceline, pulled
0.3 m inside the off-track limit, or the centre line for circuits without one. Its speed
profile is computed for the car's parameters. v2 added 19–28; 0–18 are v1 unchanged, and v1
checkpoints don't load.

**Actions:** `steer ∈ [−1, 1]` sets the steering target (rate-limited). `pedal ∈ [−1, 1]`: values above 0 are throttle and below 0 are brake. Changing any of this bumps `obs_spec_version`.

## 7. Loader asset: `src/f1sim/app/assets/geo/<track_id>.npz` (`f1sim.geo` v1)

A small, committed asset per circuit for the "track being drawn" loading animation
(`scripts/geo/loader_demo.py`, D49). Baked at build time by `scripts/geo/bake.py` from
OpenStreetMap and AWS Terrain Tiles (attribution in `THIRD_PARTY.md` and the `attribution`
field). NPZ, loads with `allow_pickle=False`; Silverstone is 45 kB.

| Array | Type, shape | Notes |
|---|---|---|
| `format`, `version` | str, int | `"f1sim.geo"`, 1 |
| `track_id`, `attribution` | str | |
| `anchor_latlon` | f8 (2,) | origin of the local east/north frame (EPSG:3857 × cos lat0) |
| `track_to_local` | f8 (3, 3) | similarity, track metres → local metres: `p' = s R p + t` |
| `fit` | f8 (6,) | RMS, p95 and max distance (m) of the centre line to the OSM ways; scale; rotation (deg); RMS to the OSM loop |
| `centre_xy` | f4 (n, 2) | centre line, every 4 m, track frame |
| `length_m` | f8 | track length |
| `elev_base_m` | f8 | lowest point of the centre line, metres above sea level |
| `elev_rel_m` | f2 (ceil(length),) | centre-line height above the base, every metre from s = 0, smoothed over 15 m |
| `grid_origin`, `grid_step_m` | f8 (2,), f8 | lower-left corner and spacing of the terrain grid (track frame) |
| `grid_rel_m` | f2 (ny, nx) | terrain height above the base; row = y, column = x |
| `osm_xy` | f4 (m, 2) | every OSM raceway way inside the grid, every 10 m, NaN rows between ways |
| `loop_xy` | f4 (k, 2) | the circuit's loop found in the OSM ways alone (the cycle whose length is nearest the track's), every 6 m, from the start line in the driving direction |

`loop_xy` and the grid need no built track, so a circuit that hasn't been built can be drawn
from them. Not implemented yet: `bake.py` still needs the built track; for an unbuilt one it
would take the circuit length instead and skip the centre-line fit and profile.
