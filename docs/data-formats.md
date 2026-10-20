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
    "inside_ratio_tol": 0.0, "edge_tolerance_m": 1.0
  },
  "built_with": {"f1sim": "0.1.0", "builder_version": 1}
}
```

- Points are stored as columns (a few thousand per track, which is fine as JSON).
- Walls aren't stored; they're derived from the points when the track loads.
- `fastf1_alignment` maps FastF1 metres (X/Y ÷ 10) into the track frame: `p' = scale · R(rotation_rad) · diag(1, −1 if mirrored else 1) · p + (tx_m, ty_m)` (`f1sim.data.alignment.apply_alignment`).
  - `inside_ratio` counts the reference lap's samples strictly inside the edges.
  - `inside_ratio_tol` counts them within `edge_tolerance_m` of the edges.
- `builder_version` bumps whenever the build algorithm changes, so stale cached files rebuild automatically.

## 2. Track catalogue: `data/catalog.py` (in code) plus `catalog.json` in the cache

This maps a `track_id` to the TUM file name, the FastF1 `Location` values (more stable than `EventName`), the layout years each source is valid for, and its availability (`tum`, `fastf1-spread`, `unsupported`). The UI's track picker reads this, so it never offers a track that can't be built.

## 3. Ghost lap: `ghosts/<year>_<track_id>_<session>_<driver>.npz` + `.json`

| Array | Unit | Notes |
|---|---|---|
| `t` | s | from lap start, resampled to 20 Hz |
| `x`, `y` | m | already in the track frame |
| `s` | m | projected onto the centre line, monotonic |
| `speed` | m/s | |
| `throttle` | 0–1 | |
| `brake` | 0/1 | |
| `gear` | int | |

The JSON sidecar holds `driver`, `team`, `lap_time_s`, `year`, `session`, `fastf1_version` and `track_build`. The live delta is `t_ai(s) − t_ghost(s)`, interpolated on `s`.

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

## 6. Observation spec v1 (`sim/sensors.py`)

| # | Feature | Normalisation |
|---|---|---|
| 0–8 | 9 rays at −90, −60, −35, −15, 0, 15, 35, 60, 90° (range 200 m) | ÷ 200 |
| 9 | speed | ÷ 92 m/s |
| 10 | heading error (yaw − track heading at s) | ÷ π |
| 11 | lateral offset n | ÷ half-width on that side |
| 12–16 | curvature at s + 20, 50, 100, 150, 250 m | × 50, clipped ±1 |
| 17–18 | previous action (steer, pedal) | already in −1..1 |

**Actions:** `steer ∈ [−1, 1]` sets the steering target (rate-limited). `pedal ∈ [−1, 1]`: values above 0 are throttle and below 0 are brake. Changing any of this bumps `obs_spec_version`.
