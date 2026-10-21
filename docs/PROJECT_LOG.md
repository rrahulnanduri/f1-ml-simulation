# Project log

## Current state

*Last updated: 2026-10-21. This section summarises where things stand right now. Rewrite it,
don't append to it, whenever something important changes.*

**What it is.** AI cars learn to drive real F1 circuits by neuroevolution. The `f1sim` package
(`src/f1sim/`) has real track geometry, F1-like physics, headless training, real-driver ghost
laps, and arcade viewers for driving and replay. The legacy app (`main.py`, `src/*.py`) still
exists, but the new package replaces it.

**What works:**
- **Tracks:** `f1sim build-track <id> --limits` builds a circuit from TUM's centre line and
  widths, aligns it to a real qualifying lap, and adds track limits from 60 real laps.
  Silverstone is complete; 7 TUM circuits pass the 2024 alignment check.
- **Physics:** the scripted reference lap at Silverstone is 86.433 s, +0.7% against Russell's
  2024 pole of 85.819 s.
- **Simulation:** 64 cars at about 64× real time with numba.
- **Ghosts:** `f1sim ghost <track> <year>` builds a smoothed real-driver ghost.
- **Viewers:** `f1sim replay` and `f1sim drive` use the dark single-red-accent UI, with kerbs and
  pixel-art cars.
- **Training:** `f1sim train` completes its first Silverstone lap within 75–126 s of training.
- **Tests:** 510 pass offline. `ruff` is clean.

**Open problems, most important first:**
1. **Trained cars are slow.** Their laps are 151–155 s against the 86 s reference. Both runs
   stopped improving after about 8 minutes, with the GA's mutation size pinned at its 0.15 cap.
   M3's plateau criterion fails, and reproducibility from a seed hasn't been checked.
2. **The new UI is unchecked at 2× scaling.** It was built on a 1× screen. The user's display is
   2×, and the user said "for now it's good".
3. **Some circuits fail the 2024 alignment check.**
   - Shanghai, Spa, Mexico City and the Red Bull Ring fail for no known reason.
   - Yas Marina, Albert Park, Barcelona and Zandvoort fail because their layouts changed after
     TUM's survey.
   - Suzuka is rejected because its figure-eight centre line crosses itself.
4. **Circuits not in TUM need a fallback** (M5): Monaco, Baku, Singapore, Jeddah, Miami, Imola,
   Las Vegas, Lusail, Madrid.
5. **Drive mode's off-track rule still uses the white line**, not the track limits
   (`app/drive.py`).

**Next step:** make the trained AI faster (open problem 1). Then M4 (the full app) and M5
(breadth).

**Gotchas:**
- **Cache redirection.** On this Windows machine, Claude Desktop's tool processes run inside an
  app package. Writes under `%LOCALAPPDATA%` are redirected to
  `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local`. Tracks built by agents don't
  appear in the user's own terminal until copied to `%LOCALAPPDATA%\f1sim` (decision D9).
- **F1 live timing needs a sign-in.** It returns 401 without an F1 TV subscription, so FastF1 only
  works for sessions already in its cache. Use OpenF1 for 2023 onwards (D6).
- **Use the repo venv:** `.venv\Scripts\python.exe` (Python 3.13). Install with
  `pip install -e ".[dev,fast,es]"`.

## How to use this file

- **Start here.** Read "Current state", then the decisions that matter for your task. Deeper
  detail lives in `docs/revamp-plan.md` (design and milestones), `docs/data-formats.md` (every
  file format) and `docs/m1/m2/m3-interfaces.md` (module contracts).
- **Add a decision** (`D<n>`) whenever you make a choice someone could reasonably have made
  differently. Say what was decided, why, and what was rejected.
- **Add a change-log entry** for each major change, with its commit.
- **Rewrite "Current state"** at the end of every work session.
- Keep entries short; link to commits and docs instead of repeating them.

## Decisions

**D1 · Revamp into a new package instead of patching the legacy app** (2026-10-20)
The legacy "track" was one driver's lap (Russell's pole) with walls a fixed 25 px (29.7 m) either
side and 31 wall self-crossings. Its physics were in pixels (top speed about 2,560 km/h), and
training was stuck. Everything was rebuilt in SI units in `src/f1sim`, and the legacy code stays
until M4. See `docs/revamp-plan.md` section 1.

**D2 · Decisions settled at the start** (2026-10-20)
- No git history rewrite: the cache was untracked from now on, but the 178 MB in the initial
  commit stays.
- Neuroevolution first, with PPO only as an optional comparison.
- A car crashes when it leaves the track.
- Tracks are built on demand into the user cache, never committed.
- Python 3.11+.

**D3 · Track geometry comes from the TUM racetrack-database** (2026-10-20)
F1's data (FastF1, OpenF1, MultiViewer) contains car positions and one lap's line, but no track
edges. TUM gives a centre line plus satellite-measured left and right widths (LGPL-3.0),
downloaded at build time from a pinned commit. Rejected:
- one lap's line plus a fixed width (what the legacy app and f1-race-replay do);
- OpenStreetMap (inconsistent between circuits, usually no width).

**D4 · Telemetry is aligned to TUM by rigid ICP, accepted loosely** (2026-10-20)
The acceptance rule is a median distance of at most 6 m from the centre line, with at least 98%
of samples within 1 m of the edges. The plan's original rule (≤ 4 m, strictly inside) rejected
correct fits: real laps use the kerbs, and TUM's own racing line sits a median 5.1 m off centre.

**D5 · Work is done by parallel agents against interface contracts** (2026-10-20)
Each milestone was split into modules with fixed signatures (`docs/m*-interfaces.md`), built by
agents in separate git worktrees, then merged and integrated here. Worktrees start from
`origin/main`, so each agent first runs `git merge --ff-only main`.

**D6 · OpenF1 for 2023 onwards, FastF1 only for cached older sessions** (2026-10-20)
F1's archive started requiring F1 TV in October 2026. OpenF1 serves identical positions
(0.03 m apart after a time shift), free and without sign-in. Its lap start differs from
FastF1's by 0–0.18 s, varying per lap, so it isn't corrected. `data/laps.py` picks the source.

**D7 · Ghosts are smoothed with a spline** (2026-10-20)
Joining 3.7 Hz positions with straight lines made the legacy ghost lurch: about 170 g at the
99th percentile, and up to 470 km/h. `make_smoothing_spline` brings that down to at most 4–6 g
and tracks the speed sensor to 6–8 km/h.

**D8 · Physics: a kinematic bicycle with a friction circle that grows with downforce**
(2026-10-20)
Calibrated with a scripted driver on TUM's racing line. Tuned to `max_brake_g=3.0` and
`grip_base_g=1.9`, giving 98 m from 300 to 80 km/h and a reference lap of +0.7%.

**D9 · The cache lives in the user cache folder (`platformdirs`)** (2026-10-20)
`F1SIM_CACHE_DIR` overrides it. See the redirection gotcha under "Current state".

**D10 · numba is an optional speed-up** (2026-10-20)
It compiles the ray casts and the track lookup, taking 64 cars from 17.6× to 64× real time.
Tests check the compiled and numpy paths agree to within 1e-9, and `F1SIM_NO_NUMBA=1` forces
numpy.

**D11 · Training: GA with a curriculum** (2026-10-20)
Stage 1 is random starts with progress as fitness; stage 2 is timed laps. The first 30-minute run
failed: cars averaged 119 km/h and ran out of the 1.5 × T_ref time limit, with sigma stuck at
0.5. Fixed with:
- a stage-2 time limit of 3 × T_ref,
- a sigma cap of 0.15,
- a 16 × 16 policy (626 parameters),
- fan-in scaled starting weights.

After that, first laps came within about 2 minutes.

**D12 · Track limits from real drivers' laps** (2026-10-21)
The limits combine every driver's fastest qualifying lap from 2023–2025, smoothed like the
ghosts, using the 90th percentile across laps plus half a car and a 0.25 m margin. They're
clamped between the white line and 4 m beyond it, and stored as track format v2. The simulation
drives on the limits through `Track.sim_track()`. Drawing uses the white line, with the kerbs
between it and the limit. Ghost bodies past the limit fell from 19–22% to 0.0–0.3%.

**D13 · UI: dark design with one red accent, based on f1.audi.com.au** (2026-10-21)
- **Path there:** "pit wall noir" (A), then a combination of the A, B and D concepts, then a glass
  version, then this restyle.
- **Look:** charcoal `#131313` background, black 8 px cards with 20% white hairlines, one accent
  `#EB0D3F`, wide uppercase headlines (Archivo Expanded), tiny bold labels, a scrambling `NOW:`
  ticker, and a footer with the controls.
- **Kept:** the timing tower, the gap card, and the paging widget (sectors, speed trace, gap bar).
- **Not used:** Audi's logo and its proprietary Audi Type font.
- **Kerbs:** red and white, about 3 m long.
- **Cars:** 12 × 30 pixel-art sprites, drawn with nearest-neighbour sampling.

## Change log

| Date | Commit | Change |
|---|---|---|
| 2026-10-20 | `da7cf6d` | Legacy UI fixes: Load Track crash, year dropdown |
| 2026-10-20 | `b488d4c` | Revamp plan written (`docs/revamp-plan.md`) |
| 2026-10-20 | `f90f34a` | **M0:** packaging, ruff, pytest, `.gitignore`, cache untracked |
| 2026-10-20 | `9e8c35c` | **M1:** track pipeline, CLI (`tracks`, `build-track`, `plot-track`) |
| 2026-10-20 | `7b456b2` | OpenF1 source and lap-source routing (D6) |
| 2026-10-20 | `0320576` | **M2:** progress and laps, ray grid and observations, `World`, scripted driver, drive viewer, numba (D10) |
| 2026-10-20 | `641f566` | Car tune (D8) |
| 2026-10-20 | `d57b562` | **M3** pieces: policy, checkpoints, GA and CMA-ES, trainer, replay, ghosts, CLI `train`/`replay`/`ghost` |
| 2026-10-20 | `32cd6fd` | Training fix (D11) |
| 2026-10-21 | `5eabd79` | Track limits, track format v2 (D12) |
| 2026-10-21 | `674de72` | Glass UI merged |
| 2026-10-21 | `15adadf` | UI restyle, kerbs, pixel-art cars (D13); pushed to `origin/main` |
