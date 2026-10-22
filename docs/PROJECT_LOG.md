# Project log

## Current state

*Last updated: 2026-10-21 (observation spec v2, clone-then-GA training). This section
summarises where things stand right now. Rewrite it, don't append to it, whenever something
important changes.*

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
- **Main menu:** `f1sim menu` (D19, D20). A dark hero card with the menu on the left (track
  picker; Drive, Replay, Train, Quit open the existing tools). On the right third is a
  metallic-black Ferrari SF-26 seen from above, with no logos. It's photographic: 45
  path-traced Blender photos with a softbox at different spots, blended by the cursor
  position (D21). Without them it falls back to relit maps, then to the car drawn in code. `python scripts/dev_menu.py` relaunches the menu
  whenever the source or the renders change.
- **Viewers:** `f1sim replay` and `f1sim drive` use the dark single-red-accent UI, with kerbs and
  pixel-art cars. Every corner has an apex and an exit kerb (D18). Motion is smooth: Hermite poses, interpolated drive physics, and a 120 Hz tick
  with vsync (D15).
- **Training:** `f1sim train` first copies the scripted driver into the policy (about 65 s),
  then evolves it with the GA (D17). After 20 min the best Silverstone laps are 85.10, 85.45
  and 85.53 s (seeds 0–2). That beats the 86.43 s reference lap and Russell's 85.82 s pole;
  before D16/D17 the best was 100.7 s. The policy sees the racing line (observation spec v2).
  `--init random` (`configs/train_scratch.toml`) still trains from scratch. The same seed
  reproduces `metrics.csv` exactly.
- **Tests:** 542 pass offline. `ruff` is clean.

**Open problems, most important first:**
1. **Training needs a racing line.** Spec v2 observes a TUM raceline, or the centre line where
   a circuit has none. The 9 circuits without TUM data (open problem 4) would train against
   their centre line, a poor line. Computing a minimum-curvature line for them is the fix.
2. **The new UI is unchecked at 2× scaling.** It was built on a 1× screen. The user's display is
   2×, and the user said "for now it's good".
3. **Some circuits fail the 2024 alignment check.**
   - Shanghai, Spa, Mexico City and the Red Bull Ring fail for no known reason.
   - Yas Marina, Albert Park, Barcelona and Zandvoort fail because their layouts changed after
     TUM's survey.
   - Suzuka is rejected because its figure-eight centre line crosses itself.
4. **Circuits not in TUM need a fallback** (M5): Monaco, Baku, Singapore, Jeddah, Miami, Imola,
   Las Vegas, Lusail, Madrid.
5. **The AI car's heading wiggles a little.** That's real: the policy's steering chatters (yaw
   acceleration p99 6 rad/s², against 1.3 for a ghost). An action-change penalty in training
   would fix it; the viewer shows it faithfully.
6. **Drive mode's off-track rule still uses the white line**, not the track limits
   (`app/drive.py`).

**Next step:** iterate on the menu hero with the user (live reload running). Then M4 (the full app) and M5 (breadth, including a computed racing line for
circuits without TUM data).

**Gotchas:**
- **The hero renders aren't in git.** The SF-26 model (the user's download) and its renders
  live in `assets_src/` (git-ignored). On a fresh checkout the menu shows the drawn car until
  you run:
  `blender -b --factory-startup -P scripts/hero/render_top.py -- assets_src/sf26/extracted/source/f1_2026_sf26.glb assets_src/sf26/render --blackout --stage`
  (Blender 5.2 on this machine; about 3 min on the GPU for the 45 photos). Drop `--stage` for
  the relit-map fallback (about 45 s).
- **Don't truncate Blender's output in PowerShell.** `| Select-Object -First N` stops the
  pipeline once it has N lines, which kills Blender mid-render.
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

**D14 · GA step size by the 1/5th success rule, with a lower floor** (2026-10-21)
The old runs' completion rate was 6.25% = exactly the 4 elites: no mutated child ever finished
a lap. The "stale" rule *raised* sigma when the best stalled, pinning it at the 0.15 cap, where
a mutation wrecks a 626-weight policy. `step_rule = "success"` (default now) shrinks sigma while
fewer than 20% of children beat the worst elite and grows it above that. Also `sigma0 = 0.05`
and a new `sigma_min = 0.003`. 12-15 min runs: fixed sigma 0.02/0.04/0.08 gave 117/156/137 s;
success rule 117/109 s (floor 0.01) and 117/104 s (floor 0.003). Rejected: CMA-ES (no laps),
population 128 (149 s in 15 min). The old behaviour stays available as `step_rule = "stale"`.

**D15 · Smooth viewer motion** (2026-10-21)
Measured, not guessed: draw cost was fine (p95 10 ms) and update and draw stay paired (arcade's
`_dispatch_frame`). The jank had three causes.
- **Ghost poses** were linear between 20 Hz samples, so velocity jumped at each sample: 15–16 g
  between frames in fast corners. `PreparedTrace.pose` now uses cubic Hermite interpolation
  (slopes from `np.gradient`) for position and heading: at most 5.6 g, the real cornering load.
- **Drive physics** ran in fixed 1/60 s steps but was drawn raw, so a frame showed 0 or 2 steps.
  `DriveSession.render_pose()` and `render_lap_time()` interpolate by the leftover accumulator.
  The drawn car's speed now changes at most 1.6% between frames.
- **Frame pacing:** the 1/60 s timer with vsync off missed a refresh on 10–12% of frames. The
  driver ignores the vsync request here (windowed OpenGL, dual-GPU laptop), so the windows now
  tick at 1/120 s with `vsync=True`: no frames over 20 ms. Rejected: `DwmFlush()` after each
  swap (GLFW's approach), which gave 30 fps here. Rejected: a damped follow camera, since
  positions are smooth once interpolated, and lag would let the car drift off the stage centre.

**D16 · Behaviour cloning shows the inputs are the bottleneck** (2026-10-21)
`agents/imitation.py` holds:
- `Teacher`, the scripted driver as a vectorised 30 Hz controller with a gentler merge onto the
  raceline (`MERGE_GAIN`). From the training start it laps in 87.05 s.
- `collect`: DAgger data collection.
- `fit`: numpy Adam, with steering weighted 4×.

Experiment scripts are in `scripts/bc/`. Results on Silverstone, 8 DAgger rounds:

| Inputs | Network | Clone result | Student crashes from random starts |
|---|---|---|---|
| v1 (19 inputs) | 16 × 16 | crashes within 100–1,100 m | 80–100% |
| v1 | 64 × 64 | crashes by 2,840 m | 33% |
| v1 + 10 raceline inputs | 16 × 16 | **87.25 s lap** | 0–6% |

The ten raceline inputs are: offset and heading error to the raceline, the raceline's lateral
position 10, 30, 60 and 100 m ahead in the car frame, and the target speed 0, 50, 100 and
200 m ahead.

The v1 inputs (rays, centre-line offset, curvature) don't say where the racing line is, so the
policy can't drive it, however big. With the raceline inputs, the GA fine-tunes the clone to:
- **85.64 s** (sigma 0.005) or 85.87 s (sigma 0.002) in 20 min;
- 86.5 s within 2 min;
- 1.4 s quicker than the teacher and 0.8 s under `T_ref`.

From scratch, the best was 100.7 s. The cost: the policy depends on a precomputed raceline
(TUM's, or one computed for circuits without it), and observation spec v2 would invalidate v1
checkpoints. The fine-tune ran the existing trainer with the inputs patched in; the production
spec is unchanged until that's decided.

**D17 · Observation spec v2 and clone-then-GA training by default** (2026-10-21)
Follows D16.
- **Spec v2:** `sim/sensors.py` appends the ten racing-line inputs (indices 19–28, see
  data-formats section 6) from `sim/raceline.py`. That module holds the scripted driver's path
  and speed profile, cached per track and car. The projection onto the line is hinted per car;
  `World.reset` calls `Sensors.forget`. v1 checkpoints no longer load.
- **Training:** `TrainConfig.init` is `"clone"` by default. `Trainer.run` first runs
  `imitation.clone` (8 DAgger rounds), in a World of `population` cars. The stage-1 World's
  192 cars took over 6 min for no better clone. Then it seeds the GA with the clone plus
  `N(0, sigma0)` copies and starts in stage 2. The clone's time counts towards `max_wall_s`.
  The defaults become the fine-tuning steps: `sigma0` 0.005, `sigma_max` 0.05, `sigma_min`
  0.0005. `configs/train_scratch.toml` and `--init random` keep the old path.
- **Results**, 20 min on Silverstone, seeds 0–2:
  - cloning takes 65–68 s;
  - first lap after 71–74 s;
  - after 5 min 85.41 / 85.87 / 85.72 s;
  - after 20 min **85.10 / 85.45 / 85.53 s**.
- `scripts/bc/` (the D16 experiment, which patched the inputs in) was removed: the trainer does
  the same now (it's in commit `1b10432`).

**D18 · Kerbs from corner geometry as well as from drivers' laps** (2026-10-21)
Kerbs were drawn only where real drivers' laps ran more than 0.3 m past the white line. 10 of
Silverstone's 19 corners lacked an apex or exit kerb, including Farm, the Copse and Vale apexes
and The Loop. Likely reason: TUM's satellite edge often includes the kerb, so riding it stays
inside the white line. `render_track.corners` now finds corners in the smoothed centre-line
curvature: radius under 300 m, turning at least 15°, 17 at Silverstone. Each gets:
- an apex kerb on the inside, over the part tighter than 60% of its peak (at least 20 m);
- an exit kerb on the outside, from 40% of the way between apex and end to 15 m past the end.

They're at least 1 m wide, and merged with the data kerbs. This is drawing only: the track
limits are unchanged. Rejected: lowering the data threshold, which still leaves unused kerbs
bare. Before/after map: `docs/kerb-audit-silverstone.png`.

**D19 · Main menu with a cursor-lit, code-drawn top-down car** (2026-10-21)
The user asked for a main menu after a React "interactive 3D" hero card, with the robot
replaced by an unbranded F1 car.
- **First version:** a procedural 3D car that turned to face the cursor. The user then
  changed the brief: a still, top-down car, nose up, filling the right third from the nose to
  just behind the driver, lit only around the cursor.
- **Their reference images weren't usable:** 740 × 416 and 500 × 281 px, one with
  "FORMULA-1" on it, the other apparently F1's own render. So the car is drawn in code
  (`app/ui/car_top.py`), using the references for proportions only.
- **How it's drawn:** a height field, one plan shape and height profile per part, so a
  normal map comes from its gradient. The parts are the 4-element front wing, nose, tyres,
  wishbones, halo, helmet, sidepods with louvres, engine cover and carbon twill.
- **How it's lit:** `app/menu.py` lights it in a shader. A point light 1.6 m above the
  cursor gives diffuse light and a gloss highlight from each material, falling off within
  1.8 m. Ambient light is 5%, so the unlit car is just visible. The light also casts a faint
  pool on the floor, and fades in after the spotlight beam.
- **Rejected:** the user's images (resolution, branding, rights) and a flat image with a
  radial reveal (no shading).
- **Live reload:** `scripts/dev_menu.py` polls `src/f1sim` and relaunches the menu on a
  change. `F1SIM_MENU_STATE` keeps the window's place across restarts.

**D20 · Hero from a real model: the SF-26, blacked out, rendered top-down in Blender** (2026-10-22)
The code-drawn car looked fake, so the hero now uses a real model, the user's choice.
- **The model:** a Ferrari SF-26 in glTF from the user's downloads, 214k vertices, 5.53 m.
  Its texture names (`AC_*`, `.dds`) suggest an Assetto Corsa conversion. When asked, the
  user confirmed it's legitimate for this use. It isn't committed.
- **`scripts/hero/inspect_model.py`** reports the model's objects, materials and textures.
- **`scripts/hero/render_top.py`** puts the nose up and renders orthographic top-down
  images with Cycles at 700 px/m. Each data image comes from an emission shader per
  material, so no compositor is needed:
  - colour, sRGB-encoded;
  - surface direction (the model's normal maps included);
  - roughness, metallic and height;
  - ambient occlusion and clear coat.

  The data images are written raw. The first renders were display-encoded by mistake, which
  turned roughness 0.14 into about 0.41 and tilted every surface direction.
- **Logo geometry is deleted**, not hidden: hidden overlays still blacked out the occlusion
  under them. The harness texture's printed text is painted over.
- **The game conversion's flat materials are replaced** with realistic roughness, metal and
  clear coat (`SURFACES`).
- **`--blackout`**, the user's request: paint solid gloss black (roughness 0.14 under a
  clear coat), everything else greyscale ×0.22 (tyres ×0.18). No colour survives: the most
  saturated pixel is 0.008.
- **`app/ui/hero_maps.py`** crops the renders to 4.05 m behind the nose, fades the cut, and
  caches the result as `hero_maps.npz` (loads in 0.05 s).
- **The menu's shader is now physically based:** GGX highlights, a clear-coat lobe of
  roughness 0.07, occlusion-weighted ambient light of 1.2 %, and a filmic curve.
- **Rejected:** BlendSwap 18333 (1988 car, livery, rFactor conversion) and FetchCFD 1636
  (its NoAI clause).

**D21 · Hero as a light stage: path-traced photos blended by the cursor** (2026-10-22)
The user wanted it photorealistic, and blacker. The relit maps (D20) faked one bare point
light with no shadows or reflections of the light source, which looked off. So Cycles now
photographs the front 4.15 m of the car at 320 px/m:
- once per position of a 0.7 m disc softbox, 1.9 m up, on a 5 × 9 grid over the car
  (45 photos, 3.6 s each);
- plus an ambient photo, the car at rest under a broad dim light.

The menu uploads them as a texture array (128 MB) and blends the four photos around the
cursor bilinearly. Each photo is path-traced, so every light position has real soft
shadows, the softbox reflected in the clear coat, and occlusion. Off the grid (over the
menu, say) the light fades out within 0.9 m.

**The finish** is metallic black, now the default: base 0.02, metal 0.85, roughness 0.30,
under a clear coat. Carbon is greyscale × 0.12 and tyres × 0.12. Photos use AgX tone mapping
with the Punchy look.

**Tuning:** softbox 300 W, ambient light 60 W, at-rest photo shown at 0.35.

**Trade-off:** between grid points the highlights cross-fade rather than slide. With a soft
light that reads naturally; a finer grid costs memory (each photo is 2.8 MB).

**Rejected:** more shader tricks on the relit maps, which can't produce soft shadows or the
reflection of a real light.

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
| 2026-10-21 | `c7fa0f7` | GA success-rule step size, `sigma_min`, new training defaults (D14) |
| 2026-10-21 | `7294c93` | Smooth viewers: Hermite poses, drive render interpolation, 120 Hz tick (D15) |
| 2026-10-21 | `1b10432` | Behaviour cloning module, experiment scripts and results (D16) |
| 2026-10-21 | (this commit) | Observation spec v2 (racing line), clone-then-GA `f1sim train` (D17) |
| 2026-10-21 | `c6f1d33` | Apex and exit kerbs at every corner (D18) |
| 2026-10-21 | `893d94d` | Main menu with the cursor-lit top-down car, live-reload script (D19) |
| 2026-10-22 | `386dbb6` | Hero from a real model: SF-26 rendered top-down, blacked out (D20) |
| 2026-10-22 | (this commit) | Photographic light-stage hero, metallic black (D21) |
