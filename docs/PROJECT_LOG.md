# Project log

## Current state

*Last updated: 2026-10-25 (2026 sprite, fixed-size coloured field, D31). This section summarises
where things stand right now. Rewrite it, don't append to it, whenever something important
changes.*

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
- **The app is one window** (`f1sim menu`, D19–D22). The menu is its home screen; Drive and
  Replay switch the window to those screens, and Esc comes back. Train opens a console. The
  window remembers its place, but only restores it if it lands on a connected screen.
  Train starts `f1sim train` in the background and switches to the training screen.
- **The menu** is a dark hero card: track picker and buttons on the left. On the right is a
  metallic-black Ferrari SF-26 seen from above (55% of the window high, no logos), driving
  endless laps of Marina Bay at Russell's 2025 pole-lap speed. It keeps overtaking five
  slower cars in gunmetal, graphite and satin grey, and red and blue lamps flash by the
  track (D24, D25). The car stays fixed while the track moves and turns underneath it;
  every car's front wheels steer. One light model lights the cars, the floor and the track
  lines: photographed diffuse light from a Blender light stage, live GGX reflections, and
  the cars' shadows cast away from the cursor (D25). Without the driving set the menu falls
  back to the still light stage, then relit maps, then the car drawn in code.
  A red and blue backdrop runs from the window's left edge, under the menu and cars (D29).
  The mouse wheel dims or brightens the cursor light (0.25–1.10 of the default, D28).
  `python scripts/dev_menu.py` relaunches the app whenever the source or the renders change;
  `scripts\run_latest.bat` pulls the latest version first, then opens it.
- **Viewers:** the drive and replay screens use the dark single-red-accent UI, with kerbs and
  pixel-art cars. Every corner has an apex and an exit kerb (D18). Motion is smooth: Hermite
  poses, interpolated drive physics, and a 120 Hz tick with vsync (D15).
- **Training:** `f1sim train` first copies the scripted driver into the policy (about 65 s),
  then evolves it with the GA (D17). After 20 min the best Silverstone laps are 85.10, 85.45
  and 85.53 s (seeds 0–2). That beats the 86.43 s reference lap and Russell's 85.82 s pole;
  before D16/D17 the best was 100.7 s. The policy sees the racing line (observation spec v2).
  `--init random` (`configs/train_scratch.toml`) still trains from scratch. The same seed
  reproduces `metrics.csv` exactly.
- **Training view** (D30): the trainer replaces `generation.npz` every generation; the app
  re-drives the newest one with every genome on track (exactly the trainer's episodes; a
  test checks lap times match). Cars keep one size at every zoom, each in its own colour,
  in the 2026 sprite (D31). Opens fitted to the circuit; wheel zoom, right-drag pan,
  `Z` fit, `F` follow the best genome, Up/Down 0.5–16×, `N` newest. Leaving the screen
  stops the training. `f1sim watch <run>` opens it on any run folder. Checked in the real
  window on a live Silverstone run.
- **Tests:** 602 pass offline. `ruff check` is clean.

**Open problems, most important first:**
1. **Training needs a racing line.** Spec v2 observes a TUM raceline, or the centre line where
   a circuit has none. The 9 circuits without TUM data (open problem 4) would train against
   their centre line, a poor line. Computing a minimum-curvature line for them is the fix.
2. **The 2× UI is checked in simulation only.** Text is sharp at any scale (D26), verified
   in a simulated 192-dpi window. A look on the user's real 2× display is still worth doing.
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

**Next step:** the user tries the training view (menu → Train). Likely tweaks: the
default playback speed (`training.DEFAULT_SPEED`), the episode cut-off
(`LEADER_GRACE_S`), and whether leaving the screen should keep training running. In stage 2
the cloned field drives as one tight pack, so even with distinct colours it shows as a
single clump; a "spread" option (staggered starts) would be only for show and break the
exact match (D31). The new 2026 sprite (D31) is also in drive and replay; worth a look
there. Then
the rest of M4 (settings) and M5 (breadth, including a computed racing line for circuits
without TUM data). The menu's `BACKDROP_ALPHA` 0.30 is the user's own uncommitted tweak.

**Gotchas:**
- **The hero photos ship with the app** in `src/f1sim/app/assets/hero/stage` (45 WebP photos,
  1.2 MB, D23) and, for the driving hero, `assets/hero/drive` (280 WebP, 2.13 MB: light
  photos and data maps, D25) plus the
  circuit `assets/hero/marina_bay.npz` (D24; rebuild it with
  `python scripts/hero/build_circuit.py`, which needs OpenF1). The SF-26 model (the user's download) and the full-size renders stay local in
  git-ignored `assets_src/`. To change the hero, re-render, then export and commit the folder:
  `blender -b --factory-startup -P scripts/hero/render_top.py -- assets_src/sf26/extracted/source/f1_2026_sf26.glb assets_src/sf26/render --blackout --stage`
  (Blender 5.2 at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`, not on
  PATH; about 3 min on the GPU, the `--drive` set about 15 min), then
  `python scripts/hero/export_stage.py`. For the driving set use `--drive` instead of
  `--stage`, then `python scripts/hero/export_stage.py --drive`.
- **No screenshots or comparison images in the repo.** The user asked for that explicitly.
  Keep them in a scratch folder.
- **Don't truncate Blender's output in PowerShell.** `| Select-Object -First N` stops the
  pipeline once it has N lines, which kills Blender mid-render.
- **Cache redirection.** On this Windows machine, Claude Desktop's tool processes run inside an
  app package. Writes under `%LOCALAPPDATA%` are redirected to
  `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local`. Tracks built by agents don't
  appear in the user's own terminal until copied to `%LOCALAPPDATA%\f1sim` (decision D9).
- **F1 live timing needs a sign-in.** It returns 401 without an F1 TV subscription, so FastF1 only
  works for sessions already in its cache. Use OpenF1 for 2023 onwards (D6).
- **Rendering quality:** `F1SIM_QUALITY=high|medium|low` forces a tier (default `auto`; the
  log says which tier it picked and why). `F1SIM_GL_DEBUG=1` turns pyglet's GL error checks
  back on, for debugging GL. This laptop runs Python on its integrated Intel Arc, not the
  RTX (so the app starts at medium here).
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
bare.

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

**D22 · One app window: menu, drive and replay are arcade Views** (2026-10-22)
Choosing Drive or Replay used to start `f1sim drive` or `f1sim replay` as a new process, so a
second window opened. Each viewer was written as its own `arcade.Window`.
- **`app/shell.py`:** one `AppWindow` holds the frame pacing (the 1/120 s tick and vsync,
  moved there from replay.py) and the remembered place and size.
- **`Screen`:** an `arcade.View` that provides `ctx`, `get_pixel_ratio` and `close`. So
  `DriveView`, `ReplayView` and `MenuView` are the old windows with their drawing code
  unchanged.
- **Navigation:** the menu shows the drive or replay screen in the same window, and Esc
  there calls back to the same menu view. Its 128 MB of light-stage textures stay loaded,
  and a newly finished run shows up on return.
- **`f1sim drive` / `f1sim replay`** open the same screens on their own, where Esc quits.
  The footer says "Esc back" or "Esc quit" to match.
- **Train stays a console process:** it's a long headless job.
- **Fix the same day:** the app opened invisibly at (−32000, −32000). That's where Windows
  parks a minimised window, and the app had saved it as the window's place. It now never
  saves a parked position, and only restores a place where at least 120 px of the title
  strip lands on a connected screen.

**D23 · The photographic hero ships with the app; no images of ours in the repo** (2026-10-22)
The user wanted the Blender render permanently in the app, not only on this machine.
- **`scripts/hero/export_stage.py`** writes the light stage to `f1sim/app/assets/hero/stage`
  as WebP at quality 95: 45 photos plus the at-rest one, 1.2 MB in all. The black car
  compresses well, and the mean error is under half a level out of 255. The folder is
  package data, so it's in the wheel too.
- **`light_stage`** loads the bundled photos by default (`F1SIM_HERO_DIR` overrides). Its
  decoded cache now lives in the user cache folder, never in the repo.
- **Removed `docs/hero-finish-compare.png` and `docs/kerb-audit-silverstone.png`** at the
  user's request: no screenshots or comparison images in the repo. `images/screen.png` is
  the user's own README screenshot.
- **History rewritten the same day,** with the user's go-ahead: `git filter-branch` removed
  both images from every commit on `main` from the kerbs commit on, and `main` was
  force-pushed. The final files didn't change. The commit hashes in this log are the new
  ones.

**D24 · The hero car drives: endless laps of Marina Bay, steering front wheels** (2026-10-22)
The user asked for the hero car to drive Singapore, with the camera fixed on it and the
track moving underneath.
- **Circuit** (`scripts/hero/build_circuit.py`): Singapore isn't in TUM, so it comes from
  Russell's 2025 pole lap via OpenF1 (1:29.158, 351 positions at 3.9 Hz).
  - Racing line: a periodic spline through the positions, one point per metre (4,867 m).
  - Speed: from the speed sensor, 66–316 km/h. Driving the line at that speed takes 88.7 s.
  - Road: 12 m wide around a centre line shifted so the racing line sits 1.8 m off the
    inside edge at apexes.
  - Kerbs: the corner rule (D18), 19 corners.
  - Output: `assets/hero/marina_bay.npz`, 158 kB.
- **Photos** (`render_top.py --drive`): the whole car at 240 px/m, on 5 × 11 softbox
  positions, in two layers:
  - **Body:** everything but the front wheels. Its at-rest photo carries a soft ground
    shadow from a shadow-catcher plane.
  - **Front wheels:** split off by material (tyre, rim, disc), then everything inside their
    volume (rim covers, brake ducts, upright).
  - The softbox photos have no ground: their shadow was huge and swung with the light. The
    wheels cast no ground shadow, since it would stay straight while they steer.
  - Exported as WebP to `assets/hero/drive` (2.55 MB, the wheel photos cropped to their
    strip).
- **Motion** (`app/ui/hero_drive.py`): the rear axle follows the line at the real speed and
  loops forever; the heading is the line's direction.
  - Steering is `atan(wheelbase × curvature)` under the front axle, × 2.5 (the real 5° is
    invisible at menu size), capped at 28° and eased.
  - The track shader rotates the world so the heading points up. The car is 40% of the
    window high, at 72% across.
  - The track is full strength right of the menu and 22% behind it.
- **Composite** (menu shader): ground shadow, then the wheels (sampled rotated about their
  centres), then the body, all lit by the four softbox photos around the cursor.
  - The shadow fades within 0.45 m of the photo edges, which removed a faint rectangle.
  - The tyres are drawn 2.6× brighter, or the steering couldn't be seen.
- **Found by checking zoomed screenshots:** the steering direction was first reversed (map y
  points down), and the rim covers first stayed straight while the tyres turned.

**D25 · Driving hero round 2: a bigger car, one light model, traffic, trackside lamps** (2026-10-22)
The user asked for a bigger car, a cursor light that behaves like a real light, and slower
cars to overtake ([hero-round2-plan.md](hero-round2-plan.md)). Midway they added red and blue
flashing lamps by the track, sharper menu text, a car that looks solid rather than
translucent, and lamps that linger and sit at random.
- **Size:** the car is 55% of the window high (was 40%), its centre at 72% across and 42% up,
  so more track shows ahead.
- **Light faults confirmed first** (zoomed sweeps of the D24 shader): the light switched off
  just past the sidepod (the grid ended at ±0.9 m), the floor pool was a separate fake that
  didn't match the car, and blending four photos made two or four faint reflections fade
  into each other instead of one that slides.
- **Photos** (`render_top.py --drive`): every material white diffuse, no bounces, exposed so
  a white floor 1.9 m under the softbox reads 0.25 linear (a calibration render measures it).
  So each photo is the light falling on the car, in known units. The grid is 9 × 15 positions:
  ±1.8 m across, 1 m past both ends. The data maps (albedo, normal, surface, AO) are rendered
  for both layers too. Exported grey, 280 WebP, 2.13 MB.
  - Rejected: one diffuse grid per finish (3 × the size), and Cycles' DiffDir/DiffCol passes
    (they need the compositor).
- **Light model** (`app/ui/drive_shaders.py`, one model for cars, floor and track):
  - *Diffuse* = analytic softbox irradiance × the visibility the photos saw (their light
    over the unshadowed light they'd give). So the photos supply shadows and occlusion, and
    beyond the grid the light keeps falling off with distance instead of switching off.
  - *Reflections* are live GGX (base and clear coat) with the softbox as a sphere light
    (representative point, energy-normalised widening). One highlight that slides.
  - *Floor and track lines* get the same light; the cursor light casts the cars' shadows
    away from it (`p = (g(z−h) + Lh)/z`, height from the map, mip-blurred). The fake pool is
    gone; the beam stays.
  - *Contact shadow* moved into the floor pass and only darkens the floor's own dim light;
    drawn over the car's quad it cut into the cursor's pool as a visible rectangle.
  - *Finishes:* the bodywork (the clear-coat weight map) takes a per-finish colour scale,
    roughness, metal and coat, so one asset set draws every car.
- **The user's "translucent, not sturdy" note:** the at-rest coat sheen fell on every
  upward face and veiled the car in milky grey; it now lights only curved shoulders, so flat
  tops stay dark like gloss black. The photographed self-shadows are lifted 30%, and the cast
  shadow is softer and lighter (seen from above, a crisp one repeats the car's outline beside
  it like a ghost).
- **Traffic** (`hero_drive.Traffic`, pure numpy, seeded):
  - Five slower cars at 0.72–0.86 pace, in gunmetal, graphite and satin grey.
  - `Drive` has an offset from the line: a critically damped spring. The heading adds
    `atan(dd/ds)`, and the steering uses the shifted path's curvature, so the wheels turn
    into a dive.
  - `track_room` gives the room to each limit (shapely, 0.2 s).
  - The hero picks a side when a car is within 35 m: the inside of the next corner, or the
    side with more room. It moves up to 2.6 m over, while the cars in reach give 0.5–1.3 m.
    It holds the side until 8 m clear, and may still switch while the car is more than 15 m
    ahead.
  - The room is checked over the pass's real track distance:
    `(gap + clear + length) × v / closing speed`, plus the spring's settling distance.
    Measured in relative metres, the hero got trapped alongside and swerved into cars.
  - Every car follows any car ahead it isn't, or won't stay, clear of sideways (planned
    and current offsets). Cars are processed front to back.
  - Cars more than 25 m behind and out of view reappear 60–160 m ahead, out of view.
  - 10 laps × 8 seeds: no overlaps (oriented-box test), no car past the limits, an overtake
    every ~3 s. The hero waits behind 10–20% of the time, mostly in slow corners.
- **Lamps** (`trackside_lamps`): red and blue head pairs at random gaps (30–140 m), random
  sides and 0.6–4.5 m outside the limits. Each unit has its own phase and rate (0.8–1.25×).
  A double strobe, red then blue, each flash decaying over 0.11 s so it lingers on the
  passing cars. Up to 8 nearest are lit (small omni lights, no shadows), and the heads glow.
- **Sharp text:** arcade's HiDPI `stretch` mode magnifies 96-dpi glyphs on a 2× display. Text
  is now rastered at the framebuffer's DPI and drawn under a pixel projection, on whole
  pixels; at 1× the paragraph was also smeared by fractional positions.
- **Cost:** 4.6 ms a frame at 2560×1440 with six cars (was 3.6 ms at 1600×900).

**D26 · Sharp text everywhere: one wrapper renders at the framebuffer's DPI** (2026-10-22)
The user asked for sharp text across the whole app. arcade forces pyglet's HiDPI "stretch"
mode: layouts are in logical pixels, stretched onto the framebuffer, so pyglet's 96-dpi
glyph bitmaps were magnified 2× on the user's display (blocky and soft). D25 fixed the menu
by hand; the HUD has dozens of texts whose layouts read back measured widths.
- **`theme.SharpText`** wraps every `arcade.Text` (`make_text` and `micro` return it). It keeps
  the logical-pixel API the layouts use (`position`, `x`, `y`, `content_width`, the wrap
  `width`, `tracking`). Underneath it rasters at `96 × pixel ratio` dpi, sits on whole
  framebuffer pixels and draws under a framebuffer-pixel projection. It re-syncs when the
  pixel ratio changes (the window moved to another display). No call site changed; the
  menu's own scaling from D25 went.
- **Rejected:** pyglet's "scaled" mode (every view's projection and input would change), and
  scaling positions per call site (the HUD measures text to place the next item).
- **Checked** by making pyglet report a 192-dpi (and 144-dpi) screen, which opens a real
  2× (1.5×) stretch window: before/after crops of the menu, drive and replay HUDs. At 1× nothing
  changes. The pixel-art cars stay pixel art on purpose.

**D27 · Lighter on slow machines: no GL error checks, batched text, quality tiers** (2026-10-22)
The user asked for the app to run smoothly on weaker devices. Measured first (GPU timer
queries per pass, cProfile per frame, startup), then fixed the largest costs:
- **pyglet's GL error checks off** (`f1sim/app/__init__.py`, before arcade loads;
  `F1SIM_GL_DEBUG=1` restores them). It called `glGetError` after each of ~1,800 GL calls a
  frame. That halved the Python calls per frame, and the old GPU timings had been inflated
  by the stalls.
- **Batched text** (`theme.text_batch`): sharp text (D26) swapped the projection twice per
  label, which cost more than drawing the label. Each HUD widget now draws its text in one
  swap. Drive-screen CPU draw: 11 → 3 ms.
- **Quality tiers** (`app/ui/quality.py`): high, medium and low set the menu floor's light
  buffer size, the lamps and shadows per car, and the glass MSAA and blur size.
  - The start tier comes from the GL renderer: integrated GPUs start at medium, software
    renderers at low. `F1SIM_QUALITY` forces a tier.
  - A governor steps down when the median of 90 frames is over 1/45 s. It settles for 3 s
    after a start or a step, ignores stalls over 0.25 s and pauses while minimised. It never
    steps back up (that would flicker).
  - It's fed from `AppWindow.flip`. A per-turn clock callback was tried first, but it kept
    pyglet's loop from sleeping: a busy core.
- **Menu floor at reduced resolution:** `FLOOR_LIGHT_FS` lights the floor (and its contact
  shadow) into a buffer of 0.25–0.5 × the logical window size. The floor and the track lines
  sample it, and its cost no longer grows with the display's pixel ratio. It looks the same
  (the light and shadows are soft).
- **Glass blur** taps follow its sigma (3σ) instead of a fixed 21.
- **Start-up 3.4 → 0.8 s:**
  - The light photos load at half size (they only give shadowing now): a quarter of the
    texture memory.
  - The decoded stage is cached in the user cache (`hero/drive_<key>.npz`, 46 MB; 0.05 s
    to load).
  - The menu no longer imports the track pipeline (numba) just to list built tracks
    (`paths.track_file`).
- **Measured in the real window on this laptop's integrated Intel Arc 140T** (Windows runs
  Python on it): every tier holds the 120 Hz tick on every screen. CPU per screen, before →
  after: menu 30 → 25%, drive 50 → 33%, replay 59 → 43% of a core.
- **Not tried:** a lower draw rate on 60 Hz displays (D15 found a 1/60 s timer hitches
  here).

**D28 · Menu: track lines back, a wheel-set light level, 30% more lamps** (2026-10-22)
- **Track lines and kerbs had vanished (D27 regression):** D27 stopped calling
  `_floor_uniforms` on the track program, the only place that set its `u_ppm`. At 0 every
  vertex collapsed onto the hero's axle, so nothing drew. `_draw_drive` now sets `u_ppm`
  itself. Checked in rendered frames: `u_ppm` 0.0 before, 73 px/m after, edges visible.
- **The mouse wheel sets the cursor light's level** (`Light.level`, `Light.scroll`): 5% of
  the default per click, from 0.25 to 1.10 of it (the user asked for "10% more" up and
  "75% less" down). The power eases to the level, so a click fades rather than jumps.
  Rejected: a ×1.1 step per click with no ceiling (the light blows out the car's finish).
- **Lamps:** `LAMP_GAP_M` 30–140 → 23–108 m, 51 → 66 units on Marina Bay (+29%). The
  8-lamp shader cap doesn't bind: 2–3 heads are near the window at a time.

**D29 · A red and blue backdrop from the window's left edge** (2026-10-23)
The user supplied a red and blue streak image for the hero. It ships as
`assets/hero/backdrop.webp` (480 × 360, 54 kB). `_BACKDROP_FS` draws it over the floor and
under the track and cars. It first covered only the area right of the menu column at 0.14;
the user then raised `BACKDROP_ALPHA` to 0.44 and asked for it at the far left. It now covers
the whole window, its left edge on the window's left edge (aspect kept, the right side cut
off), so its brightest streaks sit behind the menu text.

**D30 · Training view: the newest generation re-driven on screen** (2026-10-25)
The user asked to watch generations train in real time. Train in the menu now starts
`f1sim train` in the background (no console; output in the run's `train.log`) and switches
to the training screen (`app/training.py`, `app/ui/train_hud.py`); `f1sim watch <run>`
opens it on any run folder.
- **The trainer writes, the app re-drives.** After every generation the trainer replaces
  `generation.npz` (genomes, fitnesses, each genome's start, the episode cap). The screen
  loads the newest one into a World built like the trainer's (one car per genome, same `dt`,
  `action_repeat`, car and track limits) and drives it with the policy. The physics is
  deterministic: a test checks that the re-driven lap times and progress equal the
  trainer's stage-2 scores exactly. Stage 1 shows each genome's first episode.
- **Rejected:** drawing the trainer's own World (would hold training to 1×, ~64× slower);
  streaming poses over a pipe (more moving parts, and the trainer only evaluates in bursts);
  the old plan's "replay the latest best agent" (one car shows no population).
- **Cadence:** an episode ends when every car is out or finished, or 3 s after the first
  finish; then the newest generation loads. At the default 2× a Silverstone generation
  shows for about 45 s while dozens more train. Up/Down sets 0.5–16×, `N` skips ahead.
- **Only the latest generation is kept** (~100 kB with the default policy): a history of
  every generation would grow to hundreds of MB in a 30-minute run.
- **View:** opens fitted to the whole circuit (`Z` refits); wheel zoom at the cursor, right
  drag pan, `F` follows the generation's best genome (by the trainer's fitness, in the AI
  livery with the only label). Other cars are grey and fade once out. The HUD has the live
  order, the generation and stage, cars still running with the episode clock, and the best
  lap per generation as a chart.
- **Leaving the screen stops the training** (`TrainingProcess.stop`, also at exit), so no
  invisible process keeps a core busy; checkpoints and `best_lap.npz` are already on disk.
- `CarLayer.only_highlight_label`: 64 labels would bury the track.

**D31 · A 2026 car sprite; fixed-size, one-colour-per-car training field** (2026-10-25)
The user found the cars changed size with the zoom, looked alike, and that the sprite
should look more like a 2026 car (they shared a pixel-art reference).
- **Sprite:** `CAR_ART` is now 16 × 40 (was 12 × 30), drawn as left halves and mirrored.
  2026 traits: a narrow nose on a three-element front wing, narrow front tyres, wide rear
  tyres, Pirelli-yellow sidewalls, the halo round a 2-px cockpit, undercut sidepods into a
  coke-bottle floor, the airbox, a diffuser, a two-element rear wing with no beam wing, and
  the rain light. Used by every viewer. At 40 rows the minimum on-screen length is now 40 px
  (one art pixel per screen pixel), up from 30.
- **Fixed size** in the training view only (`CarLayer(fixed_length_px=40)`,
  `pixelcar.fixed_scale`): one physical pixel per art pixel at 1×, two at 2×. Drive and
  replay keep the real 5.6 m when zoomed in, where a fixed size would make the car shrink
  against the track.
- **Colours:** `theme.field_liveries(n)` gives each car its own primary: hues a golden turn
  apart, cycling vivid, pastel and deep shades, so consecutive cars are far apart in colour.
  The tower sliver is the car's colour. The best car keeps its colour and gets the light
  outline plus the `BEST` label (it used to be the only car in the AI livery).
- **Seen, not changed:** in stage 2 the field drives one line (the GA's small mutations of
  one clone), so the cars overlap as a single clump at any zoom. Spreading them out
  (staggered starts) would only be for show, and the replay would no longer match the
  trainer's episodes.

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
| 2026-10-21 | `888f482` | Observation spec v2 (racing line), clone-then-GA `f1sim train` (D17) |
| 2026-10-21 | `cd733ab` | Apex and exit kerbs at every corner (D18) |
| 2026-10-21 | `3c71f22` | Main menu with the cursor-lit top-down car, live-reload script (D19) |
| 2026-10-22 | `3a7863f` | Hero from a real model: SF-26 rendered top-down, blacked out (D20) |
| 2026-10-22 | `00a0201` | Photographic light-stage hero, metallic black (D21) |
| 2026-10-22 | `01cc87a` | One app window: menu, drive and replay as Views (D22) |
| 2026-10-22 | `6666cd2` | Fix: window restored off-screen after being minimised |
| 2026-10-22 | `0986f22` | Project log brought up to date; finish comparison image |
| 2026-10-22 | `d6dc3a7` | Hero photos bundled with the app; comparison and audit images removed (D23) |
| 2026-10-22 | `79d9167` | History rewritten without the two images; log hashes updated |
| 2026-10-22 | `3f0eb8b` | Driving hero: endless Marina Bay laps, steering front wheels (D24) |
| 2026-10-22 | `8e73dcb` | Round-2 driving-hero plan approved and saved to docs |
| 2026-10-22 | `2334f4f` | Driving hero round 2: bigger car, live light model, traffic, trackside lamps, sharp text (D25) |
| 2026-10-22 | `5bc7d70` | Sharp text across the app at any display scale (D26) |
| 2026-10-22 | `0753e60` | Lighter on slow machines: GL checks off, batched text, quality tiers, faster start (D27) |
| 2026-10-22 | `f06eb28` | `scripts/run_latest.bat`: pull the latest version and open the app |
| 2026-10-22 | `cd7d7ff` | Menu: track lines back, wheel-set light level, 30% more lamps (D28) |
| 2026-10-23 | `f49555c` | Faint red and blue backdrop behind the driving cars (D29) |
| 2026-10-23 | `c568d56` | Backdrop from the window's left edge, at 0.44 (D29) |
| 2026-10-25 | `1c28454` | Training view: the newest generation re-driven live, `f1sim watch` (D30) |
| 2026-10-25 | (this commit) | 2026 car sprite; fixed-size, one-colour-per-car training field (D31) |
