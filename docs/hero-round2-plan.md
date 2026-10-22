# Driving hero, round 2: a bigger car, believable light, slower cars to overtake

*Status (2026-10-22): **done** (D25 in [PROJECT_LOG.md](PROJECT_LOG.md), which records what
was built, what changed on the way and the user's additions: trackside lamps, sharp text).*

**Build order** (each step is checked in the app before the next):
1. [x] Bigger car: `DRIVE_CAR_HEIGHT` 0.55, new `DRIVE_Y` 0.42 (`src/f1sim/app/menu.py`).
2. [x] Track widths, lateral offset in `Drive`, `Traffic` (pure numpy) + tests.
3. [x] Draw the slower cars: rotated-car vertex shader, the light moved into each car's frame.
4. [x] Confirm the three lighting faults below in zoomed screenshots (scratchpad only).
5. [x] Re-render: wider softbox grid, diffuse only with a white-ish base colour, data maps
   (`albedo`, `normal`, `surface`, `ao`) for body and front wheels, then
   `export_stage.py --drive`. Render command: the one in the log's Gotchas with
   `--drive` in place of `--stage`.
6. [x] Shader: per-finish albedo × photographed diffuse + live GGX coat/metal reflection,
   floor lit by the same light, shadow cast away from the cursor; retire the fake floor pool.
7. [x] Verify (below), log D25, commit. Ask before pushing.

## Context
The driving hero (D24) works. The user wants three more things:
1. A bigger car.
2. A cursor light that spreads over the car and around it like a real light.
3. Slower cars on track that the hero car keeps overtaking.

The user chose:
- Hero height: 55% of the window (now 40%).
- The slower cars: the same SF-26 shape, in other dark finishes (gunmetal, graphite, satin grey), so the black hero car stands out.

## Why the light looks wrong (from the code)
The shader `_DRIVE_FS` in `src/f1sim/app/menu.py` blends the four softbox photos nearest the cursor. The photos come from `render_top.py --drive`: the softbox is 0.7 m wide, 1.9 m up, and the grid spacing is 0.45 × 0.55 m.
- **Reflections jump.** Bilinear blending works for diffuse light, which changes smoothly. A reflection in the clear coat is a sharp patch at a different spot in each photo. Blending gives two or four faint copies that fade into each other instead of one highlight that slides.
- **Nothing lights the ground.** The softbox photos have no ground. The pool under the cursor is a separate fake, `_SPOT_FS`, with its own falloff. Its brightness doesn't match how bright the car gets, and the car casts no shadow away from the cursor. So the car looks pasted on.
- **The light switches off beside the car.** The grid only covers the car (x ±0.9 m). Beyond it the light fades out within 0.9 m (`STAGE_FADE_OUT_M`). Moving the cursor just past the sidepod turns the light off instead of lighting the car from the side.
- **Step 1 is to confirm these** in zoomed screenshots with the cursor moved in small steps, before changing anything.

## Plan

### 1. Light (diffuse from photos, reflections computed live)
- **Render more** (`scripts/hero/render_top.py --drive`):
  - Widen the softbox grid to about ±1.8 m across and 1 m past the nose and tail, so a light beside the car is a real photographed light.
  - Also render the data maps the static hero already uses (`albedo`, `normal`, `surface`, `ao`; `render_top.py` around line 723) for the whole car at drive resolution, as body and front-wheel layers.
  - Store the photos as the **diffuse pass only**: set the material's coat and specular to zero for the grid render, or use Cycles' DiffDir/DiffInd/DiffCol passes.
- **Compute the reflection live** (in `_DRIVE_FS`):
  - Reuse the GGX clear-coat and metal code from the relit-maps shader (the `_HERO_FS` in `menu.py`, reference `car_top.relight` in `src/f1sim/app/ui/car_top.py`).
  - Use a sphere-light approximation sized to the softbox, so the highlight slides continuously and keeps its soft shape. Occlusion comes from `ao`.
  - Diffuse stays photographed (bilinear blend works there). Specular is computed, so it is never blended.
- **Light the floor with the same light** (new pass between the floor and the track):
  - Same inverse-square and cosine falloff and the same intensity units as the car, so the pool matches the car's brightness.
  - The track lines and kerbs pick up the light too: multiply their colour by the floor light.
- **Cast shadow away from the cursor:** for a floor point g and a light at L (height z), the car point that blocks it is `p = (g·(z−h) + L·h)/z`.
  - Look up the car's coverage there, using height h from the `surface` map with one refinement step.
  - Soften with a mip-level lookup scaled by distance (wider light → softer edge).
  - The at-rest contact shadow stays.
- Replace the hard fade outside the grid with distance falloff only. Retire the fake floor pool in `_SPOT_FS`, keeping only the beam.

### 2. Bigger car
- Set `DRIVE_CAR_HEIGHT` to 0.55 in `menu.py`, and move the anchor down a little (new `DRIVE_Y`, about 0.42) so more track shows ahead, where the slower cars come from.
- The photos are 240 px/m, about 1,350 px for the whole car. At 55% of a 1440p window the car is about 790 px, so no re-render is needed for size.
- Check that `drive_layout()` keeps the car clear of the menu column on narrow windows (it shrinks to fit).

### 3. Slower cars and overtakes (`src/f1sim/app/ui/hero_drive.py`)
- **Track width per metre:** compute the distance from the line to the left and right edges at load (chunked nearest point in numpy; edges are already in the npz). This gives the free room on each side.
- **`Drive` gets a lateral offset** `d(s)`, metres left of the racing line, and a `pace` factor. Its pose follows `line + n·d`. The heading includes the slope `atan(dd/ds)`. Steering uses the curvature of the shifted path, so the wheels turn into a dive.
- **New `Traffic` class:**
  - About 5 slower cars at `pace` 0.72–0.86 of the real lap speed, each with its own finish and a small random wobble off the line.
  - **Endless overtakes:** once a slower car is more than about 25 m behind the hero, out of view, it reappears 60–160 m ahead, also out of view. Cars never pop in, and with the gap closing at roughly 8–15 m/s there's an overtake every few seconds.
  - **The hero's pass:** when a car is within about 35 m ahead, the hero picks a side: the inside of the next corner, or the side with more room if that's too tight. It eases its offset to about 2.6 m (within the track limits, at least 1 m from the edge).
  - It holds that side until about 8 m clear, then eases back onto the line. The slower car moves about 0.5 m the other way (it "leaves room").
  - **Deterministic:** a seeded RNG, so tests and screenshots repeat.
- **Drawing** (`menu.py`):
  - A new car vertex shader with centre, rotation and size, so the same `_DRIVE_FS` draws a rotated car anywhere.
  - Each car gets the cursor light moved into its own frame. That picks its softbox photos (`light_stage.blend_weights`) and its reflection direction, so one light lights every car consistently.
  - **Finishes:** one extra diffuse grid per finish would be heavy (3 × about 2.5 MB). Instead, render one diffuse grid with a **white-ish base colour** (an albedo-normalised irradiance grid). The shader multiplies it by each finish's albedo, and roughness/metallic/coat come from a small per-finish table. That's one asset set for every car, the hero included.
  - Order: floor light → track → every car's shadows → slower cars → hero. Cars off screen are skipped.

### 4. Assets, tests, log
- Re-run `render_top.py --drive`, then `export_stage.py --drive`, adding the data maps.
  - Keep the bundle under about 5 MB: diffuse is smooth, so a lower WebP quality is fine.
  - Keep renders and screenshots out of the repo, as before.
- **Tests** (`tests/test_hero_drive.py`):
  - The offset path stays inside the track limits.
  - The hero's heading and steering follow the offset.
  - Every slower car gets passed and reappears ahead, never in view.
  - No two cars overlap over a 10-lap simulation.
  - Track widths are positive and plausible (5–7 m each side).
- `docs/PROJECT_LOG.md`: D25 (light model, traffic, size) and the "Current state" section.

## Files
- `scripts/hero/render_top.py`, `scripts/hero/export_stage.py`: wider grid, diffuse-only base-colour-normalised photos, data maps.
- `src/f1sim/app/ui/hero_drive.py`: track widths, offset `Drive`, `Traffic`, finish table, stage loader for the maps.
- `src/f1sim/app/menu.py`: size and anchor, car vertex shader, `_DRIVE_FS` (albedo × diffuse + GGX coat + AO), floor-light and cast-shadow pass, light on the track, drawing several cars.
- `tests/test_hero_drive.py`, `docs/PROJECT_LOG.md`.

## Verification
- `pytest -q` and `ruff check`.
- The live-reload app (`scripts/dev_menu.py`): zoomed screenshots in the scratchpad (never the repo) with the cursor stepped across the nose, the sidepod, beside the car and over the track. Check:
  - one smooth highlight that slides;
  - the floor pool matches the car's brightness;
  - a shadow falls away from the cursor;
  - no seams at the grid edge.
- Record about 20 s of frames and check: overtakes keep coming, the hero stays inside the lines, its wheels steer into each dive, and no car pops in.
- Frame time stays at 120 Hz with 6 cars (watch the cost of the shadow lookup).
