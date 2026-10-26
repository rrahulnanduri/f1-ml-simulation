# F1 ML Simulation

AI cars learn to drive real Formula 1 circuits through neuroevolution, on real track geometry,
racing ghosts of real drivers' qualifying laps (from [OpenF1](https://openf1.org) and
[FastF1](https://docs.fastf1.dev/)).

## Status

`f1sim` (`src/f1sim/`) is the whole app; the old prototype is gone (M4). See
[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) for where things stand and
[docs/revamp-plan.md](docs/revamp-plan.md) for the design.

- **Tracks:** real circuit geometry in metres from the TUM racetrack-database, aligned to a real
  qualifying lap, with track limits from where real cars drive. Circuits TUM doesn't have
  (Monaco, Baku, Singapore, ...) are rebuilt from every driver's qualifying laps on OpenF1.
- **Physics:** an F1-like car in SI units; the scripted lap at Silverstone is within 0.7% of
  the 2024 pole.
- **Training:** the scripted driver is copied into a neural network, then evolved. After 20
  minutes the best Silverstone laps beat the real pole.
- **App:** one window: a menu with a 3D-lit hero car, Drive, Replay and a live training view.

| Milestone | Scope | State |
|---|---|---|
| M0 | Repo cleanup, packaging, tooling | done |
| M1 | Real track pipeline (TUM racetrack-database + lap alignment) | done |
| M2 | Vectorised simulation core in SI units | done |
| M3 | Headless neuroevolution training | done |
| M4 | New app; legacy code removed | done, settings to come |
| M5 | More circuits, multi-track training, polish | in progress |

## Setup

Requires Python 3.11+ (tested on 3.13).

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -e ".[dev,fast,es]"
```

The `es` extra adds the CMA-ES optimiser. The `fast` extra adds numba, which compiles the ray casts and track projection. With 64 cars,
the simulation runs at 64× real time with it and 17.6× without (`python scripts/bench.py`).

**Always run the latest version (Windows):** double-click `scripts\run_latest.bat`, or a
desktop shortcut to it. It pulls the latest commits, reinstalls only if the dependencies
changed, then opens the menu. Arguments go to `f1sim`, e.g. `run_latest.bat drive silverstone`.

## Run it

```bash
f1sim menu
```

## Build tracks

List circuits, then build one (TUM data aligned to a real qualifying lap, or real laps for
circuits TUM doesn't have):

```bash
f1sim tracks
```

```bash
f1sim build-track silverstone
```

Drive it yourself, or run the scripted reference lap that calibrates the physics:

```bash
f1sim drive silverstone
```

To race a real driver's lap, pass a ghost from `f1sim ghost` (it sets off when you cross the line):

```bash
f1sim drive silverstone --ghost path/to/ghost.npz
```

Set `F1SIM_NO_GLASS=1` to draw the HUD cards flat, without the offscreen scene and its blurred backdrop (for slow GPUs or debugging).

```bash
f1sim reference-lap silverstone
```

- **Without FastF1:** add `--no-fastf1` to build the geometry only, which skips the start line, sectors and alignment check.
- **Output:** tracks go to the user cache folder (`%LOCALAPPDATA%\f1sim\Cache` on Windows).
- **Known gaps:**
  - Reference laps come from OpenF1 for 2023 onwards. Older seasons go through FastF1, which only works for sessions already in its cache, because F1's live-timing server now needs an F1 TV subscription.
  - Pick the source with `--source openf1|fastf1`.
  - Suzuka's figure-eight isn't supported yet.
  - Circuits built from laps (`--from-laps`, automatic without TUM data) are approximate: the
    edges are where cars drove plus a margin, and the road is widened to a typical width.

## Development

```bash
ruff check
```

```bash
pytest
```

## Layout

```
src/f1sim/             the package (see docs/revamp-plan.md, section 3)
f1_anatomy/            interactive figures of how an F1 car works (static site)
tests/                 pytest suite
docs/revamp-plan.md    problems, target design, milestones
docs/data-formats.md   file-format spec for tracks, ghosts, checkpoints, runs
```

## Credits

- Timing and telemetry data: [FastF1](https://github.com/theOehrly/Fast-F1). Formula 1 data is
  the property of Formula One World Championship Limited; this is an unofficial hobby project.
- Track centre lines and widths: the
  [TUM racetrack-database](https://github.com/TUMFTM/racetrack-database) (LGPL-3.0),
  downloaded at build time and not redistributed.
- Laps from 2023 onwards: [OpenF1](https://openf1.org) (CC BY-NC-SA 4.0).
