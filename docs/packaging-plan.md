# Packaging plan: a Windows and macOS app, with no terminal

Status: step 1 of section 4 is in progress (2026-10-29). Done: the track picker lists every
circuit, "Get track data" and "Get real lap" are buttons (background jobs with progress, cancel
and retry), and Train has a time control. Still open: the frozen-app worker (item 4), the
runs/ data folder (item 5), the numba cache (item 7), and everything from section 3 on.

**Goal.** A person downloads one file, opens it, and can do everything from the window:
pick a track, get its data, drive, train, replay. No Python, no terminal.

## 1. What blocks a download-and-run app today

| # | Problem | Where | Fix |
|---|---|---|---|
| 1 | Needs Python and `pip install` | README | Package with PyInstaller (section 3) |
| 2 | Tracks are built only by `f1sim build-track` | `cli.py` | A "Tracks" screen in the app (section 2) |
| 3 | Real-driver ghost laps are made only by `f1sim ghost` | `cli.py` | Same screen: a "Get real lap" button |
| 4 | Training starts a child process with `sys.executable -m f1sim.cli` | `app/training.py` `train_command`, `app/menu.py` `command` | In a packaged app `sys.executable` is the app itself. Run the same app with a hidden `--worker train ...` argument, and call `multiprocessing.freeze_support()` first |
| 5 | Training options (minutes, optimiser, seed) are flags only | `cli.py` | A small "Train" settings panel with safe defaults |
| 6 | Data needs the internet at first use (TUM, OpenF1, FastF1) | `data/*` | Show progress and clear errors in the window; work offline once built |
| 7 | numba compiles on first run and writes a cache next to the code | `_accel.py` | Point the cache at the user cache folder; show "Preparing (first start only)" |

CLI-only commands left after this: `plot-track`, `reference-lap`, `watch`, `bench`.
They are developer tools. They stay in the CLI and need no UI.

## 2. Everything a user can do, as UI

| Action | Today | After |
|---|---|---|
| Choose a track | menu picker (built tracks only) | picker lists all circuits. Unbuilt ones show "Get data" |
| Build a track | `f1sim build-track` | "Get data" button, with progress and a clear error if offline |
| Get a real lap to race | `f1sim ghost` | "Get real lap" (year picker) on the Drive card |
| Drive | UI | UI |
| Train | UI (fixed settings) | UI, plus Start / Stop / Continue and a time limit |
| Replay | UI | UI |
| Manage disk space (the cache is 442 MB here) | manual | "Storage" line with a Clear button |
| Update | `run_latest.bat` / git | the app checks for a new version (section 5) |

Rules for the new screens: same style as the menu cards; every long task shows
progress, can be cancelled, and shows a plain-English error with a Retry button.

## 3. Packaging

- **Tool:** PyInstaller, one-folder build, then wrapped:
  Windows: an installer (Inno Setup) or a zip. macOS: a `.app` inside a `.dmg`.
- **One build per system.** PyInstaller cannot cross-build. Use GitHub Actions with a
  Windows runner and a macOS runner (also build Apple Silicon and Intel, or a universal2 build).
- **Include as data files:** `src/f1sim/app/assets` (5 MB: fonts, hero images, stage),
  `configs/*.toml`.
- **Likely hidden imports:** arcade/pyglet backends, scipy and shapely native libraries,
  numba and llvmlite. Expect a few build-and-test rounds.
- **Size:** estimate 250-400 MB with numba. If too big, ship without numba (the code already
  falls back to numpy, 17.6x instead of 64x real time) and compare training speed first.
- **OpenGL:** all 21 shaders use `#version 330`. macOS supports up to 4.1 core, so shaders
  are fine. Still test on a real Mac: pyglet needs a core profile there.
- **Signing:** Windows shows a SmartScreen warning for unsigned apps (a code-signing
  certificate costs money). macOS blocks unsigned apps; to open without a warning it needs an
  Apple Developer account (99 USD/year), signing and notarisation. Without them, users must
  right-click -> Open. This is a decision for you (section 6).
- **Data location:** already outside the app: `platformdirs` user cache for tracks, ghosts
  and FastF1 data. Runs go to `runs/` in the current folder; a packaged app must use a
  user data folder instead (`platformdirs.user_data_dir`).

## 4. Order of work

1. **Make the app work without a terminal, still run from source.** Items 2-5 of section 1,
   the Tracks / Get real lap / Train panels. This is the largest part and is testable on
   this machine.
2. **Frozen-app support.** The worker argument, cache and runs folders, numba cache path.
3. **Build on Windows** with PyInstaller. Test on a clean Windows machine (no Python).
4. **CI for both systems, then test on a Mac.** I cannot run a Mac build from this machine.
5. **Signing, installer, update check.**

## 5. Updates

`run_latest.bat` uses `git pull`, which a normal user does not have. Replace it with a
version check against GitHub Releases (one small request at start) and a "New version
available" line in the menu that opens the download page.

## 6. Decisions needed from you

1. **Data licences.** OpenF1 data is CC BY-NC-SA (non-commercial) and TUM tracks are
   LGPL-3.0. Today nothing is redistributed: the app downloads at first use. Shipping
   pre-built tracks would mean redistributing derived data. Safest: keep downloading
   in the app (needs internet once per track). Is the app free and non-commercial?
2. **Signing.** Pay for signing and notarisation, or accept the warnings for now?
3. **Mac testing.** Do you have a Mac to test on, or should CI plus a tester be enough?
4. **Size.** Keep numba (faster training, bigger download) or drop it?
