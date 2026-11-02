"""Build every circuit and its newest real qualifying lap, and copy them into the package.

The app ships these files (``src/f1sim/data/bundled``, D52), so a fresh install can drive,
train and race a ghost without pressing "Get track data" or "Get real lap". Run it again to
refresh the data (a new season's poles, a pipeline change); it needs the internet.

    python scripts/bundle_data.py                  # every circuit
    python scripts/bundle_data.py monza spa        # only these
    python scripts/bundle_data.py --cache <dir>    # build in this cache (default: a fresh one)

Builds go to their own cache folder, not your user cache, so the bundle comes only from the
current pipeline. Network caches (TUM, OpenF1, FastF1) are copied over from your user cache
first to save downloads. Tracks and laps already in that folder are reused, so a rerun with
the same ``--cache`` only retries what failed; to rebuild everything, use a new folder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src" / "f1sim" / "data" / "bundled"

# TUM layouts that are out of date, built from real laps instead (D40). Circuits without TUM
# data are built from laps automatically.
FROM_LAPS = {"melbourne", "catalunya", "zandvoort", "yas_marina"}
# Suzuka's figure-eight isn't supported yet (PROJECT_LOG open problem 1).
SKIP = {"suzuka"}
NETWORK_CACHES = ("tum", "openf1", "fastf1")
# When a pole lap's position data is too gappy for a ghost (ghost_builder's 8 g and track-edge
# checks), try this many of the next-fastest drivers in the same session.
OTHER_DRIVERS = 10


def real_lap(tid: str) -> str:
    """The app's "Get real lap", then the next-fastest drivers' laps if every pole fails."""
    from f1sim.app import jobs
    from f1sim.data import ghost_builder, openf1_source

    try:
        return jobs.get_real_lap(tid)
    except LookupError:
        pass
    for year in jobs.real_lap_years():
        try:
            ranked = openf1_source.ranked_drivers(openf1_source.find_session(year, tid))
        except openf1_source.OpenF1Error:
            continue
        for number in ranked[1 : 1 + OTHER_DRIVERS]:
            try:
                ghost = ghost_builder.build_ghost(tid, year, driver=number)
            except (ValueError, openf1_source.OpenF1Error):
                continue
            driver = ghost.meta["driver"]
            ghost_builder.save_ghost(ghost, ghost_builder.ghost_path(year, tid, "Q", driver))
            return f"Got {driver}'s {year} qualifying lap (not the pole)."
    raise LookupError(f"no usable qualifying lap for {tid}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="*", help="track ids (default: every circuit)")
    ap.add_argument("--cache", help="cache folder to build in (default: a new temporary one)")
    args = ap.parse_args()

    import platformdirs

    user_cache = Path(platformdirs.user_cache_dir("f1sim", appauthor=False))
    cache = Path(args.cache) if args.cache else Path(tempfile.mkdtemp(prefix="f1sim-bundle-"))
    for name in NETWORK_CACHES:
        src, dst = user_cache / name, cache / name
        if src.is_dir() and not dst.exists():
            print(f"copying {src} -> {dst}", flush=True)
            shutil.copytree(src, dst)
    os.environ["F1SIM_CACHE_DIR"] = str(cache)

    from f1sim import paths
    from f1sim.data import catalog
    from f1sim.data.track_pipeline import build_and_cache

    ids = args.tracks or sorted(set(catalog.CATALOG) - SKIP)
    failed: list[str] = []
    (BUNDLE / "tracks").mkdir(parents=True, exist_ok=True)
    (BUNDLE / "ghosts").mkdir(parents=True, exist_ok=True)
    for tid in ids:
        print(f"== {tid}", flush=True)
        try:
            if not paths.track_file(tid).exists():
                if catalog.CATALOG[tid].tum_name is not None and tid not in FROM_LAPS:
                    build_and_cache(tid, limit_years=(2023, 2024, 2025))
                else:
                    build_and_cache(tid, from_laps=True)
            print("  track:", paths.track_file(tid), flush=True)
        except Exception as exc:  # report and go on with the rest
            print(f"  FAILED track: {type(exc).__name__}: {exc}", flush=True)
            failed.append(tid)
            continue
        shutil.copy2(paths.track_file(tid), BUNDLE / "tracks")
        try:
            if not any(paths.ghosts_dir().glob(f"*_{tid}_*.npz")):
                print("  lap:  ", real_lap(tid), flush=True)
        except Exception as exc:  # the track still ships, without a real lap
            print(f"  FAILED lap: {type(exc).__name__}: {exc}", flush=True)
            failed.append(tid)
            continue
        for old in (BUNDLE / "ghosts").glob(f"*_{tid}_*"):
            old.unlink()
        for f in paths.ghosts_dir().glob(f"*_{tid}_*"):
            shutil.copy2(f, BUNDLE / "ghosts")

    print(f"cache: {cache}")
    if failed:
        print("failed:", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
