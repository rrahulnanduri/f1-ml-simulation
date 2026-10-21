"""`f1sim` command line: list, build, plot and drive tracks. Train/replay arrive in M3-M4."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

from f1sim import __version__


def _cmd_tracks(args: argparse.Namespace) -> int:
    from f1sim.data import catalog
    from f1sim.data.track_pipeline import track_path

    print(f"{'id':<16} {'circuit':<36} {'geometry':<10} built")
    for entry in sorted(catalog.CATALOG.values(), key=lambda e: e.id):
        geometry = "TUM" if entry.tum_name else "fallback"
        built = "yes" if track_path(entry.id).exists() else ""
        print(f"{entry.id:<16} {entry.name:<36} {geometry:<10} {built}")
    return 0


def _cmd_build_track(args: argparse.Namespace) -> int:
    from f1sim.data.track_builder import TrackBuildError, wall_self_crossings
    from f1sim.data.track_pipeline import AlignmentRejected, build_and_cache

    year = None if args.no_fastf1 else args.year
    try:
        report = build_and_cache(
            args.track_id,
            year=year,
            resolution_m=args.resolution,
            source=args.source,
            limit_years=tuple(args.limit_years) if args.limits else None,
        )
    except (
        KeyError,
        ValueError,
        NotImplementedError,
        AlignmentRejected,
        TrackBuildError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    t = report.track
    total_width = t.w_left + t.w_right
    left_x, right_x = wall_self_crossings(t)
    print(f"{t.name} ({t.id}) -> {report.path}")
    print(f"  length       {t.length_m:.0f} m, {len(t.x)} points every {t.resolution_m} m")
    print(f"  width        {total_width.min():.1f}-{total_width.max():.1f} m wall to wall")
    print(f"  walls        {left_x} left / {right_x} right self-crossings")
    print(f"  direction    {t.direction}")
    if report.alignment is not None:
        a = report.alignment
        print(
            f"  reference    {report.reference_driver} {report.reference_lap_s:.3f} s "
            f"({report.lap_source})"
        )
        print(
            f"  alignment    median {a.median_dist_m:.2f} m from centre line, "
            f"{t.fastf1_alignment['inside_ratio_tol']:.1%} within 1 m of edges, "
            f"{a.inside_ratio:.1%} inside edges, rotation {a.rotation_rad:.4f} rad, "
            f"mirrored={a.mirrored}"
        )
        print(f"  sectors at   {', '.join(f'{s:.0f}' for s in t.sectors_s)} m")
    if t.has_limits:
        from f1sim.data.track_limits import limit_self_crossings

        el, er = t.limit_left - t.w_left, t.limit_right - t.w_right
        print(
            f"  limits       {t.limits['laps']} laps from {t.limits['years']}: beyond the white "
            f"line by median {np.median(el):.2f}/{np.median(er):.2f} m, p90 "
            f"{np.percentile(el, 90):.2f}/{np.percentile(er, 90):.2f} m, max "
            f"{max(el.max(), er.max()):.2f} m (left/right); widened on "
            f"{np.mean((el > 0.1) | (er > 0.1)):.0%} of the lap"
        )
        lx, rx = limit_self_crossings(t)
        print(f"  limit lines  {lx} left / {rx} right self-crossings")
    return 0


def _cmd_plot_track(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from f1sim.data.track_pipeline import load_track

    try:
        track = load_track(args.track_id)
    except FileNotFoundError:
        print(
            f"error: {args.track_id} isn't built; run `f1sim build-track {args.track_id}`",
            file=sys.stderr,
        )
        return 1
    t = track.data
    left, right = track.left_wall(), track.right_wall()
    fig, ax = plt.subplots(figsize=(10, 8))
    # Road surface between the walls, then the walls themselves
    for wall in (left, right):
        closed = np.vstack([wall, wall[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", lw=0.8)
    ax.fill(left[:, 0], left[:, 1], color="0.85", zorder=0)
    ax.fill(right[:, 0], right[:, 1], color="white", zorder=0)
    ax.plot(t.x, t.y, color="0.5", lw=0.5, ls="--", label="centre line")
    for i, s in enumerate(t.sectors_s or (0.0,)):
        p = track.point_at(s)
        wl, wr = (float(w) for w in track.widths_at(s))
        h = float(track.heading_at(s))
        normal = np.array([-np.sin(h), np.cos(h)])
        a, b = p + normal * wl, p - normal * wr
        color = "red" if i == 0 else "tab:blue"
        label = "start/finish" if i == 0 else ("sector boundary" if i == 1 else None)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2.5, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    width = t.w_left + t.w_right
    ax.set_title(
        f"{t.name}: {t.length_m:.0f} m, {width.min():.1f}-{width.max():.1f} m wide ({t.direction})"
    )
    ax.legend(loc="best")
    out = Path(args.out or f"{t.id}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(out.resolve())
    return 0


def _cmd_drive(args: argparse.Namespace) -> int:
    from f1sim.app.drive import run

    run(args.track_id, ghost=Path(args.ghost) if args.ghost else None)
    return 0


def _cmd_reference_lap(args: argparse.Namespace) -> int:
    from f1sim.data import catalog, tum_source
    from f1sim.data.track_pipeline import load_track
    from f1sim.sim.car_model import CarParams, load_car_params
    from f1sim.sim.scripted import drive_reference_lap

    try:
        track = load_track(args.track_id)
    except FileNotFoundError:
        print(
            f"error: {args.track_id} isn't built; run `f1sim build-track {args.track_id}`",
            file=sys.stderr,
        )
        return 1
    params = load_car_params(args.car) if args.car else CarParams()
    raceline = None
    line = "centre line"
    if not args.centre_line:
        tum_name = catalog.get(args.track_id).tum_name
        if tum_name is None:
            print(f"error: no TUM raceline for {args.track_id}", file=sys.stderr)
            return 1
        raceline = tum_source.parse_tum_raceline(tum_source.fetch_tum_raceline(tum_name))
        line = "TUM raceline"
    if args.white_line:
        edges = "white line"
    elif track.has_limits:
        edges = "track limits"
        track = track.sim_track()
    else:
        edges = "white line (no limits built; see build-track --limits)"
    lap = drive_reference_lap(track, params, raceline_xy=raceline)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, t=lap.t, x=lap.x, y=lap.y, s=lap.s, speed=lap.v)
        print(f"  saved trace  {out.resolve()}")
    print(f"{track.data.name}, {line}, {edges}, car {args.car or 'defaults'}")
    if not lap.completed or not np.isfinite(lap.lap_time_s):
        print(f"  lap not completed cleanly (worst excess {lap.max_abs_n_excess_m:+.2f} m)")
        return 1
    print(f"  lap time     {lap.lap_time_s:.3f} s")
    print(f"  top speed    {lap.v.max() * 3.6:.1f} km/h")
    print(f"  on track     yes (closest to the limit: {lap.max_abs_n_excess_m:+.2f} m)")
    ref = (track.data.fastf1_alignment or {}).get("reference_lap_s")
    if ref:
        driver = track.data.fastf1_alignment.get("reference_driver", "")
        delta = lap.lap_time_s / ref - 1
        print(f"  real pole    {driver} {ref:.3f} s -> {delta:+.1%} (M2 target: within 10%)")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from f1sim.app.replay import run

    traces = [Path(p) for p in args.traces]
    # A run folder means its best lap
    traces = [p / "best_lap.npz" if p.is_dir() else p for p in traces]
    run(args.track_id, traces, args.labels)
    return 0


def _cmd_ghost(args: argparse.Namespace) -> int:
    from f1sim.data.ghost_builder import build_ghost, ghost_path, save_ghost

    try:
        g = build_ghost(
            args.track_id, args.year, driver=args.driver, kind=args.session, source=args.source
        )
    except (KeyError, ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    m = g.meta
    out = (
        Path(args.out)
        if args.out
        else ghost_path(args.year, args.track_id, args.session, m["driver"])
    )
    save_ghost(g, out)
    team = f" ({m['team']})" if m.get("team") else ""
    print(
        f"{m['driver']}{team} {args.year} {args.track_id} {args.session}: {m['lap_time_s']:.3f} s"
    )
    print(f"  source       {m['lap_source']}, {g.t.size} samples at {m['rate_hz']:g} Hz")
    print(f"  top speed    {g.speed.max() * 3.6:.1f} km/h")
    print(
        f"  smoothness   max {m['max_accel_g']:.1f} g along track "
        f"(99th pct {m['p99_accel_g']:.1f} g)"
    )
    print(f"  on track     {m['near_edge_ratio']:.1%} within 1 m of the edges")
    print(f"  speed check  {m['speed_check_kmh']:.1f} km/h median vs the car's speed sensor")
    print(f"  saved        {out.resolve()}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import dataclasses
    import time

    from f1sim.data.track_pipeline import load_track
    from f1sim.train.trainer import TrainConfig, Trainer, load_train_config

    cfg = load_train_config(Path(args.config)) if args.config else TrainConfig()
    overrides = {
        "track_id": args.track,
        "seed": args.seed,
        "optimiser": args.optimiser,
        "init": args.init,
        "max_generations": args.generations,
        "max_wall_s": None if args.minutes is None else args.minutes * 60.0,
    }
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    try:
        track = load_track(cfg.track_id)
    except FileNotFoundError:
        print(
            f"error: {cfg.track_id} isn't built; run `f1sim build-track {cfg.track_id}`",
            file=sys.stderr,
        )
        return 1
    run_dir = Path(args.run_dir or f"runs/{cfg.track_id}-{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"training on {track.data.name} -> {run_dir.resolve()}")
    if cfg.init == "clone":
        print(f"cloning the scripted driver ({cfg.clone_rounds} DAgger rounds) ...", flush=True)
        clone_log = logging.getLogger("f1sim.train.trainer")
        clone_log.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("  %(message)s"))
        clone_log.addHandler(handler)

    def report(row: dict) -> None:
        lap = row.get("best_lap_s")
        lap_ok = lap not in (None, "") and math.isfinite(float(lap))
        lap_txt = f"{float(lap):7.3f} s" if lap_ok else "      -  "
        print(
            f"gen {int(row['gen']):4d}  stage {row['stage']}  {float(row['wall_s']):7.0f} s  "
            f"best {float(row['best_fitness']):9.1f}  "
            f"progress {float(row['best_progress_m']):6.0f} m  "
            f"laps {float(row['completion_rate']):5.1%}  best lap {lap_txt}  "
            f"sigma {float(row['sigma']):.3f}",
            flush=True,
        )

    result = Trainer(track, cfg, run_dir).run(on_generation=report)
    first = result.first_lap_wall_s
    print(f"done: {result.generations} generations, stage {result.stage}")
    print(
        f"  best lap     {result.best_lap_s:.3f} s" if result.best_lap_s else "  best lap     none"
    )
    print(f"  first lap    after {first:.0f} s of training" if first else "  first lap    none")
    print(f"  replay with  f1sim replay {cfg.track_id} {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1sim", description=__doc__)
    parser.add_argument("--version", action="version", version=f"f1sim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tracks", help="list circuits and whether they're built")
    p.set_defaults(func=_cmd_tracks)

    p = sub.add_parser("build-track", help="build a circuit from TUM data into the cache")
    p.add_argument("track_id")
    p.add_argument("--year", type=int, default=2024, help="qualifying session used for alignment")
    p.add_argument(
        "--no-fastf1", action="store_true", help="skip lap alignment and sectors (geometry only)"
    )
    p.add_argument(
        "--source",
        choices=("auto", "openf1", "fastf1"),
        default="auto",
        help="reference-lap source; auto = OpenF1 for 2023+, FastF1 (cached only) before",
    )
    p.add_argument("--resolution", type=float, default=2.0, help="point spacing in metres")
    p.add_argument(
        "--limits",
        action="store_true",
        help="add track limits from real drivers' qualifying laps (OpenF1)",
    )
    p.add_argument(
        "--limit-years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="seasons whose qualifying laps define the limits",
    )
    p.set_defaults(func=_cmd_build_track)

    p = sub.add_parser("plot-track", help="draw a built track to a PNG")
    p.add_argument("track_id")
    p.add_argument("--out", help="output path (default: <track_id>.png)")
    p.add_argument("--dpi", type=int, default=150)
    p.set_defaults(func=_cmd_plot_track)

    p = sub.add_parser("reference-lap", help="drive the scripted reference lap (calibration)")
    p.add_argument("track_id")
    p.add_argument("--car", help="car parameters TOML (default: built-in CarParams)")
    p.add_argument("--centre-line", action="store_true", help="follow the centre line instead")
    p.add_argument("--save", help="write the lap trace to this .npz (for f1sim replay)")
    p.add_argument(
        "--white-line", action="store_true", help="drive within the white line, not the limits"
    )
    p.set_defaults(func=_cmd_reference_lap)

    p = sub.add_parser("train", help="train cars by neuroevolution (headless)")
    p.add_argument("--config", help="training TOML (default: built-in defaults)")
    p.add_argument("--track", help="track id (default from config: silverstone)")
    p.add_argument("--minutes", type=float, help="wall-clock limit")
    p.add_argument("--generations", type=int, help="generation limit")
    p.add_argument("--seed", type=int)
    p.add_argument("--optimiser", choices=("ga", "cmaes"))
    p.add_argument(
        "--init",
        choices=("clone", "random"),
        help="clone: copy the scripted driver, then evolve (default); random: from scratch",
    )
    p.add_argument("--run-dir", help="output folder (default: runs/<track>-<timestamp>)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("ghost", help="build a smoothed ghost of a real driver's fastest lap")
    p.add_argument("track_id")
    p.add_argument("year", type=int)
    p.add_argument("--driver", help="abbreviation or number (default: the session's fastest)")
    p.add_argument("--session", default="Q", help="Q (default), R, S or SQ")
    p.add_argument(
        "--source",
        choices=("auto", "openf1", "fastf1"),
        default="auto",
        help="auto = OpenF1 for 2023+, FastF1 (cached only) before",
    )
    p.add_argument("--out", help="output .npz (default: the ghosts folder in the user cache)")
    p.set_defaults(func=_cmd_ghost)

    p = sub.add_parser("replay", help="replay recorded laps (npz traces or run folders)")
    p.add_argument("track_id")
    p.add_argument("traces", nargs="+", help=".npz trace files, or run folders (best_lap.npz)")
    p.add_argument("--labels", nargs="+", help="one label per trace")
    p.set_defaults(func=_cmd_replay)

    p = sub.add_parser("drive", help="drive a built track with the keyboard")
    p.add_argument("track_id")
    p.add_argument(
        "--ghost", help="ghost lap .npz to race against (it starts when you cross the line)"
    )
    p.set_defaults(func=_cmd_drive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
