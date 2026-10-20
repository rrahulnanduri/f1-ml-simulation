"""`f1sim` command line: list, build, plot and drive tracks. Train/replay arrive in M3-M4."""

from __future__ import annotations

import argparse
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
            args.track_id, year=year, resolution_m=args.resolution, source=args.source
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

    run(args.track_id)
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
    lap = drive_reference_lap(track, params, raceline_xy=raceline)
    print(f"{track.data.name}, {line}, car {args.car or 'defaults'}")
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
    p.set_defaults(func=_cmd_reference_lap)

    p = sub.add_parser("drive", help="drive a built track with the keyboard")
    p.add_argument("track_id")
    p.set_defaults(func=_cmd_drive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
