"""Bake a circuit's loader asset from open map data (build time only; not used by ``f1sim``).

    python scripts/geo/fetch.py silverstone     # first: map.osm + terrain into assets_src/geo/
    python scripts/geo/bake.py silverstone      # -> src/f1sim/app/assets/geo/silverstone.npz

Track files are in local metres with no geographic anchor, so the OSM ``highway=raceway``
ways are projected to local metres (:class:`geodata.LocalFrame`) and the track's centre line
is fitted onto them with a similarity transform (rotation, uniform scale, translation):
a coarse search over rotations, then trimmed ICP with the closed-form Umeyama step on
arc-length-resampled points. The fit's RMS error (centre line to nearest OSM point) is
printed and stored. The terrain is then sampled along the centre line and on a coarse grid.

The asset (``f1sim.geo`` v1, documented in ``docs/data-formats.md``) is a few tens of kB:
the centre line, the OSM raceway ways (all of them, in track metres), the elevation every
metre of centre line, a coarse terrain grid and the transform. The loader animation
(``scripts/geo/loader_demo.py``) draws from it without the track being built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import geodata  # noqa: E402

SRC_ROOT = ROOT / "assets_src" / "geo"
OUT_DIR = ROOT / "src" / "f1sim" / "app" / "assets" / "geo"
FIT_STEP_M = 4.0
CLOUD_STEP_M = 1.0
ELEV_SMOOTH_M = 15.0
GRID_CELLS = 48  # along the longer side
GRID_MARGIN_M = 250.0
OSM_STEP_M = 10.0  # the ghost ways
LOOP_STEP_M = 6.0
ATTRIBUTION = (
    "Raceway outline (c) OpenStreetMap contributors, ODbL 1.0. "
    "Elevation: AWS Terrain Tiles (Tilezen joerd; see its attribution.md)."
)


def resample(xy: np.ndarray, step: float, closed: bool) -> np.ndarray:
    """Points every ``step`` metres of arc length along a polyline."""
    if closed:
        xy = np.vstack([xy, xy[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    n = max(int(d[-1] // step), 1)
    s = np.linspace(0.0, d[-1], n, endpoint=not closed) if closed else np.linspace(0, d[-1], n + 1)
    return np.stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])], axis=1)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R src + t - dst|^2, without reflection."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = float(np.trace(np.diag(sig) @ d) / (a**2).sum(1).mean())
    return s, r, md - s * r @ ms


def _icp(track, cloud, tree, r, t, iters, trim, similarity):
    s = 1.0
    for _ in range(iters):
        dist, idx = tree.query(s * track @ r.T + t)
        keep = dist <= np.quantile(dist, trim)
        s2, r, t = umeyama(track[keep], cloud[idx[keep]])
        if similarity:
            s = s2
        else:  # rigid: refit the translation with the scale held at 1
            t = (cloud[idx[keep]] - track[keep] @ r.T).mean(0)
    dist, _ = tree.query(s * track @ r.T + t)
    return s, r, t, float(np.sqrt(np.mean(np.minimum(dist, 30.0) ** 2)))


def fit_similarity(track: np.ndarray, cloud: np.ndarray) -> dict:
    """Fit the track centre line onto the OSM cloud.

    Both are in metres, so the search is rigid: every 10 degrees of rotation, with the
    track's centre started on a 150 m grid over the OSM ways, a short trimmed ICP (the worst
    25 % of matches ignored, so karting tracks, pit lanes and other layouts don't pull it).
    The best start is then refined with the scale free, which checks the metres agree."""
    tree = cKDTree(cloud)
    coarse = track[::3]
    c_track = coarse.mean(0)
    lo, hi = cloud.min(0), cloud.max(0)
    starts = np.stack(
        np.meshgrid(np.arange(lo[0], hi[0], 150.0), np.arange(lo[1], hi[1], 150.0)), -1
    ).reshape(-1, 2)
    best = None
    for deg in range(0, 360, 10):
        a = np.radians(deg)
        r0 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for c in starts:
            _, r, t, score = _icp(coarse, cloud, tree, r0, c - r0 @ c_track, 8, 0.75, False)
            if best is None or score < best[0]:
                best = (score, r, t)
    _, r, t = best
    _, r, t, _ = _icp(track, cloud, tree, r, t, 40, 0.9, False)
    s, r, t, _ = _icp(track, cloud, tree, r, t, 40, 0.95, True)
    dist, _ = tree.query(s * track @ r.T + t)
    return {
        "scale": s,
        "rotation": r,
        "translation": t,
        "rms_m": float(np.sqrt(np.mean(dist**2))),
        "p95_m": float(np.quantile(dist, 0.95)),
        "max_m": float(dist.max()),
    }


def osm_loop(
    ways: list[dict], ways_local: list[np.ndarray], target_m: float, max_steps: int = 500_000
) -> np.ndarray | None:
    """The circuit from OSM alone: split the raceway ways where they meet, then search the
    simple cycles of that graph (depth first, pruned at 1.15x the target) for the one whose
    length is nearest ``target_m``. Needs no built track, so it also serves circuits that
    haven't been built (pass their length). None if no cycle is within 15 %."""
    uses: dict[str, int] = {}
    for w in ways:
        for n in w["nodes"]:
            uses[n] = uses.get(n, 0) + 1
        for n in (w["nodes"][0], w["nodes"][-1]):
            uses[n] = uses.get(n, 0) + 1  # ends are junctions too
    edges = []  # (a, b, points)
    for w, xy in zip(ways, ways_local, strict=True):
        cut = [i for i, n in enumerate(w["nodes"]) if uses[n] > 1 or i in (0, len(xy) - 1)]
        for i, j in zip(cut[:-1], cut[1:], strict=True):
            edges.append((w["nodes"][i], w["nodes"][j], xy[i : j + 1]))
    length = [float(np.hypot(*np.diff(p, axis=0).T).sum()) for _, _, p in edges]
    adj: dict[str, list[int]] = {}
    for k, (a, b, _) in enumerate(edges):
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)

    best: tuple[float, list[tuple[int, bool]]] | None = None
    steps = 0
    for start in sorted(adj):
        stack = [(start, [], {start}, 0.0)]
        while stack and steps < max_steps:
            node, path, seen, total = stack.pop()
            steps += 1
            for k in adj[node]:
                if path and k == path[-1][0]:
                    continue
                a, b, _ = edges[k]
                nxt = b if a == node else a
                tot = total + length[k]
                if tot > 1.15 * target_m:
                    continue
                step = [*path, (k, a == node)]
                if nxt == start:
                    err = abs(tot - target_m)
                    if best is None or err < best[0]:
                        best = (err, step)
                elif nxt not in seen and nxt > start:  # each cycle once: from its lowest node
                    stack.append((nxt, step, seen | {nxt}, tot))
    if best is None or best[0] > 0.15 * target_m:
        return None
    pts = [edges[k][2] if forward else edges[k][2][::-1] for k, forward in best[1]]
    return np.vstack([p[:-1] for p in pts])


def bake(track_id: str, track_file: Path | None = None) -> Path:
    from f1sim.data.io import read_track_json
    from f1sim.paths import track_file as default_track_file

    src = SRC_ROOT / track_id
    td = read_track_json(track_file or default_track_file(track_id))
    bbox = json.loads((src / "bbox.json").read_text())["bbox_swne"]
    frame = geodata.LocalFrame((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    ways = geodata.raceway_ways(src / "map.osm")
    if not ways:
        raise SystemExit(f"No highway=raceway ways in {src / 'map.osm'}")
    ways_local = [frame.from_latlon(w["lat"], w["lon"]) for w in ways]
    cloud = np.vstack([resample(w, CLOUD_STEP_M, closed=False) for w in ways_local])

    centre = np.stack([td.x, td.y], axis=1)
    fit = fit_similarity(resample(centre, FIT_STEP_M, closed=True), cloud)
    s, r, t = fit["scale"], fit["rotation"], fit["translation"]
    angle = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    print(
        f"{track_id}: {len(ways)} raceway ways; fit scale {s:.4f}, rotation {angle:.2f} deg, "
        f"RMS {fit['rms_m']:.2f} m, p95 {fit['p95_m']:.2f} m, max {fit['max_m']:.2f} m"
    )

    def to_local(xy):
        return s * xy @ r.T + t

    def to_track(xy):
        return (xy - t) @ r / s

    # the circuit's loop from OSM alone, turned to start at the track's start line and run
    # its way round; the centre line's distance to it checks the fit without the distractors
    loop = osm_loop(ways, ways_local, float(td.length_m))
    if loop is None:
        raise SystemExit("No OSM raceway cycle within 15 % of the track's length")
    loop = resample(to_track(loop), LOOP_STEP_M, closed=True)
    nxt = np.roll(loop, -1, axis=0)
    area = float((loop[:, 0] * nxt[:, 1] - nxt[:, 0] * loop[:, 1]).sum())  # > 0: anticlockwise
    if (area < 0) != (td.direction == "clockwise"):
        loop = loop[::-1]
    loop = np.roll(loop, -int(np.argmin(np.hypot(*(loop - centre[0]).T))), axis=0)
    d_loop, _ = cKDTree(resample(loop, CLOUD_STEP_M, closed=True)).query(
        resample(centre, FIT_STEP_M, closed=True)
    )
    loop_len = float(np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T).sum())
    fit["loop_rms_m"] = float(np.sqrt(np.mean(d_loop**2)))
    print(
        f"OSM loop {loop_len:.0f} m (track {td.length_m:.0f} m); centre line to it: "
        f"RMS {fit['loop_rms_m']:.2f} m, max {d_loop.max():.2f} m"
    )

    terrain = geodata.Terrain(sorted((src / "terrain").glob("*.tif")))

    def height(xy_track):
        return terrain.sample(*frame.to_mercator(to_local(xy_track)))

    # elevation every metre of centre line, smoothed (terrain pixels are ~6 m, noisy at 1 m)
    length = float(td.length_m)
    s_m = np.arange(0.0, length, 1.0)
    sx = np.r_[td.s, length]
    line = np.stack(
        [np.interp(s_m, sx, np.r_[td.x, td.x[0]]), np.interp(s_m, sx, np.r_[td.y, td.y[0]])],
        axis=1,
    )
    elev = gaussian_filter1d(height(line), ELEV_SMOOTH_M, mode="wrap")
    base = float(elev.min())

    # coarse grid around the circuit (track frame, axis-aligned)
    lo, hi = centre.min(0) - GRID_MARGIN_M, centre.max(0) + GRID_MARGIN_M
    step = float(np.ceil((hi - lo).max() / GRID_CELLS))
    nx, ny = (np.ceil((hi - lo) / step).astype(int) + 1).tolist()
    gx, gy = np.meshgrid(lo[0] + step * np.arange(nx), lo[1] + step * np.arange(ny))
    grid = height(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(ny, nx)
    grid = gaussian_filter(grid, 0.6, mode="nearest")

    # OSM raceway ways in track metres, NaN-separated, every ~OSM_STEP_M, cut to the grid
    # (karting tracks and other circuits further out would only cost bytes)
    parts = []
    for w in ways_local:
        p = to_track(resample(w, OSM_STEP_M, closed=False))
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        for run in np.split(p, np.flatnonzero(np.diff(inside.astype(int))) + 1):
            if len(run) >= 2 and np.all((run >= lo) & (run <= hi)):
                parts += [run, np.full((1, 2), np.nan)]
    osm = np.vstack(parts[:-1])

    matrix = np.eye(3)
    matrix[:2, :2], matrix[:2, 2] = s * r, t
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{track_id}.npz"
    np.savez_compressed(
        out,
        format=np.array("f1sim.geo"),
        version=np.array(1),
        track_id=np.array(track_id),
        attribution=np.array(ATTRIBUTION),
        anchor_latlon=np.array([frame.lat0, frame.lon0]),
        track_to_local=matrix,
        fit=np.array([fit["rms_m"], fit["p95_m"], fit["max_m"], s, angle, fit["loop_rms_m"]]),
        centre_xy=resample(centre, FIT_STEP_M, closed=True).astype(np.float32),
        length_m=np.array(length),
        elev_base_m=np.array(base),
        elev_rel_m=(elev - base).astype(np.float16),
        grid_origin=lo.astype(np.float64),
        grid_step_m=np.array(step),
        grid_rel_m=(grid - base).astype(np.float16),
        osm_xy=osm.astype(np.float32),
        loop_xy=loop.astype(np.float32),
    )
    print(
        f"elevation {base:.1f} to {elev.max():.1f} m (range {elev.max() - base:.1f} m); "
        f"grid {nx}x{ny} at {step:.0f} m, {grid.min():.1f} to {grid.max():.1f} m"
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} kB)")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("track_id")
    ap.add_argument("--track-file", type=Path, help="default: the built track in the cache")
    args = ap.parse_args(argv)
    bake(args.track_id, args.track_file)


if __name__ == "__main__":
    main()
