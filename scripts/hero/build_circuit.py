"""Build the menu hero's circuit from a real qualifying lap (OpenF1, no sign-in needed).

    python scripts/hero/build_circuit.py [track_id] [year]     (default: marina_bay 2025)

The hero car drives this lap forever (:mod:`f1sim.app.ui.hero_drive`). Singapore isn't in
the TUM data the simulator's tracks come from, so everything comes from the session's
fastest qualifying lap:

- **Racing line:** a periodic cubic spline through the lap's car positions (about 4 Hz),
  evaluated in time and resampled every ``STEP_M``. Speed along it comes from the lap's car
  data (the speed sensor), so the hero car brakes and accelerates as the real one did.
- **Track:** a ``WIDTH_M`` road whose centre line is the racing line shifted towards the
  outside of each corner (the line runs ``INSIDE_GAP_M`` from the inner edge at the apex),
  built with :func:`f1sim.data.track_builder.build_track`.
- **Kerbs:** an apex kerb inside and an exit kerb outside every corner
  (:func:`f1sim.app.render_track.kerb_geometry`; there's no surveyed kerb data here).

Writes ``src/f1sim/app/assets/hero/<track_id>.npz`` (a few hundred kB; commit it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "src" / "f1sim" / "app" / "assets" / "hero"
STEP_M = 1.0
WIDTH_M = 12.0
INSIDE_GAP_M = 1.8
CURVE_SMOOTH_M = 25.0
OFFSET_RADIUS_M = 60.0
"""The line is fully over to the inside (``WIDTH_M / 2 - INSIDE_GAP_M``) in bends tighter than
about this radius (``tanh(radius_scale * curvature)``)."""


def racing_line(t, x, y, lap_time: float, rate_hz: float = 50.0):
    """Closed racing line from lap samples: ``(xs, ys, ts)`` every ``STEP_M``, where ``ts``
    is the lap time at each point."""
    t = np.asarray(t, float)
    keep = np.concatenate([[True], np.diff(t) > 1e-6]) & (t < lap_time - 1e-6)
    t, x, y = t[keep], np.asarray(x, float)[keep], np.asarray(y, float)[keep]
    tt = np.append(t, t[0] + lap_time)
    pts = np.column_stack([np.append(x, x[0]), np.append(y, y[0])])
    spline = CubicSpline(tt, pts, bc_type="periodic")
    td = np.arange(t[0], t[0] + lap_time, 1.0 / rate_hz)
    p = spline(td)
    seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = s[-1]
    sq = np.arange(0.0, length, STEP_M)
    pc = np.vstack([p, p[:1]])
    tc = np.append(td, td[0] + lap_time)
    xs = np.interp(sq, s, pc[:, 0])
    ys = np.interp(sq, s, pc[:, 1])
    ts = np.interp(sq, s, tc) - t[0]
    return xs, ys, ts


def heading_curvature(x, y, smooth_m: float):
    """Heading and signed curvature (left positive) of a closed line, smoothed."""
    sig = smooth_m / STEP_M
    xs = gaussian_filter1d(x, sig, mode="wrap")
    ys = gaussian_filter1d(y, sig, mode="wrap")
    dx = (np.roll(xs, -1) - np.roll(xs, 1)) / (2 * STEP_M)
    dy = (np.roll(ys, -1) - np.roll(ys, 1)) / (2 * STEP_M)
    ddx = (np.roll(xs, -1) - 2 * xs + np.roll(xs, 1)) / STEP_M**2
    ddy = (np.roll(ys, -1) - 2 * ys + np.roll(ys, 1)) / STEP_M**2
    heading = np.arctan2(dy, dx)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    return heading, kappa


def main(track_id: str = "marina_bay", year: int = 2025) -> int:
    from f1sim.app import render_track as rt
    from f1sim.data import openf1_source
    from f1sim.data.catalog import get
    from f1sim.data.track_builder import build_track
    from f1sim.track.track import Track

    session = openf1_source.find_session(year, track_id, "Q")
    lap = openf1_source.fastest_lap_xy(session)
    car = openf1_source.fastest_lap_car_data(session)
    xs, ys, ts = racing_line(lap.t, lap.x_m, lap.y_m, lap.lap_time_s)
    xs, ys = xs - xs.mean(), ys - ys.mean()
    # Speed along the line from the speed sensor, by lap time
    ct, cv = np.asarray(car.t, float), np.asarray(car.speed_ms, float)
    order = np.argsort(ct)
    speed = np.interp(ts, ct[order], cv[order])
    speed = gaussian_filter1d(speed, 3.0, mode="wrap")
    heading, kappa = heading_curvature(xs, ys, 4.0)
    # Centre line: the racing line moved to the outside of each bend
    _, kappa_s = heading_curvature(xs, ys, CURVE_SMOOTH_M)
    offset = (WIDTH_M / 2 - INSIDE_GAP_M) * np.tanh(OFFSET_RADIUS_M * kappa_s)
    nx, ny = -np.sin(heading), np.cos(heading)  # left normal
    cx, cy = xs - nx * offset, ys - ny * offset
    half = np.full(cx.size, WIDTH_M / 2)
    data = build_track(
        cx[::2],
        cy[::2],
        half[::2],
        half[::2],
        track_id=f"{track_id}_hero",
        name=get(track_id).name,
        resolution_m=2.0,
        max_shift_m=1.0,
    )
    track = Track(data)
    tris, red = [], []
    for side in ("left", "right"):
        g = rt.kerb_geometry(track, side)
        tris.append(g.kerb)
        red.append(g.red)
    kerb = np.concatenate(tris)
    kerb_red = np.concatenate(red)
    length = xs.size * STEP_M
    lap_from_speed = float(np.sum(STEP_M / np.maximum(speed, 1.0)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{track_id}.npz"
    np.savez_compressed(
        out,
        line=np.column_stack([xs, ys]).astype(np.float32),
        speed=speed.astype(np.float32),
        heading=heading.astype(np.float32),
        curvature=kappa.astype(np.float32),
        left=track.left_wall().astype(np.float32),
        right=track.right_wall().astype(np.float32),
        kerb=kerb.astype(np.float32),
        kerb_red=kerb_red,
        meta=np.array([f"{get(track_id).name}", f"{year} Q {lap.driver}", f"{lap.lap_time_s:.3f}"]),
    )
    print(
        f"{get(track_id).name} {year} Q, {lap.driver} {lap.lap_time_s:.3f} s: line {length:.0f} m, "
        f"{len(rt.corners(track))} corners, {len(kerb)} kerb triangles, lap by speed "
        f"{lap_from_speed:.2f} s, speed {speed.min() * 3.6:.0f}-{speed.max() * 3.6:.0f} km/h "
        f"-> {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB)"
    )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(*(args[:1]), *([int(args[1])] if len(args) > 1 else [])))
