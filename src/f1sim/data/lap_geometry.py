"""Track geometry from real laps, for circuits TUM's database doesn't have (M5, D40).

Monaco, Baku, Singapore, Jeddah, Miami, Imola, Las Vegas, Lusail and Madrid have no TUM file,
and some TUM layouts are out of date (Yas Marina, Albert Park, Barcelona, Zandvoort). Every
qualifying lap since 2023 is on OpenF1, so the road is rebuilt from where the cars drove:

1. **Laps.** Each driver's fastest qualifying lap for ``years`` (OpenF1), smoothed and
   resampled at 20 Hz like the ghosts. The fastest lap of the newest year is the
   *reference*. Every other lap is fitted to it (``alignment.align_to_centreline``, so a
   frame shift between years doesn't matter) and dropped if it sits a median more than
   ``MAX_LAP_DIST_M`` off it: a different layout, or a bad trace.
2. **Envelope.** The reference lap becomes a provisional track (its line as the centre). Each
   lap is projected onto it; per ``BIN_M`` of distance and per side, each lap's furthest
   excursion, then the ``LAP_PERCENTILE`` across laps (one lap's mistake doesn't widen the
   road), smoothed like the track limits.
3. **Road.** Each edge is the envelope plus half a car and ``KERB_M``. Where that is narrower
   than the circuit's :func:`min_width_m` the extra width goes to the outside of the bend in
   corners (the cars take the apex, so the inside edge is known; the outside isn't) and to
   both sides on straights, blended by ``tanh(|curvature| * BEND_RADIUS_M)``.
4. **Track.** The centre is half-way between the edges; ``track_builder.build_track``
   smooths, resamples and repairs it as for a TUM file. Start/finish, direction and sectors
   come from the reference lap, and the stored alignment is the identity (OpenF1's frame),
   so ghosts and limits line up with no fitting.

The result is approximate: the edges are where cars went plus a margin, not the painted lines.
Every driver's fastest lap follows nearly the same line (at Monaco the envelope is a 4.4 m
road, median), so in practice the width is the circuit's minimum width almost everywhere.
``source["widths"]`` is ``"lap-envelope"`` so the UI can say so. No separate track limits
are built: the edges already are where real cars drive.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d

from f1sim.data.track_limits import CAR_HALF_WIDTH_M, RESAMPLE_HZ, LapSample
from f1sim.track.track import Track, TrackData

log = logging.getLogger(__name__)

BIN_M = 5.0
LAP_PERCENTILE = 90.0
KERB_M = 0.75
"""Room past the furthest car centre plus half a car: kerbs, and position noise."""
MIN_WIDTH_M = 12.0
"""Narrowest road the envelope is widened to: the FIA minimum for a permanent circuit."""
NARROW_CIRCUITS_M = {"monaco": 9.0, "baku": 10.0}
"""Street circuits narrower than :data:`MIN_WIDTH_M` for much of the lap."""
BEND_RADIUS_M = 60.0
EDGE_SMOOTH_M = 30.0
ROLLING_MAX_M = 15.0
MAX_LAP_DIST_M = 3.0
REFERENCE_WIDTH_M = 8.0
"""Half-width of the provisional track the laps are projected on (wide enough for any lap)."""
SEAM_TRIM_S = 0.5
"""Trimmed off both ends of the smoothed reference lap: a smoothing spline is loose at its
ends, and joining them put a kink (a 1 m radius) on the start/finish line. The periodic
spline in ``build_track`` bridges the gap."""
GEOMETRY_LICENSE = "CC BY-NC-SA 4.0 (OpenF1 data)"
DEFAULT_YEARS = (2023, 2024, 2025, 2026)
"""OpenF1's seasons; a year without a qualifying session at the circuit is skipped."""


@dataclass(frozen=True)
class LapSet:
    """The reference lap and every accepted lap, all in the reference lap's frame."""

    reference: object  # laps.LapXY
    reference_year: int
    years: list[int]  # the years that had a session
    samples: list[LapSample]
    dropped: list[str]


def _smooth(lap) -> tuple[np.ndarray, np.ndarray]:
    from f1sim.data.ghost_builder import smooth_positions

    t = np.arange(0.0, lap.lap_time_s, 1.0 / RESAMPLE_HZ)
    sm = smooth_positions(lap.t, lap.x_m, lap.y_m, t)
    return sm["x"], sm["y"]


def collect_laps(track_id: str, years, *, max_drivers: int = 25) -> LapSet:
    """Every driver's fastest qualifying lap of ``years`` (OpenF1), fitted to the reference
    (the newest year's pole lap). Years without a qualifying session there are skipped."""
    from f1sim.data import laps, openf1_source
    from f1sim.data.alignment import align_to_centreline, apply_alignment

    sessions = {}
    for year in sorted({int(y) for y in years}, reverse=True):
        try:
            sessions[year] = openf1_source.find_session(year, track_id, "Q")
        except openf1_source.OpenF1Error as exc:
            log.info("%s %d: no qualifying session (%s)", track_id, year, exc)
    if not sessions:
        raise ValueError(f"{track_id}: no OpenF1 qualifying session in {sorted(years)}")
    years = list(sessions)
    ref_year = years[0]
    ref, _ = laps.reference_lap(track_id, ref_year, source="openf1")
    rx, ry = _smooth(ref)
    wide = np.full(rx.size, REFERENCE_WIDTH_M)
    samples: list[LapSample] = []
    dropped: list[str] = []
    for year, session in sessions.items():
        drivers = openf1_source._get("drivers", {"session_key": session["session_key"]})
        numbers = sorted({int(d["driver_number"]) for d in drivers})[:max_drivers]
        for number in numbers:
            try:
                lap, _ = laps.reference_lap(track_id, year, driver=number, source="openf1")
                x, y = _smooth(lap)
            except Exception:  # no timed lap, missing or too few samples
                continue
            label = f"{lap.driver} {year}"
            al = align_to_centreline(x, y, rx, ry, wide, wide)
            if al.median_dist_m > MAX_LAP_DIST_M:
                dropped.append(f"{label} ({al.median_dist_m:.1f} m off)")
                continue
            ax, ay = apply_alignment(al, x, y)
            samples.append(LapSample(label, ax, ay))
    if dropped:
        log.warning("%s: dropped %d laps that don't fit: %s", track_id, len(dropped), dropped)
    if not samples:
        raise ValueError(f"{track_id}: no laps fit the reference lap")
    return LapSet(ref, ref_year, sorted(years), samples, dropped)


def _reference_track(ref) -> Track:
    """The reference lap as a closed track, rolled so ``s = 0`` is where the lap starts."""
    from dataclasses import replace

    from f1sim.data.track_builder import build_track

    rx, ry = _smooth(ref)
    k = int(round(SEAM_TRIM_S * RESAMPLE_HZ))
    rx, ry = rx[k:-k], ry[k:-k]
    w = np.full(rx.size, REFERENCE_WIDTH_M)
    d = build_track(rx, ry, w, w, track_id="reference", name="reference", max_shift_m=1.0)
    s0, _, _ = Track(d).project(np.array([ref.x_m[0]]), np.array([ref.y_m[0]]))
    i = int(round(float(s0[0]) / (d.length_m / d.x.size))) % d.x.size
    rolled = {f: np.roll(getattr(d, f), -i) for f in ("x", "y", "w_left", "w_right")}
    rolled.update(heading=np.roll(d.heading, -i), curvature=np.roll(d.curvature, -i))
    return Track(replace(d, **rolled))


def envelope(track: Track, samples: list[LapSample]) -> tuple[np.ndarray, np.ndarray]:
    """Per track point, how far left and right of ``track``'s centre line cars' centres go
    (signed: ``left`` is the furthest-left offset, ``right`` the furthest-right as a positive
    number; either may be negative when every lap stays on the other side)."""
    length = track.length
    n_bins = max(1, int(round(length / BIN_M)))
    left = np.full((len(samples), n_bins), np.nan)
    right = np.full((len(samples), n_bins), np.nan)
    for i, lap in enumerate(samples):
        s, n, _ = track.project(lap.x, lap.y)
        b = np.minimum((s / length * n_bins).astype(int), n_bins - 1)
        np.fmax.at(left[i], b, n)
        np.fmax.at(right[i], b, -n)
    with warnings.catch_warnings():  # bins no lap reached: filled by interpolation below
        warnings.simplefilter("ignore", RuntimeWarning)
        env_l = np.nanpercentile(left, LAP_PERCENTILE, axis=0)
        env_r = np.nanpercentile(right, LAP_PERCENTILE, axis=0)
    centres = (np.arange(n_bins) + 0.5) * length / n_bins
    out = []
    for env in (env_l, env_r):
        ok = np.isfinite(env)
        if not ok.any():
            raise ValueError("no lap samples on the track")
        xp = np.concatenate([centres[ok] - length, centres[ok], centres[ok] + length])
        out.append(np.interp(track.data.s, xp, np.tile(env[ok], 3)))
    return out[0], out[1]


def min_width_m(track_id: str) -> float:
    """The road width the envelope is widened to on ``track_id``."""
    return NARROW_CIRCUITS_M.get(track_id, MIN_WIDTH_M)


def road_edges(
    track: Track, env_l: np.ndarray, env_r: np.ndarray, min_width: float = MIN_WIDTH_M
) -> tuple[np.ndarray, np.ndarray]:
    """``(edge_left, edge_right)``: signed offsets of the road's edges from ``track``'s centre
    line (left positive; ``edge_right`` is the right edge as a positive distance)."""
    d = track.data
    spacing = d.length_m / d.x.size
    k_max = max(1, int(round(ROLLING_MAX_M / spacing)) | 1)
    k_avg = max(1, int(round(EDGE_SMOOTH_M / spacing)))

    def smooth(v):
        return uniform_filter1d(maximum_filter1d(v, size=k_max, mode="wrap"), k_avg, mode="wrap")

    edge_l = smooth(env_l + CAR_HALF_WIDTH_M + KERB_M)
    edge_r = smooth(env_r + CAR_HALF_WIDTH_M + KERB_M)
    k = gaussian_filter1d(d.curvature, EDGE_SMOOTH_M / spacing, mode="wrap")
    deficit = np.maximum(min_width - (edge_l + edge_r), 0.0)
    deficit = gaussian_filter1d(deficit, EDGE_SMOOTH_M / spacing, mode="wrap")
    outside = 0.5 * (1.0 + np.tanh(np.abs(k) * BEND_RADIUS_M))  # share to the bend's outside
    to_left = np.where(k < 0, outside, 1.0 - outside)  # turning right: the outside is left
    edge_l = edge_l + deficit * to_left
    edge_r = edge_r + deficit * (1.0 - to_left)
    return edge_l, edge_r


def geometry_from_laps(lapset: LapSet, min_width: float = MIN_WIDTH_M):
    """``(x, y, w_right, w_left)`` of the road: a closed loop starting at the reference lap's
    start, in its direction."""
    ref_track = _reference_track(lapset.reference)
    env_l, env_r = envelope(ref_track, lapset.samples)
    edge_l, edge_r = road_edges(ref_track, env_l, env_r, min_width)
    d = ref_track.data
    shift = 0.5 * (edge_l - edge_r)
    half = 0.5 * (edge_l + edge_r)
    nx, ny = -np.sin(d.heading), np.cos(d.heading)
    return d.x + shift * nx, d.y + shift * ny, half, half


def build_from_laps(track_id: str, years=DEFAULT_YEARS, *, resolution_m: float = 2.0) -> TrackData:
    """A track for ``track_id`` from the qualifying laps of ``years``; see the module doc."""
    from dataclasses import replace

    from f1sim.data import catalog
    from f1sim.data.alignment import Alignment
    from f1sim.data.track_builder import build_track
    from f1sim.data.track_pipeline import EDGE_TOLERANCE_M, inside_ratio

    entry = catalog.get(track_id)
    lapset = collect_laps(track_id, years)
    x, y, w_right, w_left = geometry_from_laps(lapset, min_width_m(track_id))
    years = lapset.years
    data = build_track(
        x,
        y,
        w_right,
        w_left,
        track_id=entry.id,
        name=entry.name,
        resolution_m=resolution_m,
        source={
            "geometry": "openf1-laps",
            "license": GEOMETRY_LICENSE,
            "widths": "lap-envelope",
            "laps": len(lapset.samples),
            "years": years,
            "min_width_m": min_width_m(track_id),
            "approximate": True,
        },
    )
    ref = lapset.reference
    track = Track(data)
    rx, ry = ref.x_m, ref.y_m
    _, n_ref, _ = track.project(rx, ry)
    sector_s, _, _ = track.project(
        np.interp(ref.sector_end_t, ref.t, rx), np.interp(ref.sector_end_t, ref.t, ry)
    )
    identity = Alignment(
        rotation_rad=0.0,
        tx_m=0.0,
        ty_m=0.0,
        scale=1.0,
        mirrored=False,
        median_dist_m=float(np.median(np.abs(n_ref))),
        inside_ratio=inside_ratio(track, rx, ry),
    )
    return replace(
        data,
        sectors_s=(0.0, *(round(float(s), 1) for s in sector_s)),
        fastf1_location=entry.fastf1_locations[0],
        fastf1_alignment={
            "years_validated": years,
            "lap_source": "openf1",
            "reference_driver": ref.driver,
            "reference_lap_s": ref.lap_time_s,
            "reference_year": lapset.reference_year,
            **asdict(identity),
            "inside_ratio_tol": inside_ratio(track, rx, ry, EDGE_TOLERANCE_M),
            "edge_tolerance_m": EDGE_TOLERANCE_M,
        },
    )
