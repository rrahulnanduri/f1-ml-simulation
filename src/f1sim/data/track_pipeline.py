"""Catalogue entry -> TUM geometry -> built track -> FastF1 alignment -> cached track JSON.

Implements revamp-plan sections 4.2 (orientation, sectors) and 4.3 (acceptance) on top of the
owner modules in docs/m1-interfaces.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from f1sim import paths
from f1sim.data import catalog, tum_source
from f1sim.data.io import read_track_json, write_track_json
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track, TrackData

if TYPE_CHECKING:
    from f1sim.data.alignment import Alignment

# Section 4.3 acceptance. A real lap runs kerb to kerb (TUM's own optimal racing line sits a median
# 5.1 m off the centre line at Silverstone), and position noise puts corner exits just past the
# edges, so "inside" allows EDGE_TOLERANCE_M. A lap from a different layout scores far below both.
MAX_MEDIAN_DIST_M = 6.0
MIN_INSIDE_RATIO = 0.98
EDGE_TOLERANCE_M = 1.0


class AlignmentRejected(RuntimeError):
    """The FastF1 lap doesn't fit this TUM layout (e.g. the circuit changed since TUM's survey)."""


@dataclass(frozen=True)
class BuildReport:
    track: TrackData
    path: Path
    alignment: Alignment | None
    reference_driver: str | None
    reference_lap_s: float | None
    lap_source: str | None = None


def track_path(track_id: str) -> Path:
    return paths.tracks_dir() / f"{track_id}.track.json"


def load_track(track_id: str) -> Track:
    """Load a built track from the cache. Raises FileNotFoundError if it hasn't been built."""
    return Track(read_track_json(track_path(track_id)))


def build_and_cache(
    track_id: str, *, year: int | None = 2024, resolution_m: float = 2.0, source: str = "auto"
) -> BuildReport:
    """Build `track_id` from TUM data and write it to the cache.

    With `year`, the fastest qualifying lap of that year is aligned to the track to fix the
    start/finish line, racing direction and sector boundaries, and to validate the layout.
    `source` picks where the lap comes from (see `laps.reference_lap`): "auto" uses OpenF1 for
    2023 onwards and FastF1 (cached sessions only) before that.
    Raises AlignmentRejected if the lap doesn't fit (section 4.3).
    """
    entry = catalog.get(track_id)
    if entry.tum_name is None:
        raise NotImplementedError(
            f"{entry.name} isn't in the TUM racetrack-database; the FastF1 width-estimation "
            "fallback arrives in M5"
        )
    raw = tum_source.parse_tum_csv(tum_source.fetch_tum_csv(entry.tum_name))
    geometry_source = {
        "geometry": "tum-racetrack-database",
        "commit": tum_source.TUM_COMMIT,
        "license": "LGPL-3.0",
        "widths": "tum",
    }

    def build(x, y, w_right, w_left) -> TrackData:
        return build_track(
            x,
            y,
            w_right,
            w_left,
            track_id=entry.id,
            name=entry.name,
            resolution_m=resolution_m,
            source=geometry_source,
        )

    data = build(raw.x, raw.y, raw.w_right, raw.w_left)
    alignment = None
    driver = lap_time = lap_source = None

    if year is not None:
        from f1sim.data import laps
        from f1sim.data.alignment import align_to_centreline, apply_alignment

        lap, lap_source = laps.reference_lap(entry.id, year, source=source)
        driver, lap_time = lap.driver, lap.lap_time_s

        alignment = align_to_centreline(lap.x_m, lap.y_m, data.x, data.y, data.w_left, data.w_right)
        lx, ly = apply_alignment(alignment, lap.x_m, lap.y_m)
        inside_tol = inside_ratio(Track(data), lx, ly, EDGE_TOLERANCE_M)
        if alignment.median_dist_m > MAX_MEDIAN_DIST_M or inside_tol < MIN_INSIDE_RATIO:
            raise AlignmentRejected(
                f"{entry.name} {year}: {lap.driver}'s lap is a median "
                f"{alignment.median_dist_m:.1f} m from the TUM centre line with {inside_tol:.1%} "
                f"within {EDGE_TOLERANCE_M:g} m of the edges (need <= {MAX_MEDIAN_DIST_M:g} m and "
                f">= {MIN_INSIDE_RATIO:.0%}). The layout may have changed."
            )

        x, y, w_right, w_left = _orient_like_lap(raw, data, lx, ly)
        data = build(x, y, w_right, w_left)

        track = Track(data)
        sector_t = np.asarray(lap.sector_end_t)
        sx = np.interp(sector_t, lap.t, lx)
        sy = np.interp(sector_t, lap.t, ly)
        sector_s, _, _ = track.project(sx, sy)
        data = replace(
            data,
            sectors_s=(0.0, *(round(float(s), 1) for s in sector_s)),
            fastf1_location=entry.fastf1_locations[0],
            fastf1_alignment={
                "years_validated": [year],
                "lap_source": lap_source,
                "reference_driver": lap.driver,
                "reference_lap_s": lap.lap_time_s,
                **asdict(alignment),
                "inside_ratio_tol": inside_tol,
                "edge_tolerance_m": EDGE_TOLERANCE_M,
            },
        )

    out = track_path(entry.id)
    write_track_json(data, out)
    return BuildReport(data, out, alignment, driver, lap_time, lap_source)


def inside_ratio(track: Track, x: np.ndarray, y: np.ndarray, tolerance_m: float = 0.0) -> float:
    """Share of points within `tolerance_m` of the track edges (or inside them)."""
    s, n, _ = track.project(x, y)
    w_left, w_right = track.widths_at(s)
    inside = (n <= w_left + tolerance_m) & (n >= -w_right - tolerance_m)
    return float(inside.mean())


def _orient_like_lap(raw, data: TrackData, lx: np.ndarray, ly: np.ndarray):
    """Reorder the raw TUM loop so it runs in the lap's direction and starts at the lap start."""
    track = Track(data)
    s, _, _ = track.project(lx, ly)
    # Progress along the lap, unwrapped across the start/finish; its sign gives the direction
    ds = np.diff(s)
    ds = (ds + track.length / 2) % track.length - track.length / 2
    forward = ds.sum() > 0

    x, y, w_right, w_left = raw.x, raw.y, raw.w_right, raw.w_left
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-6:  # drop a repeated end point
        x, y, w_right, w_left = x[:-1], y[:-1], w_right[:-1], w_left[:-1]
    if not forward:
        # Driving the other way swaps which side is left
        x, y, w_right, w_left = x[::-1], y[::-1], w_left[::-1], w_right[::-1]

    start = int(np.argmin(np.hypot(x - lx[0], y - ly[0])))
    return tuple(np.roll(a, -start) for a in (x, y, w_right, w_left))
