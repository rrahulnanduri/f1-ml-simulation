"""Track limits from where real cars drive: every driver's fastest qualifying lap, combined.

The TUM widths mark the painted white lines, measured from satellite images. Real cars legally
use the kerbs (F1 only requires one wheel on the line), and TUM's widths are locally wrong in
places. This module builds a per-side **limit** at each track point:

1. Take each driver's fastest qualifying lap for several years (OpenF1, 2023 onwards), map it
   into the track frame with the track's stored alignment, smooth and resample it at 20 Hz
   like the ghosts, and project it to ``(s, n)``.
2. Per ``bin_m`` of ``s`` and per side, take each lap's furthest excursion, then the
   ``lap_percentile`` across laps (robust to one lap's mistake).
3. Add half a car width plus a margin, smooth along ``s`` (a rolling max, then a moving average),
   and never go below the white line or more than ``max_extra_m`` beyond it (no shortcuts).
4. Repair the limit lines like the walls in ``track_builder`` (clip inside bends, keep the
   rungs from crossing) and check they're simple.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from f1sim.track.track import Track, TrackData

CAR_HALF_WIDTH_M = 1.0  # same as the off-track rule (progress.ProgressParams.car_half_width_m)
RESAMPLE_HZ = 20.0  # laps are smoothed and resampled like the ghosts (ghost_builder)


@dataclass(frozen=True)
class LapSample:
    """One lap in the track frame."""

    label: str  # e.g. "VER 2025"
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class LimitParams:
    bin_m: float = 10.0
    lap_percentile: float = 90.0
    margin_m: float = 0.25
    max_extra_m: float = 4.0
    rolling_max_m: float = 20.0
    smooth_m: float = 30.0
    outlier_m: float = 6.0  # samples this far past the white line are treated as glitches


DEFAULT_PARAMS = LimitParams()


def collect_laps(track: Track, track_id: str, years, *, max_drivers: int = 25) -> list[LapSample]:
    """Every driver's fastest qualifying lap for ``years`` (OpenF1), in the track frame.

    Uses the track's stored alignment; drivers without a timed lap are skipped.
    """
    from f1sim.data import laps, openf1_source
    from f1sim.data.alignment import Alignment, apply_alignment
    from f1sim.data.ghost_builder import smooth_positions

    meta = track.data.fastf1_alignment
    if not meta:
        raise ValueError(f"{track_id} has no stored alignment; build it with a year first")
    al = Alignment(**{k: meta[k] for k in Alignment.__dataclass_fields__})
    out: list[LapSample] = []
    for year in years:
        session = openf1_source.find_session(int(year), track_id, "Q")
        drivers = openf1_source._get("drivers", {"session_key": session["session_key"]})
        numbers = sorted({int(d["driver_number"]) for d in drivers})[:max_drivers]
        for number in numbers:
            try:
                lap, _ = laps.reference_lap(track_id, int(year), driver=number, source="openf1")
            except Exception:  # no timed lap, missing data
                continue
            x, y = apply_alignment(al, lap.x_m, lap.y_m)
            # Raw positions are ~3.7 Hz (~20 m apart at speed) and skip each lap's widest
            # points; smooth and resample like the ghosts do so the envelope sees them
            t = np.arange(0.0, lap.lap_time_s, 1.0 / RESAMPLE_HZ)
            sm = smooth_positions(lap.t, x, y, t)
            out.append(LapSample(f"{lap.driver} {year}", sm["x"], sm["y"]))
    return out


def driving_envelope(
    track: Track, samples: list[LapSample], params: LimitParams = DEFAULT_PARAMS
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per track point, how far left and right real cars' centres go (metres, >= 0)."""
    if not samples:
        raise ValueError("no laps to build an envelope from")
    length = track.length
    n_bins = max(1, int(round(length / params.bin_m)))
    per_lap_left = np.full((len(samples), n_bins), np.nan)
    per_lap_right = np.full((len(samples), n_bins), np.nan)
    w_left_pts, w_right_pts = track.data.w_left, track.data.w_right
    for i, lap in enumerate(samples):
        s, n, _ = track.project(lap.x, lap.y)
        wl, wr = track.widths_at(s)
        glitch = (n > wl + params.outlier_m) | (-n > wr + params.outlier_m)
        s, n = s[~glitch], n[~glitch]
        b = np.minimum((s / length * n_bins).astype(int), n_bins - 1)
        left = np.where(n > 0, n, 0.0)
        right = np.where(n < 0, -n, 0.0)
        np.fmax.at(per_lap_left[i], b, left)
        np.fmax.at(per_lap_right[i], b, right)
    with warnings.catch_warnings():  # bins no lap reached are all-NaN; filled below
        warnings.simplefilter("ignore", RuntimeWarning)
        env_left_bins = np.nanpercentile(per_lap_left, params.lap_percentile, axis=0)
        env_right_bins = np.nanpercentile(per_lap_right, params.lap_percentile, axis=0)
    # Bins no lap reached (rare) fall back to the white line
    centres = (np.arange(n_bins) + 0.5) * length / n_bins
    fill_l = np.interp(centres, track.data.s, w_left_pts) - CAR_HALF_WIDTH_M
    fill_r = np.interp(centres, track.data.s, w_right_pts) - CAR_HALF_WIDTH_M
    env_left_bins = np.where(np.isfinite(env_left_bins), env_left_bins, fill_l)
    env_right_bins = np.where(np.isfinite(env_right_bins), env_right_bins, fill_r)
    # Bins -> track points (periodic)
    s_pts = track.data.s
    xp = np.concatenate([centres - length, centres, centres + length])
    env_left = np.interp(s_pts, xp, np.tile(env_left_bins, 3))
    env_right = np.interp(s_pts, xp, np.tile(env_right_bins, 3))
    stats = {"laps": len(samples), "bins": n_bins}
    return env_left, env_right, stats


def _smooth_periodic(values: np.ndarray, spacing_m: float, params: LimitParams) -> np.ndarray:
    k_max = max(1, int(round(params.rolling_max_m / spacing_m)) | 1)
    k_avg = max(1, int(round(params.smooth_m / spacing_m)))
    v = maximum_filter1d(values, size=k_max, mode="wrap")
    return uniform_filter1d(v, size=k_avg, mode="wrap")


def compute_limits(
    track: Track, env_left: np.ndarray, env_right: np.ndarray, params: LimitParams = DEFAULT_PARAMS
) -> tuple[np.ndarray, np.ndarray]:
    """Limit widths per point: the envelope plus half a car and a margin, repaired and clamped."""
    from f1sim.data.track_builder import _WALL_CLIP, _local_reach

    d = track.data
    spacing = d.length_m / d.x.size
    want_l = _smooth_periodic(env_left + CAR_HALF_WIDTH_M + params.margin_m, spacing, params)
    want_r = _smooth_periodic(env_right + CAR_HALF_WIDTH_M + params.margin_m, spacing, params)
    lim_l = np.clip(want_l, d.w_left, d.w_left + params.max_extra_m)
    lim_r = np.clip(want_r, d.w_right, d.w_right + params.max_extra_m)
    # Inside of a bend: keep the limit line from folding (as in track_builder)
    k = d.curvature
    with np.errstate(divide="ignore"):
        bend = np.where(np.abs(k) > 1e-12, _WALL_CLIP / np.abs(k), np.inf)
    lim_l = np.where(k > 0, np.minimum(lim_l, bend), lim_l)
    lim_r = np.where(k < 0, np.minimum(lim_r, bend), lim_r)
    nl = np.column_stack([-np.sin(d.heading), np.cos(d.heading)])
    reach_l, reach_r = _local_reach(d.x, d.y, nl, float(max(lim_l.max(), lim_r.max())))
    lim_l = np.minimum(lim_l, _WALL_CLIP * reach_l)
    lim_r = np.minimum(lim_r, _WALL_CLIP * reach_r)
    # Never inside the white line
    return np.maximum(lim_l, d.w_left), np.maximum(lim_r, d.w_right)


def limit_self_crossings(data: TrackData) -> tuple[int, int]:
    """Self-crossings of the (left, right) limit lines; (0, 0) when simple or absent."""
    from f1sim.data.track_builder import _self_crossing_points

    if data.limit_left is None:
        return 0, 0
    t = Track(data)
    return (
        len(_self_crossing_points(t.limit_left_wall())),
        len(_self_crossing_points(t.limit_right_wall())),
    )


def with_limits(
    data: TrackData, samples: list[LapSample], years, params: LimitParams = DEFAULT_PARAMS
) -> TrackData:
    """``data`` with limit widths built from ``samples``; raises if a limit line folds."""
    track = Track(data)
    env_l, env_r, stats = driving_envelope(track, samples, params)
    lim_l, lim_r = compute_limits(track, env_l, env_r, params)
    out = replace(
        data,
        limit_left=lim_l,
        limit_right=lim_r,
        limits={
            "source": "openf1-qualifying-envelope",
            "years": [int(y) for y in years],
            "laps": stats["laps"],
            "lap_percentile": params.lap_percentile,
            "car_half_width_m": CAR_HALF_WIDTH_M,
            "margin_m": params.margin_m,
            "max_extra_m": params.max_extra_m,
        },
    )
    crossings = limit_self_crossings(out)
    if crossings != (0, 0):
        raise ValueError(f"{data.id}: limit lines cross themselves {crossings}")
    return out
