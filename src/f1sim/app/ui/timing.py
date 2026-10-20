"""Timing for the HUD, pure numpy: sector times, gaps at equal ``s``, speed traces, sector
colours, the live-lap recorder of the drive viewer and the timing tower's rows.

Arc lengths are in the lap frame of :mod:`f1sim.app.replay`: ``s = 0`` is the start/finish
line and a lap runs to ``s = length``. Times are lap-clock seconds (0 at the line).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NAN = float("nan")


# ------------------------------------------------------------------------- formatting


def format_lap(t: float | None) -> str:
    """``1:24.892``, or ``-:--.---`` for ``None``/nan."""
    if t is None or not math.isfinite(t):
        return "-:--.---"
    minutes, seconds = divmod(float(t), 60.0)
    return f"{int(minutes)}:{seconds:06.3f}"


def format_gap(d: float | None) -> str:
    """``+0.927`` / ``-0.312``; ``--.---`` for ``None``/nan."""
    if d is None or not math.isfinite(d):
        return "--.---"
    return f"{d:+.3f}"


def format_sector(t: float | None) -> str:
    """``28.410``; ``--.---`` for ``None``/nan."""
    if t is None or not math.isfinite(t):
        return "--.---"
    return f"{t:.3f}"


# ------------------------------------------------------------------- along-track maths


def monotonic(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples where ``s`` reaches a new maximum: strictly increasing ``s`` with its ``t``."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if s.size == 0:
        return s, t
    keep = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]])
    return s[keep], t[keep]


def time_at(s_mono: np.ndarray, t_mono: np.ndarray, s_query) -> np.ndarray:
    """Time at arc length(s) ``s_query``, linear; nan outside the covered range."""
    q = np.asarray(s_query, dtype=np.float64)
    if s_mono.size == 0:
        return np.full(q.shape, np.nan)
    out = np.interp(q, s_mono, t_mono)
    return np.where((q < s_mono[0]) | (q > s_mono[-1]), np.nan, out)


def gap_at_s(t_self, s_self, t_ref, s_ref, s_query) -> np.ndarray:
    """``t_self(s) - t_ref(s)`` at equal arc length: positive = self is behind."""
    sm, tm = monotonic(s_self, t_self)
    rm, rt = monotonic(s_ref, t_ref)
    return time_at(sm, tm, s_query) - time_at(rm, rt, s_query)


def sector_boundaries(sectors_s, length: float) -> np.ndarray:
    """The split points inside the lap (``0 < b < length``), sorted; ``sectors_s`` is the
    track's list of sector starts (``0.0`` first). No sectors -> one sector, no splits."""
    b = np.asarray([float(v) for v in sectors_s], dtype=np.float64)
    return np.sort(b[(b > 0.0) & (b < length)])


def sector_times(t, s, sectors_s, length: float) -> list[float]:
    """Sector times of one lap trace: the time between consecutive crossings of ``0``, the
    split points and ``length``, interpolated in ``t``. nan for a sector the trace doesn't
    fully cover."""
    sm, tm = monotonic(s, t)
    marks = np.concatenate([[0.0], sector_boundaries(sectors_s, length), [length]])
    at = time_at(sm, tm, marks)
    return [float(v) for v in np.diff(at)]


def resample_speed(s, speed, length: float, n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Speed on ``n`` evenly spaced arc lengths over ``[0, length]``: ``(s_grid, v_grid)``,
    nan where the trace doesn't reach. Uses the monotonic part of ``s``."""
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(speed, dtype=np.float64)
    grid = np.linspace(0.0, length, n)
    if s.size == 0:
        return grid, np.full(n, np.nan)
    keep = np.concatenate([[True], s[1:] > np.maximum.accumulate(s)[:-1]])
    sm, vm = s[keep], v[keep]
    out = np.interp(grid, sm, vm)
    out[(grid < sm[0]) | (grid > sm[-1])] = np.nan
    return grid, out


def sector_colour(time: float, personal_best: float | None, session_best: float | None) -> str:
    """F1 sector colour: ``"purple"`` for the session's best (``time <= session_best``),
    ``"green"`` for a personal best (``time <= personal_best``, or no earlier personal best),
    ``"yellow"`` when slower, ``"unset"`` when ``time`` is nan/None."""
    if time is None or not math.isfinite(time):
        return "unset"
    if session_best is None or not math.isfinite(session_best) or time <= session_best + 1e-9:
        return "purple"
    if personal_best is None or not math.isfinite(personal_best) or time <= personal_best + 1e-9:
        return "green"
    return "yellow"


# ------------------------------------------------------------------------ live laps


class LapTelemetry:
    """Records the drive viewer's live lap: sector splits as the car crosses them, a speed
    trace binned along the lap, and per-sector personal bests.

    Call :meth:`record` after every physics step with the distance since the line
    (``progress``), the lap clock and the speed; :meth:`complete_lap` when a lap is counted
    (before recording the next lap's first sample) and :meth:`discard_lap` on a reset.
    """

    def __init__(self, length: float, sectors_s=(), n_bins: int = 160):
        self.length = float(length)
        self.bounds = sector_boundaries(sectors_s, self.length)
        self.n_sectors = self.bounds.size + 1
        self.n_bins = int(n_bins)
        self.grid = np.linspace(0.0, self.length, self.n_bins)
        self.speed = np.full(self.n_bins, np.nan)
        self.current = [NAN] * self.n_sectors
        self.last: list[float] | None = None
        self.best = [NAN] * self.n_sectors  # personal best per sector, all laps so far
        self.best_before = [NAN] * self.n_sectors  # personal bests when this lap started
        self._last_best_before = [NAN] * self.n_sectors
        self._splits = [NAN] * self.bounds.size  # lap clock at each split this lap
        self._prev: tuple[float, float] | None = None

    def record(self, progress: float, lap_time: float, speed: float) -> None:
        prev = self._prev
        self._prev = (float(progress), float(lap_time))
        if 0.0 <= progress <= self.length:
            k = int(round(progress / self.length * (self.n_bins - 1)))
            self.speed[k] = speed
        if prev is None:
            return
        p0, t0 = prev
        if progress <= p0:
            return
        for k, b in enumerate(self.bounds):
            if p0 < b <= progress and not math.isfinite(self._splits[k]):
                t_cross = t0 + (lap_time - t0) * (b - p0) / (progress - p0)
                self._splits[k] = t_cross
                start = 0.0 if k == 0 else self._splits[k - 1]
                self.current[k] = t_cross - start if math.isfinite(start) else NAN
                self._update_best(k)

    def complete_lap(self, lap_time: float) -> None:
        last_split = self._splits[-1] if self._splits else 0.0
        k = self.n_sectors - 1
        self.current[k] = lap_time - last_split if math.isfinite(last_split) else NAN
        self._update_best(k)
        self.last = list(self.current)
        self._last_best_before = list(self.best_before)
        self._new_lap()

    def discard_lap(self) -> None:
        self._new_lap()

    def _new_lap(self) -> None:
        self.current = [NAN] * self.n_sectors
        self._splits = [NAN] * self.bounds.size
        self.best_before = list(self.best)
        self.speed = np.full(self.n_bins, np.nan)
        self._prev = None

    def _update_best(self, k: int) -> None:
        v = self.current[k]
        if math.isfinite(v) and not (math.isfinite(self.best[k]) and self.best[k] <= v):
            self.best[k] = v

    def displayed(self) -> tuple[list[float], list[float]]:
        """``(times, personal_bests_before)`` to show: the current lap's sectors, or the last
        completed lap's while the current lap has none yet."""
        if self.last is not None and not any(math.isfinite(v) for v in self.current):
            return list(self.last), list(self._last_best_before)
        return list(self.current), list(self.best_before)


# ----------------------------------------------------------------------- the tower


@dataclass(frozen=True)
class TowerEntry:
    """One car or lap for the tower: ``lap_time`` (s, None if not set), sliver colour and
    whether it's the highlighted "self" row."""

    name: str
    lap_time: float | None
    colour: tuple
    is_self: bool = False


@dataclass(frozen=True)
class TowerRow:
    position: str
    name: str
    value: str
    colour: tuple
    is_self: bool


def tower_rows(entries: list[TowerEntry]) -> list[TowerRow]:
    """Rank entries by lap time (no time last, input order kept on ties). The leader shows
    its absolute lap time (``1:24.892``), the others the gap to it (``+0.927``); entries
    without a time show ``-:--.---`` and no position."""
    has_time = [e.lap_time is not None and math.isfinite(e.lap_time) for e in entries]
    timed = sorted(
        (e for e, h in zip(entries, has_time, strict=True) if h), key=lambda e: e.lap_time
    )
    untimed = [e for e, h in zip(entries, has_time, strict=True) if not h]
    rows: list[TowerRow] = []
    leader = timed[0].lap_time if timed else None
    for i, e in enumerate(timed):
        value = format_lap(e.lap_time) if i == 0 else format_gap(e.lap_time - leader)
        rows.append(TowerRow(str(i + 1), e.name, value, e.colour, e.is_self))
    for e in untimed:
        rows.append(TowerRow("", e.name, format_lap(None), e.colour, e.is_self))
    return rows


def live_positions(s_values: list[float]) -> list[int]:
    """Race position (1 = furthest along) of each car from its current arc length."""
    order = sorted(range(len(s_values)), key=lambda i: -s_values[i])
    pos = [0] * len(s_values)
    for p, i in enumerate(order):
        pos[i] = p + 1
    return pos
