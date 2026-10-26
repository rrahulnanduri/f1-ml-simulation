"""The training view's focus overlay, held on Enter: the logic, without a window.

While Enter is held the overlay fades in (:class:`Fade`): the field and the track dim, the
generation's best car is ringed, the real pole lap drives beside it as a ghost, each of the
two gets a callout on a leader line, and the three sectors are traced on the track. Release
Enter and it fades out. The drawing is :mod:`f1sim.app.ui.focus_layer`.

- :class:`LapTimer` times one car of the playback live, from its progress: when it crosses
  the line, each sector split and the end of the lap.
- :class:`PoleGhost` is the real lap (a ghost NPZ) in the lap-clock frame, with its sector
  times; it's placed on the AI car's lap clock, so both start the lap together.
- :func:`callout` turns either into what a callout shows: the clock, the sector times
  revealed so far, how far each sector's bar is filled, and (for the AI) the gap to the
  pole in each sector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from f1sim.app.ui import timing as tm

FADE_IN_S = 0.35
FADE_OUT_S = 0.25
N_SECTORS = 3


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class Fade:
    """A 0..1 level that rises while ``held`` and falls when not, eased at both ends."""

    def __init__(self) -> None:
        self.u = 0.0

    def update(self, dt: float, held: bool) -> None:
        step = dt / (FADE_IN_S if held else FADE_OUT_S)
        self.u = min(max(self.u + (step if held else -step), 0.0), 1.0)

    @property
    def value(self) -> float:
        return smoothstep(self.u)


def sector_marks(sectors_s, length: float) -> np.ndarray:
    """``[0, split_1, split_2, length]``: three sectors (the track's, or thirds if it has
    another number of them)."""
    b = tm.sector_boundaries(sectors_s, length)
    if b.size != N_SECTORS - 1:
        b = np.array([length / 3.0, 2.0 * length / 3.0])
    return np.concatenate([[0.0], b, [length]])


class LapTimer:
    """One car's lap timed live from its progress since the reset.

    The car started at ``start_s`` along the track, so it crosses the line after
    ``lead = (length - start_s) % length`` metres of progress, and each mark ``m`` of
    :func:`sector_marks` after ``lead + m``. Crossing times are interpolated between steps.
    Before the line the lap clock is negative: the time left to the line at the start
    speed ``start_v``.
    """

    def __init__(self, start_s: float, start_v: float, length: float, marks: np.ndarray):
        self.length = float(length)
        self.lead = float((self.length - start_s) % self.length)
        self.marks = np.asarray(marks, dtype=np.float64)
        self.t_line_est = self.lead / max(float(start_v), 1.0)
        self.t_line: float | None = None if self.lead > 0 else 0.0
        self.splits: list[float] = []  # lap clock at each mark after 0 (the last: lap end)
        self._prev: tuple[float, float] | None = None

    def feed(self, t: float, progress: float) -> None:
        """The car's state after a step: time since the reset and progress (m)."""
        if self._prev is not None:
            t0, p0 = self._prev
            targets = [self.lead + m for m in self.marks]
            done = (0 if self.t_line is None else 1) + len(self.splits)
            while done < len(targets) and p0 < targets[done] <= progress:
                tc = t0 + (t - t0) * (targets[done] - p0) / max(progress - p0, 1e-9)
                if self.t_line is None:
                    self.t_line = tc
                else:
                    self.splits.append(tc - self.t_line)
                done += 1
        self._prev = (t, progress)

    def lap_clock(self, t: float) -> float:
        return t - (self.t_line if self.t_line is not None else self.t_line_est)

    def lap_s(self, progress: float) -> float:
        """Metres into the lap (negative before the line)."""
        return progress - self.lead

    @property
    def sector_times(self) -> list[float]:
        cum = [0.0, *self.splits]
        out = [cum[k + 1] - cum[k] for k in range(len(cum) - 1)]
        return out + [math.nan] * (N_SECTORS - len(out))

    @property
    def lap_time(self) -> float:
        return self.splits[-1] if len(self.splits) == N_SECTORS else math.nan


@dataclass(frozen=True)
class PoleGhost:
    """The real lap: a :class:`~f1sim.app.replay.PreparedTrace`, its sector times and lap
    time, and how to name it."""

    prepared: object
    sector_times: tuple[float, ...]
    lap_time: float
    label: str  # "Russell · 2024 pole"
    team: str | None

    @classmethod
    def load(cls, path, track, label: str) -> PoleGhost:
        from f1sim.app.replay import PreparedTrace, lap_time_of, load_trace, with_track_s

        trace = with_track_s(load_trace(path), track)
        p = PreparedTrace(trace, track.length)
        marks = sector_marks(track.data.sectors_s, track.length)
        secs = tm.sector_times(p.t, p.s, list(marks[:-1]), track.length)
        team = (trace.meta or {}).get("team")
        return cls(p, tuple(secs), lap_time_of(p, track.length), label, team)


@dataclass(frozen=True)
class Callout:
    """What one callout shows."""

    label: str
    clock: float  # lap clock now, or the lap time once the lap is done
    finished: bool
    sector_times: tuple[float, ...]  # nan until that sector is done
    fills: tuple[float, ...]  # 0..1 per sector bar
    widths: tuple[float, ...]  # share of the bar per sector (sums to 1)
    deltas: tuple[float, ...] | None  # this minus the pole per sector (nan where unknown)


def _fills(lap_s: float, marks: np.ndarray) -> tuple[float, ...]:
    out = []
    for k in range(N_SECTORS):
        a, b = marks[k], marks[k + 1]
        out.append(min(max((lap_s - a) / (b - a), 0.0), 1.0))
    return tuple(out)


def bar_widths(reference: tuple[float, ...] | list[float], marks: np.ndarray) -> tuple:
    """Each sector's share of the bar: by the reference's sector times when it has all three,
    else by the sectors' lengths."""
    ref = [float(v) for v in reference]
    if len(ref) == N_SECTORS and all(math.isfinite(v) and v > 0 for v in ref):
        total = sum(ref)
        return tuple(v / total for v in ref)
    lengths = np.diff(marks)
    return tuple(float(v) for v in lengths / lengths.sum())


def revealed(times, clock: float) -> tuple[float, ...]:
    """Sector times whose sector has ended by ``clock`` (nan for the rest)."""
    out, cum = [], 0.0
    for v in times:
        cum += v
        out.append(v if math.isfinite(cum) and clock >= cum - 1e-9 else math.nan)
    return tuple(out)


def callout(
    label: str,
    clock: float,
    lap_s: float,
    sector_times,
    lap_time: float,
    marks: np.ndarray,
    widths: tuple[float, ...],
    pole_times=None,
) -> Callout:
    """A callout from a car's lap clock, metres into the lap and sector times (nan until
    known); ``pole_times`` adds the per-sector gaps to the pole."""
    finished = math.isfinite(lap_time) and clock >= lap_time - 1e-9
    secs = tuple(float(v) for v in sector_times)
    deltas = None
    if pole_times is not None:
        pole_now = revealed(pole_times, float("inf"))
        deltas = tuple(a - b for a, b in zip(secs, pole_now, strict=True))
    return Callout(
        label=label,
        clock=lap_time if finished else clock,
        finished=finished,
        sector_times=secs,
        fills=(1.0,) * N_SECTORS if finished else _fills(lap_s, marks),
        widths=widths,
        deltas=deltas,
    )
