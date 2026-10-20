"""Progress along the track, lap and sector timing, and termination checks for N cars.

Everything is vectorised over the cars (docs/m2-interfaces.md, section E):

- **Projection.** Each car keeps its segment index as a hint, so :meth:`Track.project` only
  searches ``hint ± window`` segments. A car is re-projected over the whole loop when its
  wrap-aware ``|Δs|`` exceeds :data:`GLOBAL_REPROJECT_DS_M` (the contract rule), and also
  when the hinted pick moved ``window`` or more segments, or when the car moved more than
  ``min(50 m, window/2 segment lengths)`` in a straight line since the last step. The hinted
  search can't see past ``window`` segments (about 20 m with the defaults on a 2 m track), so
  a jump never shows up as a 50 m ``Δs`` on its own; a teleport to the far side of a loop can
  even project to a nearby ``s`` with a huge ``n``. Driving cars (at most a few metres per
  step) never trigger the extra checks.
- **Progress** is the sum of the wrap-aware ``Δs`` since the reset, so it goes down when a
  car drives backwards.
- **Crossings.** A boundary (the start line at ``s = 0``, or a sector boundary
  ``sectors_s[1:]``) is crossed forward when ``Δs > 0`` and the step's arc ``(s_prev, s_new]``
  contains it. Arriving exactly on a boundary counts as crossing it. The crossing time is
  interpolated linearly inside the step: ``frac = (boundary - s_prev) / Δs`` (wrap-aware), so
  lap and sector times are sub-step accurate. When a step crosses several boundaries (only
  after a jump) they're handled in the order they were reached.
- **Laps.** A forward crossing of the start line completes a lap only if
  ``progress - progress_at_lap_start >= lap_min_fraction * length``. Either way the crossing
  becomes the new lap start (time and progress). So driving back and forth over the line never
  counts, and a car reset behind the line starts its first timed lap when it crosses it.
- **Sectors.** Sector ``k`` runs from ``sectors_s[k]`` to the next boundary (the line for the
  last sector). It's reported on a forward crossing of its end boundary when the car was in
  sector ``k`` and its progress since the sector started is at least
  ``lap_min_fraction`` times the sector length (the lap rule, per sector). Any forward
  crossing moves the car into the sector that starts there, so the index returns to 0 at the
  line. After a reset the index is the sector containing the car.
- **Termination** (checked after the step's crossings; the first that applies wins):
  off track ``|n| > w_side - car_half_width_m`` (``w_left`` when ``n >= 0``, else
  ``w_right``); wrong way ``Δs/dt < wrong_way_speed_m_s`` on every step for at least
  ``wrong_way_time_s``; stall (below).
- **Stall checkpoint.** Each car keeps a checkpoint ``(progress, t)``, set at reset. Whenever
  progress reaches the checkpoint's progress plus ``stall_distance_m`` the checkpoint moves to
  the current ``(progress, t)``. The car stalls when ``stall_time_s`` has passed since its
  checkpoint. So a car is flagged once it has gone ``stall_time_s`` without gaining
  ``stall_distance_m`` of new progress: at most ``stall_time_s`` after it stops, never within
  ``stall_time_s`` of a reset, and never while it keeps averaging more than
  ``stall_distance_m / stall_time_s``.
- **Done cars** are frozen: no field changes until :meth:`ProgressTracker.reset` with their
  mask. ``newly_done`` is True only on the step a car became done.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from f1sim.track.track import Track

GLOBAL_REPROJECT_DS_M = 50.0
"""A wrap-aware ``|Δs|`` above this in one step triggers a global re-projection."""

_TIME_EPS = 1e-9  # slack on accumulated-time comparisons (60 * (1/60) may be 0.999...)


class DoneReason(IntEnum):
    """Why a car stopped. ``TIME_LIMIT`` is set by the World, not by the tracker."""

    NONE = 0
    OFF_TRACK = 1
    WRONG_WAY = 2
    STALL = 3
    TIME_LIMIT = 4


@dataclass(frozen=True)
class ProgressParams:
    """Thresholds for laps and termination."""

    car_half_width_m: float = 1.0  # off track when |n| > w_side - car_half_width_m
    wrong_way_speed_m_s: float = -1.0  # ds/dt below this ...
    wrong_way_time_s: float = 1.0  # ... for this long -> WRONG_WAY
    stall_distance_m: float = 5.0  # less forward progress than this ...
    stall_time_s: float = 5.0  # ... within this window -> STALL
    lap_min_fraction: float = 0.98  # share of the length needed since the last lap start
    window: int = 10  # projection hint window (segments each side)

    def __post_init__(self) -> None:
        if self.car_half_width_m < 0:
            raise ValueError("car_half_width_m must be >= 0")
        if self.wrong_way_time_s <= 0 or self.stall_time_s <= 0:
            raise ValueError("wrong_way_time_s and stall_time_s must be > 0")
        if self.stall_distance_m <= 0:
            raise ValueError("stall_distance_m must be > 0")
        if not 0.0 < self.lap_min_fraction <= 1.0:
            raise ValueError("lap_min_fraction must be in (0, 1]")
        if self.window < 1:
            raise ValueError("window must be >= 1")


_DEFAULT_PARAMS = ProgressParams()


@dataclass
class ProgressState:
    """Per-car state; all arrays have shape ``(N,)``.

    The fields after ``done_reason`` are additions to the contract, kept for the lap, sector,
    wrong-way and stall rules.
    """

    idx: np.ndarray  # int64 segment index (projection hint)
    s: np.ndarray
    n: np.ndarray
    progress: np.ndarray  # unwrapped distance along the track since reset (can go negative)
    lap: np.ndarray  # int64 completed laps
    lap_start_t: np.ndarray
    last_lap_time: np.ndarray  # nan = none yet
    best_lap_time: np.ndarray  # nan = none yet
    sector: np.ndarray  # int64 current sector index, 0 if no sectors
    sector_start_t: np.ndarray
    t: np.ndarray  # seconds since reset
    done: np.ndarray  # bool
    done_reason: np.ndarray  # int8 DoneReason
    progress_at_lap_start: np.ndarray
    progress_at_sector_start: np.ndarray
    wrong_way_t: np.ndarray  # how long ds/dt has been below wrong_way_speed_m_s
    stall_progress: np.ndarray  # stall checkpoint: progress ...
    stall_t: np.ndarray  # ... and time

    @classmethod
    def zeros(cls, n_cars: int) -> ProgressState:
        """Initial state for ``n_cars`` cars (before any reset)."""

        def f():
            return np.zeros(n_cars, dtype=np.float64)

        def i():
            return np.zeros(n_cars, dtype=np.int64)

        return cls(
            idx=i(),
            s=f(),
            n=f(),
            progress=f(),
            lap=i(),
            lap_start_t=f(),
            last_lap_time=np.full(n_cars, np.nan),
            best_lap_time=np.full(n_cars, np.nan),
            sector=i(),
            sector_start_t=f(),
            t=f(),
            done=np.zeros(n_cars, dtype=bool),
            done_reason=np.zeros(n_cars, dtype=np.int8),
            progress_at_lap_start=f(),
            progress_at_sector_start=f(),
            wrong_way_t=f(),
            stall_progress=f(),
            stall_t=f(),
        )


@dataclass(frozen=True)
class ProgressEvents:
    """What happened during one :meth:`ProgressTracker.update`; arrays of shape ``(N,)``."""

    lap_completed: np.ndarray  # bool
    lap_time: np.ndarray  # float, nan where no lap completed this step
    sector_completed: np.ndarray  # int64 index of the sector just finished, -1 if none
    sector_time: np.ndarray  # float, nan where none
    newly_done: np.ndarray  # bool, became done this step

    @classmethod
    def empty(cls, n_cars: int) -> ProgressEvents:
        """No events for ``n_cars`` cars."""
        return cls(
            lap_completed=np.zeros(n_cars, dtype=bool),
            lap_time=np.full(n_cars, np.nan),
            sector_completed=np.full(n_cars, -1, dtype=np.int64),
            sector_time=np.full(n_cars, np.nan),
            newly_done=np.zeros(n_cars, dtype=bool),
        )


def _wrap_ds(d: np.ndarray, length: float) -> np.ndarray:
    """Wrap a difference in ``s`` to ``[-length/2, length/2]``."""
    half = 0.5 * length
    return np.where(d > half, d - length, np.where(d < -half, d + length, d))


class ProgressTracker:
    """Tracks progress, laps, sectors and termination for ``n_cars`` cars on one track.

    Call :meth:`reset` (for all cars) before the first :meth:`update`.
    """

    def __init__(self, track: Track, n_cars: int, params: ProgressParams = _DEFAULT_PARAMS):
        if n_cars < 1:
            raise ValueError("n_cars must be >= 1")
        self.track = track
        self.n_cars = int(n_cars)
        self.params = params
        length = track.length
        sec = tuple(float(v) for v in track.data.sectors_s)
        if len(sec) >= 2:
            starts = np.asarray(sec, dtype=np.float64)
            if abs(starts[0]) > 1e-9 or np.any(np.diff(starts) <= 0) or starts[-1] >= length:
                raise ValueError(
                    f"sectors_s must be (0, s1, ...) strictly increasing below the track "
                    f"length {length}, got {sec}"
                )
            self._sector_starts = starts
        else:
            self._sector_starts = np.zeros(1)
        self._n_sectors = self._sector_starts.size
        self._has_sectors = self._n_sectors > 1
        # End boundary of each sector; the last one is the start line, written as s = length.
        self._bounds = np.append(self._sector_starts[1:], length)
        self._sector_len = self._bounds - self._sector_starts
        self._n_segments = track.data.n_points
        self._pt_s = track.data.s
        self._seg_len = np.diff(np.append(track.data.s, length))
        self._widths = np.vstack([track.data.w_left, track.data.w_right])  # (2, M)
        # A step longer than this (straight-line) can leave the hinted search window.
        self._jump_m = min(GLOBAL_REPROJECT_DS_M, 0.5 * params.window * length / self._n_segments)
        self.state = ProgressState.zeros(self.n_cars)
        self._ready = np.zeros(self.n_cars, dtype=bool)
        self._x = np.zeros(self.n_cars)  # last position, for the jump check
        self._y = np.zeros(self.n_cars)

    # ------------------------------------------------------------------------- helpers

    def _xy(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != (self.n_cars,) or y.shape != (self.n_cars,):
            raise ValueError(f"x and y must have shape ({self.n_cars},), got {x.shape}, {y.shape}")
        return x, y

    def _sector_of(self, s: np.ndarray) -> np.ndarray:
        if not self._has_sectors:
            return np.zeros(s.shape, dtype=np.int64)
        k = np.searchsorted(self._sector_starts, s, side="right") - 1
        return np.clip(k, 0, self._n_sectors - 1).astype(np.int64)

    # -------------------------------------------------------------------------- public

    def reset(self, x, y, mask=None) -> None:
        """Project the masked cars globally and zero their counters (all cars if mask is None)."""
        x, y = self._xy(x, y)
        if mask is None:
            m = np.ones(self.n_cars, dtype=bool)
        else:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.n_cars,):
                raise ValueError(f"mask must have shape ({self.n_cars},), got {m.shape}")
        if not m.any():
            return
        s, n, idx = self.track.project(x[m], y[m])
        st = self.state
        self._x[m] = x[m]
        self._y[m] = y[m]
        st.idx[m] = idx
        st.s[m] = s
        st.n[m] = n
        st.sector[m] = self._sector_of(s)
        for name in (
            "progress",
            "lap_start_t",
            "sector_start_t",
            "t",
            "progress_at_lap_start",
            "progress_at_sector_start",
            "wrong_way_t",
            "stall_progress",
            "stall_t",
        ):
            getattr(st, name)[m] = 0.0
        st.lap[m] = 0
        st.last_lap_time[m] = np.nan
        st.best_lap_time[m] = np.nan
        st.done[m] = False
        st.done_reason[m] = DoneReason.NONE
        self._ready[m] = True

    def update(self, x, y, dt: float) -> ProgressEvents:
        """Advance every car that isn't done to position ``(x, y)`` after ``dt`` seconds."""
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be a positive number, got {dt}")
        if not self._ready.all():
            raise RuntimeError("call reset() for every car before update()")
        x, y = self._xy(x, y)
        p = self.params
        st = self.state
        length = self.track.length
        ev = ProgressEvents.empty(self.n_cars)

        active = ~st.done
        n_active = int(np.count_nonzero(active))
        if n_active == 0:
            return ev
        # All cars active (the usual case): work on views, skipping the gather/scatter copies.
        # Every read of the old state happens before the write-back at the end.
        a = slice(None) if n_active == self.n_cars else np.flatnonzero(active)
        xa, ya = x[a], y[a]

        # --- projection: hinted, with a global fallback for jumps.
        s0 = st.s[a]
        idx0 = st.idx[a]
        s1, n1, i1 = self.track.project(xa, ya, hint=idx0, window=p.window)
        ds = _wrap_ds(s1 - s0, length)
        m_seg = self._n_segments
        di = (i1 - idx0 + m_seg // 2) % m_seg - m_seg // 2
        moved2 = (xa - self._x[a]) ** 2 + (ya - self._y[a]) ** 2
        far = (
            (np.abs(ds) > GLOBAL_REPROJECT_DS_M)
            | (np.abs(di) >= p.window)
            | (moved2 > self._jump_m**2)
        )
        if far.any():
            sg, ng, ig = self.track.project(xa[far], ya[far])
            s1[far], n1[far], i1[far] = sg, ng, ig
            ds[far] = _wrap_ds(sg - s0[far], length)

        t0 = st.t[a]
        t1 = t0 + dt
        prog0 = st.progress[a]
        prog1 = prog0 + ds

        # --- boundary crossings (sector boundaries, then the line as s = length).
        lap = st.lap[a]
        lap_start_t = st.lap_start_t[a]
        last_lap = st.last_lap_time[a]
        best_lap = st.best_lap_time[a]
        p_lap = st.progress_at_lap_start[a]
        sector = st.sector[a]
        sector_start_t = st.sector_start_t[a]
        p_sec = st.progress_at_sector_start[a]
        lap_done = np.zeros(n_active, dtype=bool)
        lap_time = np.full(n_active, np.nan)
        sec_done = np.full(n_active, -1, dtype=np.int64)
        sec_time = np.full(n_active, np.nan)

        b = self._bounds[None, :]  # (1, K)
        fwd = (ds > 0)[:, None]
        wrapped = fwd & (s1 < s0)[:, None]
        s0c, s1c = s0[:, None], s1[:, None]
        crossed = fwd & ((~wrapped & (s0c < b) & (b <= s1c)) | (wrapped & ((b > s0c) | (b <= s1c))))
        if crossed.any():
            rel = np.where(b > s0c, b - s0c, b - s0c + length)  # distance ahead to each boundary
            rel_sort = np.where(crossed, rel, np.inf)
            order = np.argsort(rel_sort, axis=1) if self._n_sectors > 1 else None
            rows = np.arange(n_active)
            safe_ds = np.where(ds > 0, ds, 1.0)
            k_last = self._n_sectors - 1
            for c in range(self._n_sectors):
                j = order[:, c] if order is not None else np.zeros(n_active, dtype=np.int64)
                h = crossed[rows, j]
                if not h.any():
                    break
                rj = rel[rows, j]
                tc = t0 + np.clip(rj / safe_ds, 0.0, 1.0) * dt
                pc = prog0 + rj
                if self._has_sectors:
                    ok = (
                        h & (sector == j) & (pc - p_sec >= p.lap_min_fraction * self._sector_len[j])
                    )
                    sec_done = np.where(ok, j, sec_done)
                    sec_time = np.where(ok, tc - sector_start_t, sec_time)
                    sector = np.where(h, (j + 1) % self._n_sectors, sector)
                    sector_start_t = np.where(h, tc, sector_start_t)
                    p_sec = np.where(h, pc, p_sec)
                line = h & (j == k_last)
                counted = line & (pc - p_lap >= p.lap_min_fraction * length)
                lt = tc - lap_start_t
                lap = lap + counted
                lap_done |= counted
                lap_time = np.where(counted, lt, lap_time)
                last_lap = np.where(counted, lt, last_lap)
                best_lap = np.where(counted, np.fmin(best_lap, lt), best_lap)
                lap_start_t = np.where(line, tc, lap_start_t)
                p_lap = np.where(line, pc, p_lap)

        # --- termination.
        # Width on the car's side at s1, interpolated on segment i1 (= Track.widths_at(s1)).
        u = s1 - self._pt_s[i1]
        u = np.where(u < 0.0, u + length, u) / self._seg_len[i1]
        side = np.where(n1 >= 0.0, 0, 1)
        w0 = self._widths[side, i1]
        w_side = w0 + u * (self._widths[side, (i1 + 1) % m_seg] - w0)
        limit = w_side - p.car_half_width_m
        off = np.abs(n1) > limit
        ww_t = np.where(ds / dt < p.wrong_way_speed_m_s, st.wrong_way_t[a] + dt, 0.0)
        wrong = ww_t >= p.wrong_way_time_s - _TIME_EPS
        stall_p = st.stall_progress[a]
        stall_t = st.stall_t[a]
        advanced = prog1 - stall_p >= p.stall_distance_m
        stall_p = np.where(advanced, prog1, stall_p)
        stall_t = np.where(advanced, t1, stall_t)
        stall = t1 - stall_t >= p.stall_time_s - _TIME_EPS
        reason = np.where(
            off,
            DoneReason.OFF_TRACK,
            np.where(wrong, DoneReason.WRONG_WAY, np.where(stall, DoneReason.STALL, 0)),
        ).astype(np.int8)
        newly = reason != DoneReason.NONE

        # --- write back.
        self._x[a] = xa
        self._y[a] = ya
        st.idx[a] = i1
        st.s[a] = s1
        st.n[a] = n1
        st.progress[a] = prog1
        st.lap[a] = lap
        st.lap_start_t[a] = lap_start_t
        st.last_lap_time[a] = last_lap
        st.best_lap_time[a] = best_lap
        st.sector[a] = sector
        st.sector_start_t[a] = sector_start_t
        st.t[a] = t1
        st.done[a] = newly
        st.done_reason[a] = reason
        st.progress_at_lap_start[a] = p_lap
        st.progress_at_sector_start[a] = p_sec
        st.wrong_way_t[a] = ww_t
        st.stall_progress[a] = stall_p
        st.stall_t[a] = stall_t

        ev.lap_completed[a] = lap_done
        ev.lap_time[a] = lap_time
        ev.sector_completed[a] = sec_done
        ev.sector_time[a] = sec_time
        ev.newly_done[a] = newly
        return ev
