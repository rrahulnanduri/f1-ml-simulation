"""FastF1 access: event schedules, sessions and fastest-lap paths.

This is the only f1sim module that imports ``fastf1``. FastF1's cache lives in
``paths.fastf1_dir()`` and is enabled lazily on first use.

FastF1 position data is in 1/10 m in FastF1's own frame; ``fastest_lap_xy`` converts it to
metres but does not change the frame (``alignment.py`` does that).
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from f1sim import paths
from f1sim.data.laps import CarTrace, LapXY

__all__ = [
    "CarTrace",
    "EventInfo",
    "LapXY",
    "event_schedule",
    "fastest_lap_car_data",
    "fastest_lap_xy",
    "load_session",
]

_cache_enabled = False


def _ensure_cache() -> None:
    """Point FastF1 at ``paths.fastf1_dir()``, once per process."""
    global _cache_enabled
    if not _cache_enabled:
        fastf1.Cache.enable_cache(str(paths.fastf1_dir()))
        _cache_enabled = True


@dataclass(frozen=True)
class EventInfo:
    """One championship round (testing excluded)."""

    year: int
    round: int
    name: str
    location: str
    country: str


# ---------------------------------------------------------------- pure helpers (offline tests)


def _seconds(value: Any) -> float:
    """Timedelta-like or number -> float seconds; NaT/NaN/None -> nan."""
    if value is None or pd.isna(value):
        return math.nan
    if isinstance(value, pd.Timedelta | np.timedelta64):
        return pd.Timedelta(value).total_seconds()
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def _norm_location(location: str) -> str:
    decomposed = unicodedata.normalize("NFKD", location.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _driver_identifier(driver: str | int) -> str:
    """Normalise a driver abbreviation ('ham' -> 'HAM') or number (44 / '44' -> '44')."""
    ident = str(driver).strip()
    if not ident:
        raise ValueError("Empty driver identifier")
    return ident if ident.isdigit() else ident.upper()


def _clean_xy(
    t_s: np.ndarray, x_dm: np.ndarray, y_dm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by time, drop non-finite samples and duplicate timestamps, convert dm -> m."""
    t = np.asarray(t_s, dtype=np.float64)
    x = np.asarray(x_dm, dtype=np.float64)
    y = np.asarray(y_dm, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0  # keep the first sample of each timestamp
    return t[keep], x[keep] / 10.0, y[keep] / 10.0


def _sector_end_times(
    s1: Any, s2: Any, s1_session: Any = None, s2_session: Any = None, lap_start: Any = None
) -> tuple[float, float]:
    """End of S1 and S2 in seconds from lap start.

    Uses Sector1Time and Sector1Time + Sector2Time; where those are missing, falls back to
    the Sector{1,2}SessionTime minus LapStartTime. Unknown values come back as nan.
    """
    t1 = _seconds(s1)
    t2 = t1 + _seconds(s2)
    start = _seconds(lap_start)
    if math.isnan(t1):
        t1 = _seconds(s1_session) - start
    if math.isnan(t2):
        t2 = _seconds(s2_session) - start
    return t1, t2


# ---------------------------------------------------------------- FastF1 calls


def event_schedule(year: int) -> list[EventInfo]:
    """Championship rounds of ``year`` in order (pre-season testing, round 0, excluded)."""
    _ensure_cache()
    sched = fastf1.get_event_schedule(year, include_testing=False)
    events = []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd <= 0:
            continue
        events.append(
            EventInfo(
                year=year,
                round=rnd,
                name=str(ev["EventName"]),
                location=str(ev["Location"]),
                country=str(ev["Country"]),
            )
        )
    return events


def load_session(year: int, location: str, kind: str = "Q"):
    """Load laps and telemetry (not weather or race-control messages) for one session.

    ``location`` is a FastF1 event ``Location`` such as ``"Silverstone"`` (matched ignoring
    case and accents); ``kind`` is a FastF1 session identifier ("Q", "R", "FP1", "S", ...).
    Returns the loaded ``fastf1.core.Session``.
    """
    events = event_schedule(year)
    key = _norm_location(location)
    matches = [e for e in events if _norm_location(e.location) == key]
    if not matches:
        known = ", ".join(e.location for e in events)
        raise ValueError(f"No {year} event at location {location!r}. Locations: {known}")
    _ensure_cache()
    session = fastf1.get_session(year, matches[0].round, kind)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    # FastF1 logs download failures as warnings and leaves the data unset; fail loudly here.
    try:
        n_laps = len(session.laps)
        session.pos_data  # noqa: B018 (raises if position data was not loaded)
    except Exception as e:
        raise RuntimeError(
            f"FastF1 could not load laps/position data for {year} {location} {kind}: {e}. "
            "Check the network. The F1 live-timing archive now answers 401 without "
            "authentication, which FastF1 supports from 3.9; data already in the cache "
            f"({paths.fastf1_dir()}) still loads."
        ) from e
    if n_laps == 0:
        raise RuntimeError(f"FastF1 returned no laps for {year} {location} {kind}")
    return session


def fastest_lap_xy(session, driver: str | int | None = None) -> LapXY:
    """Position trace of the fastest valid lap, overall or for one driver.

    ``driver`` is an abbreviation ("HAM") or a car number (44 or "44"); None means the
    session's overall fastest lap. Laps with deleted times are not considered.
    """
    laps = session.laps
    if driver is None:
        lap = laps.pick_fastest()
        if lap is None:
            raise ValueError(f"No valid timed lap in session {session}")
    else:
        ident = _driver_identifier(driver)
        drv_laps = laps.pick_drivers(ident)
        if len(drv_laps) == 0:
            known = ", ".join(sorted(set(laps["Driver"].dropna().astype(str))))
            raise ValueError(f"Driver {driver!r} has no laps in {session}. Drivers: {known}")
        lap = drv_laps.pick_fastest()
        if lap is None:
            raise ValueError(f"Driver {driver!r} has no valid timed lap in {session}")

    lap_time_s = _seconds(lap["LapTime"])
    if not math.isfinite(lap_time_s):
        raise ValueError(f"Fastest lap of {lap['Driver']} has no lap time")

    pos = lap.get_pos_data(interpolate_edges=True)
    if pos.empty:
        raise ValueError(f"No position data for {lap['Driver']}'s fastest lap")
    lap_start = lap["LapStartTime"]
    t_rel = (pos["SessionTime"] - lap_start).dt.total_seconds().to_numpy()
    t, x_m, y_m = _clean_xy(t_rel, pos["X"].to_numpy(), pos["Y"].to_numpy())
    if t.size < 10:
        raise ValueError(f"Too few position samples ({t.size}) for {lap['Driver']}'s lap")

    sector_end_t = _sector_end_times(
        lap["Sector1Time"],
        lap["Sector2Time"],
        lap.get("Sector1SessionTime"),
        lap.get("Sector2SessionTime"),
        lap_start,
    )
    return LapXY(
        driver=str(lap["Driver"]),
        lap_time_s=lap_time_s,
        t=t,
        x_m=x_m,
        y_m=y_m,
        sector_end_t=sector_end_t,
    )


# ---------------------------------------------------------------- car data (speed, pedals, gear)


def _pick_lap(session, driver: str | int | None):
    """The lap ``fastest_lap_xy`` uses: fastest valid lap, overall or for one driver."""
    laps = session.laps
    if driver is None:
        lap = laps.pick_fastest()
        if lap is None:
            raise ValueError(f"No valid timed lap in session {session}")
        return lap
    ident = _driver_identifier(driver)
    drv_laps = laps.pick_drivers(ident)
    if len(drv_laps) == 0:
        known = ", ".join(sorted(set(laps["Driver"].dropna().astype(str))))
        raise ValueError(f"Driver {driver!r} has no laps in {session}. Drivers: {known}")
    lap = drv_laps.pick_fastest()
    if lap is None:
        raise ValueError(f"Driver {driver!r} has no valid timed lap in {session}")
    return lap


def _car_trace(
    t_s: Any, speed_kmh: Any, throttle: Any, brake: Any, gear: Any, driver: str, team: str
) -> CarTrace:
    """Car-data columns -> a ``CarTrace``: sorted, non-finite rows and duplicate times dropped.

    ``speed_kmh`` is km/h, ``throttle`` 0-100 and ``brake`` a switch (bool or 0/100).
    """
    cols = [np.asarray(c, dtype=np.float64) for c in (t_s, speed_kmh, throttle, brake, gear)]
    ok = np.logical_and.reduce([np.isfinite(c) for c in cols])
    t, speed, thr, brk, gr = (c[ok] for c in cols)
    order = np.argsort(t, kind="stable")
    t, speed, thr, brk, gr = (c[order] for c in (t, speed, thr, brk, gr))
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0
    return CarTrace(
        t=t[keep],
        speed_ms=speed[keep] / 3.6,
        throttle01=np.clip(thr[keep] / 100.0, 0.0, 1.0),
        brake01=(brk[keep] > 0).astype(np.float64),
        gear=gr[keep].astype(np.int64),
        driver=driver,
        team=team,
    )


def fastest_lap_car_data(session, driver: str | int | None = None) -> CarTrace:
    """Car data (``lap.get_car_data()``) of the lap ``fastest_lap_xy`` picks.

    ``t`` runs from the lap's ``LapStartTime``; one sample either side of the lap is kept
    (``pad=1``) so the lap's ends can be interpolated.
    """
    lap = _pick_lap(session, driver)
    car = lap.get_car_data(pad=1, pad_side="both")
    if car.empty:
        raise ValueError(f"No car data for {lap['Driver']}'s fastest lap")
    t_rel = (car["SessionTime"] - lap["LapStartTime"]).dt.total_seconds().to_numpy()
    team = lap.get("Team")
    trace = _car_trace(
        t_rel,
        car["Speed"].to_numpy(),
        car["Throttle"].to_numpy(),
        car["Brake"].to_numpy(),
        car["nGear"].to_numpy(),
        driver=str(lap["Driver"]),
        team="" if team is None or pd.isna(team) else str(team),
    )
    if trace.t.size < 10:
        raise ValueError(f"Too few car-data samples ({trace.t.size}) for {lap['Driver']}'s lap")
    return trace
