"""Reference laps from whichever source can serve them.

F1's live-timing archive needs an F1 TV subscription since October 2026, so:
- 2023 onwards: OpenF1 (free, no sign-in, same position data as the archive).
- Before 2023: FastF1, which only works for sessions already in its cache.

Both sources return the same ``LapXY``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPENF1_FIRST_YEAR = 2023
SOURCES = ("auto", "openf1", "fastf1")


@dataclass(frozen=True)
class LapXY:
    """A lap's position trace in metres (the source's raw frame), time from lap start."""

    driver: str
    lap_time_s: float
    t: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    sector_end_t: tuple[float, float]


def resolve_source(year: int, source: str = "auto") -> str:
    """The concrete source for ``year``: "openf1" or "fastf1"."""
    if source not in SOURCES:
        raise ValueError(f"Unknown lap source {source!r}; use one of {', '.join(SOURCES)}")
    if source == "auto":
        return "openf1" if year >= OPENF1_FIRST_YEAR else "fastf1"
    if source == "openf1" and year < OPENF1_FIRST_YEAR:
        raise ValueError(f"OpenF1 has no data before {OPENF1_FIRST_YEAR} (asked for {year})")
    return source


def reference_lap(
    track_id: str,
    year: int,
    *,
    kind: str = "Q",
    driver: str | int | None = None,
    source: str = "auto",
) -> tuple[LapXY, str]:
    """Fastest lap of ``track_id``'s ``kind`` session in ``year`` (overall, or ``driver``'s).

    Returns the lap and the source that served it ("openf1" or "fastf1").
    """
    from f1sim.data import catalog

    entry = catalog.get(track_id)
    chosen = resolve_source(year, source)
    if chosen == "openf1":
        from f1sim.data import openf1_source

        session = openf1_source.find_session(year, entry.id, kind)
        return openf1_source.fastest_lap_xy(session, driver), chosen

    from f1sim.data import fastf1_source  # imports fastf1, which is slow

    session = fastf1_source.load_session(year, entry.fastf1_locations[0], kind)
    return fastf1_source.fastest_lap_xy(session, driver), chosen
