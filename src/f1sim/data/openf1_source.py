"""OpenF1 access (https://openf1.org): sessions and fastest-lap position traces, 2023 onwards.

Historical data is free and needs no sign-in. The free tier allows 30 requests a minute, so
requests are spaced out and every response is cached in ``paths.openf1_dir()``. OpenF1 is
unofficial and licensed CC BY-NC-SA 4.0 (non-commercial use).

Positions are the F1 live-timing X/Y in 1/10 m at about 3.7 Hz, the same raw data FastF1 reads,
so they share FastF1's frame. Shifted onto FastF1's lap start, the traces agree to about 0.03 m.
The two sources mark the lap start differently, though: OpenF1's ``date_start`` is 0 to 0.18 s
later than FastF1's, varying from lap to lap (measured on 7 laps at Silverstone and Imola 2024).
No constant correction fits and the positions alone can't fix it (on a flying lap the start and
end of the lap overlap for any shift), so ``date_start`` is used as is.

Unlike FastF1, OpenF1 doesn't flag laps deleted for track limits; ``fastest_lap_xy`` picks the
fastest timed lap that isn't a pit out-lap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from f1sim import paths
from f1sim.data.laps import OPENF1_FIRST_YEAR, CarTrace, LapXY

BASE_URL = "https://api.openf1.org/v1"
SESSION_NAMES = {"Q": "Qualifying", "R": "Race", "S": "Sprint", "SQ": "Sprint Qualifying"}
MIN_REQUEST_INTERVAL_S = 2.1  # free tier: 30 requests/minute
MAX_RESPONSE_BYTES = 256 * 1024 * 1024  # a whole session's car data is tens of MB
SESSION_LIST_MAX_AGE_S = 24 * 3600  # schedules change during a season; lap data doesn't
_EDGE_PAD_S = 2.0

_last_request = 0.0


def _reject_constant(name: str) -> float:
    """``json.loads`` hook: OpenF1 never sends NaN or Infinity, so refuse them at the door."""
    raise ValueError(f"non-finite number {name} in an OpenF1 response")


class OpenF1Error(RuntimeError):
    """OpenF1 request failed or returned no usable data."""


# ---------------------------------------------------------------- HTTP + cache


def _url(endpoint: str, params: dict[str, Any], filters: tuple[str, ...] = ()) -> str:
    # OpenF1 filters look like "date>=2024-07-06T15:08:48"; keep the operators unencoded
    parts = [f"{k}={v}" for k, v in params.items()] + list(filters)
    return f"{BASE_URL}/{endpoint}" + ("?" + "&".join(parts) if parts else "")


def _cache_path(url: str) -> Path:
    return paths.openf1_dir() / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")


def _get(
    endpoint: str,
    params: dict[str, Any],
    filters: tuple[str, ...] = (),
    max_age_s: float | None = None,
) -> list[dict]:
    """GET an OpenF1 endpoint, from the cache when possible. "No results" returns []."""
    global _last_request
    url = _url(endpoint, params, filters)
    cached = _cache_path(url)
    if cached.exists() and (max_age_s is None or time.time() - cached.stat().st_mtime < max_age_s):
        return json.loads(cached.read_text(encoding="utf-8"))

    for attempt in range(4):
        wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise OpenF1Error(f"OpenF1 response over {MAX_RESPONSE_BYTES} bytes for {url}")
            try:
                data = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
            except ValueError as e:  # JSONDecodeError and UnicodeDecodeError included
                raise OpenF1Error(f"Unreadable OpenF1 response for {url}: {e}") from e
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 {"detail": "No results found."}
                data = []
                break
            if e.code == 429 and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise OpenF1Error(f"OpenF1 {e.code} for {url}") from e
        except urllib.error.URLError as e:
            raise OpenF1Error(f"Can't reach OpenF1 ({e.reason}) for {url}") from e
    if isinstance(data, dict):  # error payloads are objects, results are lists
        data = []

    fd, tmp = tempfile.mkstemp(dir=cached.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, cached)
    return data


# ---------------------------------------------------------------- pure helpers (offline tests)


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _lap_trace(
    t: np.ndarray, x_dm: np.ndarray, y_dm: np.ndarray, lap_time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut [0, lap_time] from samples timed from the lap start, add interpolated edges, dm -> m."""
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x_dm[order] / 10.0, y_dm[order] / 10.0
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x, y = t[keep], x[keep], y[keep]
    inner = (t > 0) & (t < lap_time)
    edges = np.array([0.0, lap_time])
    tt = np.concatenate([edges[:1], t[inner], edges[1:]])
    return (
        tt,
        np.concatenate([np.interp(edges[:1], t, x), x[inner], np.interp(edges[1:], t, x)]),
        np.concatenate([np.interp(edges[:1], t, y), y[inner], np.interp(edges[1:], t, y)]),
    )


def _pick_fastest(laps: list[dict], driver_number: int | None) -> dict:
    timed = [
        lap
        for lap in laps
        if lap.get("lap_duration")
        and not lap.get("is_pit_out_lap")
        and (driver_number is None or lap["driver_number"] == driver_number)
    ]
    if not timed:
        who = f"driver {driver_number}" if driver_number is not None else "any driver"
        raise OpenF1Error(f"No timed laps for {who}")
    return min(timed, key=lambda lap: lap["lap_duration"])


def _driver_number(drivers: list[dict], driver: str | int) -> int:
    ident = str(driver).strip()
    if ident.isdigit():
        return int(ident)
    for d in drivers:
        if str(d.get("name_acronym", "")).upper() == ident.upper():
            return int(d["driver_number"])
    known = ", ".join(sorted({str(d.get("name_acronym")) for d in drivers}))
    raise OpenF1Error(f"Unknown driver {driver!r}. Drivers: {known}")


# ---------------------------------------------------------------- OpenF1 calls


def find_session(year: int, track_id: str, kind: str = "Q") -> dict:
    """The OpenF1 session record for ``track_id``'s ``kind`` session in ``year``."""
    from f1sim.data import catalog

    if year < OPENF1_FIRST_YEAR:
        raise OpenF1Error(f"OpenF1 has no data before {OPENF1_FIRST_YEAR} (asked for {year})")
    name = SESSION_NAMES.get(kind, kind)
    sessions = _get(
        "sessions",
        {"year": year, "session_name": name.replace(" ", "%20")},
        max_age_s=SESSION_LIST_MAX_AGE_S,
    )
    matches = [
        s
        for s in sessions
        if (e := catalog.entry_for_location(s.get("location", "")))
        and e.id == track_id
        and not s.get("is_cancelled")
    ]
    if not matches:
        seen = ", ".join(sorted({s.get("location", "?") for s in sessions}))
        raise OpenF1Error(f"No {year} {name} at {track_id} in OpenF1. Locations: {seen}")
    return matches[0]


def fastest_lap_xy(session: dict, driver: str | int | None = None) -> LapXY:
    """Position trace of the session's fastest lap, overall or for one driver."""
    key = session["session_key"]
    drivers = _get("drivers", {"session_key": key})
    number = None if driver is None else _driver_number(drivers, driver)
    lap = _pick_fastest(_get("laps", {"session_key": key}), number)
    number = int(lap["driver_number"])
    lap_time = float(lap["lap_duration"])

    start = _parse_time(lap["date_start"])
    pad = timedelta(seconds=_EDGE_PAD_S)
    lo = (start - pad).isoformat(timespec="milliseconds").replace("+00:00", "")
    hi = (start + timedelta(seconds=lap_time) + pad).isoformat(timespec="milliseconds")
    hi = hi.replace("+00:00", "")
    pos = _get(
        "location",
        {"session_key": key, "driver_number": number},
        filters=(f"date>={lo}", f"date<={hi}"),
    )
    if len(pos) < 10:
        raise OpenF1Error(f"Too few position samples ({len(pos)}) for car {number}'s lap")
    t = np.array([(_parse_time(p["date"]) - start).total_seconds() for p in pos])
    x = np.array([p["x"] for p in pos], dtype=np.float64)
    y = np.array([p["y"] for p in pos], dtype=np.float64)
    tt, xm, ym = _lap_trace(t, x, y, lap_time)

    s1, s2 = lap.get("duration_sector_1"), lap.get("duration_sector_2")
    sector_end_t = (
        float(s1) if s1 is not None else math.nan,
        float(s1) + float(s2) if s1 is not None and s2 is not None else math.nan,
    )
    acronym = next(
        (str(d["name_acronym"]) for d in drivers if int(d["driver_number"]) == number),
        str(number),
    )
    return LapXY(
        driver=acronym, lap_time_s=lap_time, t=tt, x_m=xm, y_m=ym, sector_end_t=sector_end_t
    )


# ---------------------------------------------------------------- car data (speed, pedals, gear)


def _car_trace(samples: list[dict], start: datetime, driver: str, team: str) -> CarTrace:
    """OpenF1 ``car_data`` records -> a ``CarTrace`` timed from ``start``, sorted, deduplicated.

    ``speed`` is km/h, ``throttle`` 0-100 and ``brake`` 0/100 (any value above 0 counts as on).
    Records with a missing channel are dropped.
    """
    keys = ("speed", "throttle", "brake", "n_gear")
    rows = [p for p in samples if all(p.get(k) is not None for k in keys)]
    t = np.array([(_parse_time(p["date"]) - start).total_seconds() for p in rows], dtype=float)
    cols = {k: np.array([p[k] for p in rows], dtype=np.float64) for k in keys}
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0  # keep the first sample of each timestamp
    t = t[keep]
    cols = {k: v[order][keep] for k, v in cols.items()}
    return CarTrace(
        t=t,
        speed_ms=cols["speed"] / 3.6,
        throttle01=np.clip(cols["throttle"] / 100.0, 0.0, 1.0),
        brake01=(cols["brake"] > 0).astype(np.float64),
        gear=cols["n_gear"].astype(np.int64),
        driver=driver,
        team=team,
    )


def fastest_lap_car_data(session: dict, driver: str | int | None = None) -> CarTrace:
    """Car data of the lap :func:`fastest_lap_xy` picks, timed from that lap's ``date_start``.

    Samples run from 2 s before the lap to 2 s after it, the same window as the positions.
    The team comes from the ``drivers`` endpoint's ``team_name`` (``""`` if missing).
    """
    key = session["session_key"]
    drivers = _get("drivers", {"session_key": key})
    number = None if driver is None else _driver_number(drivers, driver)
    lap = _pick_fastest(_get("laps", {"session_key": key}), number)
    number = int(lap["driver_number"])
    lap_time = float(lap["lap_duration"])

    start = _parse_time(lap["date_start"])
    pad = timedelta(seconds=_EDGE_PAD_S)
    lo = (start - pad).isoformat(timespec="milliseconds").replace("+00:00", "")
    hi = (start + timedelta(seconds=lap_time) + pad).isoformat(timespec="milliseconds")
    hi = hi.replace("+00:00", "")
    samples = _get(
        "car_data",
        {"session_key": key, "driver_number": number},
        filters=(f"date>={lo}", f"date<={hi}"),
    )
    record = next((d for d in drivers if int(d["driver_number"]) == number), {})
    trace = _car_trace(
        samples,
        start,
        driver=str(record.get("name_acronym") or number),
        team=str(record.get("team_name") or ""),
    )
    if trace.t.size < 10:
        raise OpenF1Error(f"Too few car-data samples ({trace.t.size}) for car {number}'s lap")
    return trace
