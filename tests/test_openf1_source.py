"""OpenF1 source: offline tests with canned API responses, plus one network check."""

import io
import json
import math
import urllib.error
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from f1sim.data import openf1_source as o

START = datetime(2024, 7, 6, 15, 8, 48, 984000, tzinfo=UTC)
LAP_TIME = 60.0
RADIUS_DM = 3000.0  # 300 m circle, positions in 1/10 m like the live-timing feed


def _location_samples():
    """Positions every 0.27 s from 2 s before the lap to 2 s after, at 1 lap per LAP_TIME."""
    out = []
    for t in np.arange(-2.0, LAP_TIME + 2.0, 0.27):
        ang = 2 * np.pi * t / LAP_TIME
        out.append(
            {
                "date": (START + timedelta(seconds=float(t))).isoformat(),
                "x": RADIUS_DM * np.cos(ang),
                "y": RADIUS_DM * np.sin(ang),
            }
        )
    return out


LAPS = [
    {"driver_number": 63, "lap_duration": None, "is_pit_out_lap": True, "date_start": "x"},
    {
        "driver_number": 63,
        "lap_duration": 61.0,
        "is_pit_out_lap": False,
        "date_start": START.isoformat(),
        "duration_sector_1": 20.0,
        "duration_sector_2": 21.0,
    },
    {
        "driver_number": 63,
        "lap_duration": LAP_TIME,
        "is_pit_out_lap": False,
        "date_start": START.isoformat(),
        "duration_sector_1": 19.5,
        "duration_sector_2": 20.5,
    },
    {"driver_number": 44, "lap_duration": 59.0, "is_pit_out_lap": True, "date_start": "x"},
    {
        "driver_number": 44,
        "lap_duration": 60.5,
        "is_pit_out_lap": False,
        "date_start": START.isoformat(),
        "duration_sector_1": 20.0,
        "duration_sector_2": None,
    },
]
DRIVERS = [
    {"driver_number": 63, "name_acronym": "RUS"},
    {"driver_number": 44, "name_acronym": "HAM"},
]


@pytest.fixture
def canned(monkeypatch):
    responses = {"drivers": DRIVERS, "laps": LAPS, "location": _location_samples()}
    calls = []

    def fake_get(endpoint, params, filters=(), max_age_s=None):
        calls.append((endpoint, params, filters))
        return responses[endpoint]

    monkeypatch.setattr(o, "_get", fake_get)
    return calls


def test_fastest_lap_skips_pit_out_laps_and_converts_units(canned):
    lap = o.fastest_lap_xy({"session_key": 9554})
    assert lap.driver == "RUS"  # 59.0 s is a pit out-lap, so 60.0 s wins
    assert lap.lap_time_s == LAP_TIME
    assert lap.t[0] == 0.0 and lap.t[-1] == pytest.approx(LAP_TIME)
    assert np.all(np.diff(lap.t) > 0)
    r = np.hypot(lap.x_m, lap.y_m)
    assert r == pytest.approx(300.0, abs=0.5)  # dm -> m (chords cut the circle slightly)
    assert lap.sector_end_t == (19.5, 40.0)
    # The position query is bounded around the lap with unencoded OpenF1 date filters
    endpoint, params, filters = canned[-1]
    assert endpoint == "location" and params["driver_number"] == 63
    assert filters[0].startswith("date>=2024-07-06T15:08:46") and filters[1].startswith("date<=")


def test_fastest_lap_for_one_driver_by_acronym_or_number(canned):
    assert o.fastest_lap_xy({"session_key": 1}, "ham").lap_time_s == 60.5
    lap = o.fastest_lap_xy({"session_key": 1}, 44)
    assert lap.driver == "HAM"
    assert lap.sector_end_t[0] == 20.0 and math.isnan(lap.sector_end_t[1])


def test_unknown_driver_lists_known_ones(canned):
    with pytest.raises(o.OpenF1Error, match="HAM, RUS"):
        o.fastest_lap_xy({"session_key": 1}, "VER")


def test_find_session_matches_catalogue_location(monkeypatch):
    sessions = [
        {"session_key": 1, "location": "Imola", "is_cancelled": False},
        {"session_key": 2, "location": "Silverstone", "is_cancelled": True},
        {"session_key": 3, "location": "Silverstone", "is_cancelled": False},
    ]
    monkeypatch.setattr(o, "_get", lambda *a, **k: sessions)
    assert o.find_session(2024, "silverstone")["session_key"] == 3
    with pytest.raises(o.OpenF1Error, match="Locations: Imola, Silverstone"):
        o.find_session(2024, "monza")


def test_find_session_refuses_years_before_openf1():
    with pytest.raises(o.OpenF1Error, match="before 2023"):
        o.find_session(2022, "silverstone")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_caches_responses_and_maps_404_to_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(o, "MIN_REQUEST_INTERVAL_S", 0.0)
    hits = []

    def fake_urlopen(url, timeout):
        hits.append(url)
        if "missing" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _Response(json.dumps([{"ok": 1}]).encode())

    monkeypatch.setattr(o.urllib.request, "urlopen", fake_urlopen)
    assert o._get("laps", {"session_key": 7}) == [{"ok": 1}]
    assert o._get("laps", {"session_key": 7}) == [{"ok": 1}]  # served from the cache
    assert len(hits) == 1
    assert hits[0] == "https://api.openf1.org/v1/laps?session_key=7"
    assert o._get("missing", {}) == []


@pytest.mark.network
def test_silverstone_2024_matches_fastf1_positions():
    from f1sim.data import fastf1_source

    lap = o.fastest_lap_xy(o.find_session(2024, "silverstone"), "RUS")
    assert lap.lap_time_s == pytest.approx(85.819)
    assert lap.sector_end_t == pytest.approx((28.016, 62.524))
    ff = fastf1_source.fastest_lap_xy(fastf1_source.load_session(2024, "Silverstone", "Q"), "RUS")
    # Same raw positions; the lap starts differ by up to ~0.2 s, so compare shapes, not clocks
    best = min(
        np.median(
            np.hypot(
                np.interp(ff.t + k, lap.t, lap.x_m) - ff.x_m,
                np.interp(ff.t + k, lap.t, lap.y_m) - ff.y_m,
            )
        )
        for k in np.arange(-0.3, 0.3, 0.005)
    )
    assert best < 0.5
