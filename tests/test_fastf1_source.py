import math

import numpy as np
import pandas as pd
import pytest

from f1sim import paths
from f1sim.data import fastf1_source as fs

td = pd.Timedelta


# ---------------------------------------------------------------- pure helpers


def test_seconds():
    assert fs._seconds(td(seconds=85.819)) == pytest.approx(85.819)
    assert fs._seconds(np.timedelta64(1500, "ms")) == pytest.approx(1.5)
    assert fs._seconds(2) == 2.0
    for missing in (pd.NaT, None, float("nan"), np.nan):
        assert math.isnan(fs._seconds(missing))


def test_driver_identifier():
    assert fs._driver_identifier("ham") == "HAM"
    assert fs._driver_identifier(" Rus ") == "RUS"
    assert fs._driver_identifier(44) == "44"
    assert fs._driver_identifier("1") == "1"
    with pytest.raises(ValueError):
        fs._driver_identifier("  ")


def test_clean_xy_sorts_dedupes_and_converts():
    t = np.array([0.0, 0.5, 0.25, 0.5, 1.0, np.nan, 1.5])
    x = np.array([0.0, 50.0, 25.0, 999.0, 100.0, 1.0, np.nan])
    y = np.array([10.0, 20.0, 15.0, 999.0, 30.0, 1.0, 40.0])
    tt, xm, ym = fs._clean_xy(t, x, y)
    assert tt.tolist() == [0.0, 0.25, 0.5, 1.0]
    assert xm.tolist() == [0.0, 2.5, 5.0, 10.0]  # decimetres -> metres; first duplicate kept
    assert ym.tolist() == [1.0, 1.5, 2.0, 3.0]
    assert np.all(np.diff(tt) > 0)
    assert tt.dtype == xm.dtype == ym.dtype == np.float64


def test_sector_end_times():
    assert fs._sector_end_times(td(seconds=28.016), td(seconds=34.508)) == pytest.approx(
        (28.016, 62.524)
    )
    # Missing sector times fall back to session times minus the lap start
    start = td(seconds=1000)
    got = fs._sector_end_times(pd.NaT, pd.NaT, td(seconds=1028), td(seconds=1062.5), start)
    assert got == pytest.approx((28.0, 62.5))
    got = fs._sector_end_times(td(seconds=28), pd.NaT, None, td(seconds=1062.5), start)
    assert got == pytest.approx((28.0, 62.5))
    assert all(math.isnan(v) for v in fs._sector_end_times(pd.NaT, pd.NaT))


def test_cache_enabled_once_lazily(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    calls = []
    monkeypatch.setattr(fs.fastf1.Cache, "enable_cache", lambda d: calls.append(d))
    monkeypatch.setattr(fs, "_cache_enabled", False)
    fs._ensure_cache()
    fs._ensure_cache()
    assert calls == [str(tmp_path / "fastf1")]


# ---------------------------------------------------------------- with FastF1 stubbed out


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(fs, "_ensure_cache", lambda: None)


def _schedule_df():
    return pd.DataFrame(
        {
            "RoundNumber": [0, 1, 2],
            "EventName": ["Pre-Season Testing", "Bahrain Grand Prix", "Canadian Grand Prix"],
            "Location": ["Sakhir", "Sakhir", "Montréal"],
            "Country": ["Bahrain", "Bahrain", "Canada"],
        }
    )


def test_event_schedule_excludes_testing(no_cache, monkeypatch):
    seen = {}

    def fake_schedule(year, include_testing=True):
        seen["include_testing"] = include_testing
        return _schedule_df()

    monkeypatch.setattr(fs.fastf1, "get_event_schedule", fake_schedule)
    events = fs.event_schedule(2024)
    assert seen["include_testing"] is False
    assert events == [
        fs.EventInfo(2024, 1, "Bahrain Grand Prix", "Sakhir", "Bahrain"),
        fs.EventInfo(2024, 2, "Canadian Grand Prix", "Montréal", "Canada"),
    ]


class FakeSession:
    def __init__(self, ok=True):
        self.ok = ok
        self.load_kwargs = None

    def load(self, **kwargs):
        self.load_kwargs = kwargs

    @property
    def laps(self):
        if not self.ok:
            raise RuntimeError("The data you are trying to access has not been loaded yet")
        return [1, 2, 3]

    @property
    def pos_data(self):
        return {}


def test_load_session_matches_location(no_cache, monkeypatch):
    monkeypatch.setattr(fs.fastf1, "get_event_schedule", lambda y, include_testing: _schedule_df())
    got = {}

    def fake_get_session(year, rnd, kind):
        got.update(year=year, rnd=rnd, kind=kind)
        got["session"] = FakeSession()
        return got["session"]

    monkeypatch.setattr(fs.fastf1, "get_session", fake_get_session)
    s = fs.load_session(2024, " montreal ", "Q")
    assert (got["year"], got["rnd"], got["kind"]) == (2024, 2, "Q")
    assert s is got["session"]
    assert s.load_kwargs == dict(laps=True, telemetry=True, weather=False, messages=False)

    with pytest.raises(ValueError, match="Locations: Sakhir"):
        fs.load_session(2024, "Monaco")


def test_load_session_reports_missing_data(no_cache, monkeypatch):
    monkeypatch.setattr(fs.fastf1, "get_event_schedule", lambda y, include_testing: _schedule_df())
    monkeypatch.setattr(fs.fastf1, "get_session", lambda y, r, k: FakeSession(ok=False))
    with pytest.raises(RuntimeError, match="could not load"):
        fs.load_session(2024, "Sakhir")


class FakeLap(pd.Series):
    """Enough of fastf1.core.Lap for fastest_lap_xy."""

    _metadata = ["pos"]

    @property
    def _constructor(self):
        return FakeLap

    def get_pos_data(self, **kwargs):
        assert kwargs == {"interpolate_edges": True}
        return self.pos


def _lap(driver, number, lap_time, start=1000.0, n=50):
    lap = FakeLap(
        {
            "Driver": driver,
            "DriverNumber": number,
            "LapTime": td(seconds=lap_time),
            "LapStartTime": td(seconds=start),
            "Sector1Time": td(seconds=lap_time * 0.3),
            "Sector2Time": td(seconds=lap_time * 0.4),
            "Sector1SessionTime": td(seconds=start + lap_time * 0.3),
            "Sector2SessionTime": td(seconds=start + lap_time * 0.7),
        }
    )
    t = np.linspace(0.0, lap_time, n)
    ang = np.linspace(0.0, 2 * np.pi, n)
    session_time = pd.to_timedelta(start + t, unit="s")
    pos = pd.DataFrame(
        {"SessionTime": session_time, "X": 10000 * np.cos(ang), "Y": 10000 * np.sin(ang)}
    )
    lap.pos = pd.concat([pos, pos.iloc[[10]]]).sort_values("SessionTime")  # one duplicate
    return lap


class FakeLaps:
    def __init__(self, laps):
        self.laps = laps

    def __len__(self):
        return len(self.laps)

    def __getitem__(self, column):
        return pd.Series([lap[column] for lap in self.laps], dtype=object)

    def pick_drivers(self, ident):
        return FakeLaps([lap for lap in self.laps if ident in (lap["Driver"], lap["DriverNumber"])])

    def pick_fastest(self):
        timed = [lap for lap in self.laps if not pd.isna(lap["LapTime"])]
        return min(timed, key=lambda lap: lap["LapTime"]) if timed else None


class FakeLoadedSession:
    def __init__(self, laps):
        self.laps = FakeLaps(laps)

    def __str__(self):
        return "Fake Q"


@pytest.fixture
def session():
    return FakeLoadedSession([_lap("RUS", "63", 85.819), _lap("HAM", "44", 85.99)])


def test_fastest_lap_overall(session):
    lap = fs.fastest_lap_xy(session)
    assert lap.driver == "RUS"
    assert lap.lap_time_s == pytest.approx(85.819)
    assert lap.t.size == 50  # duplicate timestamp dropped
    assert lap.t[0] == pytest.approx(0.0)  # relative to lap start
    assert lap.t[-1] == pytest.approx(85.819)
    assert np.all(np.diff(lap.t) > 0)
    assert np.hypot(lap.x_m, lap.y_m) == pytest.approx(np.full(50, 1000.0))  # dm -> m
    assert lap.sector_end_t == pytest.approx((0.3 * 85.819, 0.7 * 85.819))


@pytest.mark.parametrize("driver", ["HAM", "ham", 44, "44"])
def test_fastest_lap_by_driver(session, driver):
    lap = fs.fastest_lap_xy(session, driver)
    assert lap.driver == "HAM"
    assert lap.lap_time_s == pytest.approx(85.99)


def test_fastest_lap_errors(session):
    with pytest.raises(ValueError, match="no laps.*Drivers: HAM, RUS"):
        fs.fastest_lap_xy(session, "VER")
    session.laps.laps[1]["LapTime"] = pd.NaT
    with pytest.raises(ValueError, match="no valid timed lap"):
        fs.fastest_lap_xy(session, "HAM")
    with pytest.raises(ValueError, match="No valid timed lap"):
        fs.fastest_lap_xy(FakeLoadedSession([]))


# ---------------------------------------------------------------- car data


class FakeCarLap(FakeLap):
    """A FakeLap that also serves get_car_data()."""

    _metadata = ["pos", "car"]

    @property
    def _constructor(self):
        return FakeCarLap

    def get_car_data(self, **kwargs):
        assert kwargs == {"pad": 1, "pad_side": "both"}
        return self.car


def _car_lap(driver, number, lap_time, team="Mercedes", start=1000.0):
    lap = FakeCarLap({**_lap(driver, number, lap_time, start=start), "Team": team})
    t = np.arange(-0.25, lap_time + 0.5, 0.25)
    car = pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta(start + t, unit="s"),
            "Speed": np.full(t.size, 180.0),
            "Throttle": np.where(t < lap_time / 2, 100.0, 104.0),  # 104 = sensor overrange
            "Brake": t > lap_time / 2,
            "nGear": np.full(t.size, 7),
        }
    )
    lap.car = pd.concat([car, car.iloc[[3]]]).sort_values("SessionTime")  # one duplicate
    return lap


def test_fastest_lap_car_data_matches_the_fastest_lap():
    session = FakeLoadedSession(
        [_car_lap("RUS", "63", 85.819), _car_lap("HAM", "44", 85.99, team=np.nan)]
    )
    car = fs.fastest_lap_car_data(session)
    assert (car.driver, car.team) == ("RUS", "Mercedes")
    assert car.t[0] == pytest.approx(-0.25)  # from the lap start, padded by one sample
    assert np.all(np.diff(car.t) > 0)  # sorted, duplicate dropped
    assert car.speed_ms == pytest.approx(np.full(car.t.size, 50.0))  # km/h -> m/s
    assert car.throttle01.max() == 1.0  # 0-100 -> 0..1, clipped
    assert set(car.brake01.tolist()) == {0.0, 1.0}
    assert car.gear.dtype == np.int64 and set(car.gear.tolist()) == {7}

    ham = fs.fastest_lap_car_data(session, 44)
    assert (ham.driver, ham.team) == ("HAM", "")
    with pytest.raises(ValueError, match="no laps"):
        fs.fastest_lap_car_data(session, "VER")


# ---------------------------------------------------------------- network


@pytest.fixture(scope="module")
def silverstone_q():
    return fs.load_session(2024, "Silverstone", "Q")


def _path_length(lap):
    x = np.append(lap.x_m, lap.x_m[0])
    y = np.append(lap.y_m, lap.y_m[0])
    return np.hypot(np.diff(x), np.diff(y)).sum()


@pytest.mark.network
def test_silverstone_2024_q_fastest_lap(silverstone_q):
    lap = fs.fastest_lap_xy(silverstone_q)
    assert lap.driver == "RUS"
    assert lap.lap_time_s == pytest.approx(85.819, abs=0.05)
    assert 5800 < _path_length(lap) < 5950
    assert lap.t[0] == pytest.approx(0.0, abs=1e-6)
    assert lap.t[-1] == pytest.approx(lap.lap_time_s, abs=0.01)
    assert np.all(np.diff(lap.t) > 0)
    s1, s2 = lap.sector_end_t
    assert 0.2 * lap.lap_time_s < s1 < s2 < 0.9 * lap.lap_time_s
    assert s1 == pytest.approx(28.0, abs=1.0)
    assert s2 == pytest.approx(62.5, abs=1.0)


@pytest.mark.network
@pytest.mark.parametrize("driver", ["HAM", 44])
def test_silverstone_2024_q_hamilton(silverstone_q, driver):
    lap = fs.fastest_lap_xy(silverstone_q, driver)
    assert lap.driver == "HAM"
    assert lap.lap_time_s == pytest.approx(85.99, abs=0.05)
    assert 5800 < _path_length(lap) < 5950
    s1, s2 = lap.sector_end_t
    assert 0 < s1 < s2 < lap.lap_time_s


@pytest.mark.network
def test_event_schedule_2024():
    events = fs.event_schedule(2024)
    assert [e.round for e in events] == list(range(1, 25))
    silverstone = events[11]
    assert (silverstone.name, silverstone.location) == ("British Grand Prix", "Silverstone")


@pytest.mark.network
def test_silverstone_2024_q_hamilton_car_data(silverstone_q):
    car = fs.fastest_lap_car_data(silverstone_q, "HAM")
    assert (car.driver, car.team) == ("HAM", "Mercedes")
    assert car.t[0] <= 0.0 and car.t[-1] >= 85.9
    assert 300 / 3.6 < car.speed_ms.max() < 360 / 3.6
    assert set(car.gear.tolist()) <= set(range(9))
