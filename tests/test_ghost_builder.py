"""Ghost builder: offline tests on a synthetic stadium track, plus marked network checks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim import paths
from f1sim.data import ghost_builder as gb
from f1sim.data import laps
from f1sim.data.io import write_track_json
from f1sim.data.laps import CarTrace, LapXY
from f1sim.data.track_builder import build_track
from f1sim.data.track_pipeline import track_path
from f1sim.track.track import Track

G = 9.80665
SPEED = 50.0  # m/s
RAW_DT = 0.27  # ~3.7 Hz, like the live-timing positions
NOISE_M = 0.5
HALF_WIDTH = 6.0
IDENTITY = {"rotation_rad": 0.0, "tx_m": 0.0, "ty_m": 0.0, "scale": 1.0, "mirrored": False}


@pytest.fixture(scope="module")
def track_data():
    x, y = stadium(400.0, 80.0)
    w = np.full(x.size, HALF_WIDTH)
    data = build_track(x, y, w, w, track_id="stadium", name="Stadium")
    alignment = {**IDENTITY, "median_dist_m": 0.0, "inside_ratio": 1.0, "lap_source": "openf1"}
    return replace(data, fastf1_alignment=alignment)


@pytest.fixture(scope="module")
def track(track_data):
    return Track(track_data)


def _synthetic_lap(
    track: Track,
    s_of_t,
    lap_time: float,
    *,
    noise_m: float = NOISE_M,
    offset_n: float = 0.0,
    seed: int = 0,
    driver: str = "HAM",
) -> LapXY:
    """Positions along s(t) sampled every RAW_DT with noise, cut to [0, lap_time] like LapXY."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.11, lap_time, RAW_DT)
    t = np.concatenate([[0.0], t, [lap_time]])
    s = s_of_t(t)
    p = track.point_at(s)
    h = track.heading_at(s)
    x = p[:, 0] - np.sin(h) * offset_n + rng.normal(0.0, noise_m, t.size)
    y = p[:, 1] + np.cos(h) * offset_n + rng.normal(0.0, noise_m, t.size)
    return LapXY(driver=driver, lap_time_s=lap_time, t=t, x_m=x, y_m=y, sector_end_t=(1.0, 2.0))


def _car(lap_time: float, *, driver: str = "HAM", team: str = "Mercedes") -> CarTrace:
    t = np.arange(-1.0, lap_time + 1.0, 0.25)
    return CarTrace(
        t=t,
        speed_ms=np.full(t.size, SPEED),
        throttle01=np.clip(t / lap_time, 0.0, 1.0),
        brake01=(t > lap_time / 2).astype(float),
        gear=np.where(t > lap_time / 2, 4, 7),
        driver=driver,
        team=team,
    )


@pytest.fixture
def cache(tmp_path, monkeypatch, track_data):
    """A temporary cache with the stadium track built and aligned."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    write_track_json(track_data, track_path("stadium"))
    return tmp_path


def _stub_sources(monkeypatch, lap: LapXY, car: CarTrace, source: str = "openf1"):
    calls = []

    def fake_lap(track_id, year, *, kind="Q", driver=None, source="auto"):
        calls.append(("lap", track_id, year, kind, driver, source))
        return lap, "openf1" if source == "auto" else source

    def fake_car(track_id, year, *, kind="Q", driver=None, source="auto"):
        calls.append(("car", track_id, year, kind, driver, source))
        return car

    monkeypatch.setattr(laps, "reference_lap", fake_lap)
    monkeypatch.setattr(laps, "reference_car_data", fake_car)
    return calls


def _constant_speed_lap(track, **kw):
    lap_time = track.length / SPEED
    return _synthetic_lap(track, lambda t: SPEED * t, lap_time, **kw), lap_time


# ---------------------------------------------------------------- smoothing


def test_smoothing_removes_the_linear_interpolation_jitter(track):
    lap, lap_time = _constant_speed_lap(track)
    t = gb._eval_times(lap_time, 20.0)
    sm = gb.smooth_positions(lap.t, lap.x_m, lap.y_m, t)

    # The legacy way: straight lines between the raw samples, speed by finite differences
    lx, ly = np.interp(t, lap.t, lap.x_m), np.interp(t, lap.t, lap.y_m)
    lin_speed = np.hypot(np.gradient(lx, t), np.gradient(ly, t))
    lin_accel_g = np.abs(np.gradient(lin_speed, t)) / G

    accel_g = np.abs(sm["accel"]) / G
    assert lin_accel_g.max() > 5.0  # the jitter the legacy ghost had
    assert accel_g.max() < 0.1 * lin_accel_g.max()
    assert accel_g.max() < 1.0  # true value 0
    # Speed within 2% of the truth, without bias. The spline's free ends average less noise,
    # so within the first and last second 0.5 m of noise leaves errors of up to ~3%.
    err = np.abs(sm["speed"] / SPEED - 1.0)
    inner = (t > 1.0) & (t < lap_time - 1.0)
    assert np.median(err) < 0.01
    assert np.percentile(err[inner], 95) < 0.02
    assert err.max() < 0.04
    assert sm["speed"].mean() == pytest.approx(SPEED, rel=0.002)


def test_eval_times_cover_the_lap_at_the_rate():
    t = gb._eval_times(1.0, 20.0)
    assert t.size == 21 and t[0] == 0.0 and t[-1] == 1.0
    assert np.diff(t) == pytest.approx(np.full(20, 0.05))
    t = gb._eval_times(1.01, 20.0)
    assert t[-1] == 1.01 and t[-2] == pytest.approx(1.0)


def test_s_is_monotonic_and_starts_near_zero_across_the_line(track):
    # Start 3 m before the line: the first projection wraps to length - 3 and is unwrapped
    lap, lap_time = _constant_speed_lap(track)
    shifted = _synthetic_lap(track, lambda t: SPEED * t - 3.0, lap_time)
    for raw, s_start in ((lap, 0.0), (shifted, -3.0)):
        g = gb.ghost_from_positions(track, raw.t, raw.x_m, raw.y_m, lap_time, _car(lap_time))
        assert np.all(np.diff(g.s) > 0)
        assert g.s[0] == pytest.approx(s_start, abs=1.0)
        assert g.s[-1] - g.s[0] == pytest.approx(track.length, abs=2.0)
        truth = SPEED * g.t + s_start
        assert np.abs(g.s - truth).max() < 1.0


# ---------------------------------------------------------------- car-data channels


def test_resample_channels_linear_throttle_nearest_brake_and_gear():
    car = CarTrace(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        speed_ms=np.array([10.0, 20.0, 30.0, 40.0]),
        throttle01=np.array([0.0, 1.0, 0.5, 0.5]),
        brake01=np.array([0.0, 0.0, 1.0, 1.0]),
        gear=np.array([3, 4, 5, 6]),
    )
    ch = gb.resample_channels(car, np.array([-1.0, 0.25, 0.5, 0.75, 1.6, 5.0]))
    assert ch["throttle"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 0.7, 0.5])
    assert ch["brake"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # a tie goes to the earlier
    assert ch["gear"].tolist() == [3, 3, 3, 4, 5, 6]
    assert ch["speed_sensor"] == pytest.approx([10.0, 12.5, 15.0, 17.5, 26.0, 40.0])


# ---------------------------------------------------------------- build_ghost with stubbed sources


def test_build_ghost_from_stubbed_sources(cache, track, monkeypatch):
    lap, lap_time = _constant_speed_lap(track)
    calls = _stub_sources(monkeypatch, lap, _car(lap_time))
    g = gb.build_ghost("stadium", 2024, driver=44, kind="Q")

    assert calls[0] == ("lap", "stadium", 2024, "Q", 44, "auto")
    assert calls[1] == ("car", "stadium", 2024, "Q", "HAM", "openf1")  # same lap, same source
    assert g.t[0] == 0.0 and g.t[-1] == pytest.approx(lap_time)
    assert np.diff(g.t)[:-1] == pytest.approx(np.full(g.t.size - 2, 0.05))
    m = g.meta
    assert (m["driver"], m["team"], m["year"], m["session"]) == ("HAM", "Mercedes", 2024, "Q")
    assert m["lap_source"] == "openf1" and m["lap_time_s"] == pytest.approx(lap_time)
    assert m["track_build"]["track_id"] == "stadium"
    assert m["track_build"]["builder_version"] is not None
    assert m["speed_check_kmh"] < 0.02 * SPEED * 3.6
    assert m["near_edge_ratio"] == 1.0
    # Channels come from the car data at the ghost's times
    ramp = g.t < lap_time - 0.25  # the ramp's end falls between car-data samples
    assert g.throttle[ramp] == pytest.approx(g.t[ramp] / lap_time, abs=1e-9)
    half = g.t > lap_time / 2 + 0.2
    early = g.t < lap_time / 2 - 0.2
    assert np.all(g.brake[half] == 1.0) and np.all(g.brake[early] == 0.0)
    assert np.all(g.gear[half] == 4) and np.all(g.gear[early] == 7)
    assert g.gear.dtype.kind == "i"


def test_build_ghost_applies_the_stored_alignment(cache, track, track_data, monkeypatch):
    # Store a non-trivial transform and feed the lap in the "raw" frame it maps from
    a = {**IDENTITY, "rotation_rad": 0.3, "tx_m": 120.0, "ty_m": -40.0, "mirrored": True}
    write_track_json(
        replace(track_data, fastf1_alignment={**track_data.fastf1_alignment, **a}),
        track_path("stadium"),
    )
    lap, lap_time = _constant_speed_lap(track)
    c, k = np.cos(-0.3), np.sin(-0.3)
    dx, dy = lap.x_m - 120.0, lap.y_m + 40.0
    raw = replace(lap, x_m=c * dx - k * dy, y_m=-(k * dx + c * dy))
    _stub_sources(monkeypatch, raw, _car(lap_time))
    g = gb.build_ghost("stadium", 2024, driver="HAM")
    assert g.meta["near_edge_ratio"] == 1.0
    p = track.point_at(SPEED * g.t)
    assert np.median(np.hypot(g.x - p[:, 0], g.y - p[:, 1])) < 0.5


def test_build_ghost_fastest_lap_names_its_driver(cache, track, monkeypatch):
    lap, lap_time = _constant_speed_lap(track, driver="RUS")
    calls = _stub_sources(monkeypatch, lap, _car(lap_time, driver="RUS", team=""))
    g = gb.build_ghost("stadium", 2024)
    assert calls[1][4] == "RUS"
    assert (g.meta["driver"], g.meta["team"]) == ("RUS", "")
    assert gb.ghost_path(2024, "stadium", "Q", g.meta["driver"]).name == "2024_stadium_Q_RUS.npz"


def test_build_ghost_needs_a_stored_alignment(cache, track, track_data, monkeypatch):
    write_track_json(replace(track_data, fastf1_alignment=None), track_path("stadium"))
    lap, lap_time = _constant_speed_lap(track)
    _stub_sources(monkeypatch, lap, _car(lap_time))
    with pytest.raises(ValueError, match="no fastf1_alignment"):
        gb.build_ghost("stadium", 2024, driver="HAM")


# ---------------------------------------------------------------- validation


def test_rejects_a_lap_with_impossible_acceleration(track):
    # 90 m/s, then 10 m/s from mid-lap on: a speed step no car can make
    t1, v1, v2 = 8.0, 90.0, 10.0
    lap_time = t1 + (track.length - v1 * t1) / v2

    def s_of_t(t):
        return np.where(t < t1, v1 * t, v1 * t1 + v2 * (t - t1))

    lap = _synthetic_lap(track, s_of_t, lap_time, noise_m=0.1)
    with pytest.raises(ValueError, match=r"\|dv/dt\| reaches"):
        gb.ghost_from_positions(track, lap.t, lap.x_m, lap.y_m, lap_time, _car(lap_time))


def test_rejects_a_lap_off_the_track(track):
    lap_time = track.length / SPEED
    lap = _synthetic_lap(track, lambda t: SPEED * t, lap_time, offset_n=HALF_WIDTH + 3.0)
    with pytest.raises(ValueError, match="within 1 m of the track edges"):
        gb.ghost_from_positions(track, lap.t, lap.x_m, lap.y_m, lap_time, _car(lap_time))
    # Just inside the tolerance is fine
    lap = _synthetic_lap(track, lambda t: SPEED * t, lap_time, offset_n=HALF_WIDTH, noise_m=0.1)
    g = gb.ghost_from_positions(track, lap.t, lap.x_m, lap.y_m, lap_time, _car(lap_time))
    assert g.meta["near_edge_ratio"] >= 0.95


# ---------------------------------------------------------------- paths, save and load


def test_ghost_path(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    p = gb.ghost_path(2024, "silverstone", "Q", "ham")
    assert p == tmp_path / "ghosts" / "2024_silverstone_Q_HAM.npz"
    assert gb.ghost_path(2019, "spielberg", "r", "VER").name == "2019_spielberg_R_VER.npz"


def test_save_load_round_trip(cache, track):
    lap, lap_time = _constant_speed_lap(track)
    meta = {"driver": "HAM", "team": "Mercedes", "year": 2024, "session": "Q"}
    g = gb.ghost_from_positions(track, lap.t, lap.x_m, lap.y_m, lap_time, _car(lap_time), meta=meta)
    path = gb.ghost_path(2024, "stadium", "Q", "HAM")
    gb.save_ghost(g, path)
    assert path.exists() and path.with_suffix(".json").exists()
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "2024_stadium_Q_HAM.json",
        "2024_stadium_Q_HAM.npz",
    ]  # no temporary files left behind

    back = gb.load_ghost(path)
    for name in ("t", "x", "y", "s", "speed", "throttle", "brake", "gear"):
        a, b = getattr(g, name), getattr(back, name)
        assert np.array_equal(a, b) and a.dtype == b.dtype, name
    assert back.meta == g.meta

    np.savez(path, t=g.t)  # an NPZ with columns missing
    with pytest.raises(ValueError, match="missing arrays"):
        gb.load_ghost(path)
    path.with_suffix(".json").write_text('{"format": "other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an f1sim ghost"):
        gb.load_ghost(path)


# ---------------------------------------------------------------- network (Silverstone 2024)


@pytest.mark.network
@pytest.mark.parametrize(("driver", "lap_time"), [("HAM", 85.99), ("RUS", 85.819)])
def test_silverstone_2024_ghost(driver, lap_time):
    g = gb.build_ghost("silverstone", 2024, driver=driver)
    m = g.meta
    assert m["driver"] == driver and m["team"] == "Mercedes"
    assert m["lap_time_s"] == pytest.approx(lap_time)
    assert g.t[-1] == pytest.approx(lap_time)
    assert m["max_accel_g"] < 6.0
    assert m["near_edge_ratio"] >= 0.98
    assert m["speed_check_kmh"] < 15.0
    assert np.all(np.diff(g.s) > 0) and abs(g.s[0]) < 10.0
    assert 330 / 3.6 < g.speed.max() < 360 / 3.6
