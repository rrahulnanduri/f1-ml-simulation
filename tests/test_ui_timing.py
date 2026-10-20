"""HUD timing (pure numpy) and the viewers' HUD data: no window is opened."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from _geometry_helpers import circle
from f1sim.app.ui import theme
from f1sim.app.ui import timing as tm
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track

NAN = float("nan")


def test_formatting():
    assert tm.format_lap(84.892) == "1:24.892"
    assert tm.format_lap(None) == "-:--.---"
    assert tm.format_lap(NAN) == "-:--.---"
    assert tm.format_gap(0.927) == "+0.927"
    assert tm.format_gap(-0.3124) == "-0.312"
    assert tm.format_gap(None) == "--.---"
    assert tm.format_sector(28.4104) == "28.410"
    assert tm.format_sector(NAN) == "--.---"


def test_sector_times_interpolate_crossings():
    # 100 m lap at 10 m/s with a 2 s lead-in; sectors at 30 and 70 m.
    t = np.arange(-2.0, 10.0001, 0.3)
    s = 10.0 * t
    times = tm.sector_times(t, s, (0.0, 30.0, 70.0), 100.0)
    assert times == pytest.approx([3.0, 4.0, 3.0])
    # No sectors: one sector, the lap.
    assert tm.sector_times(t, s, (), 100.0) == pytest.approx([10.0])
    # A trace that stops at 80 m leaves the last sector unset.
    keep = s <= 80.0
    assert math.isnan(tm.sector_times(t[keep], s[keep], (0.0, 30.0, 70.0), 100.0)[2])


def test_gap_at_equal_s():
    t = np.linspace(0.0, 10.0, 101)
    s = 10.0 * t
    slow_t = t * 1.1
    gap = tm.gap_at_s(slow_t, s, t, s, [0.0, 50.0, 100.0])
    assert gap == pytest.approx([0.0, 0.5, 1.0])
    assert np.isnan(tm.gap_at_s(slow_t, s, t, s, 150.0))
    # Stalls and backward steps are ignored.
    s2 = s.copy()
    s2[50:55] = s2[49]
    assert np.isfinite(tm.gap_at_s(t, s2, t, s, 70.0))


def test_resample_speed():
    s = np.linspace(-10.0, 60.0, 71)
    v = s * 2.0
    grid, out = tm.resample_speed(s, v, 100.0, n=11)
    assert grid == pytest.approx(np.linspace(0, 100, 11))
    assert out[:7] == pytest.approx(grid[:7] * 2.0)
    assert np.all(np.isnan(out[7:]))  # not reached
    _, empty = tm.resample_speed([], [], 100.0, n=5)
    assert np.all(np.isnan(empty))


def test_sector_colours():
    assert tm.sector_colour(NAN, 30.0, 29.0) == "unset"
    assert tm.sector_colour(28.9, 30.0, 29.0) == "purple"
    assert tm.sector_colour(29.5, 30.0, 29.0) == "green"
    assert tm.sector_colour(30.5, 30.0, 29.0) == "yellow"
    assert tm.sector_colour(30.5, None, 29.0) == "green"  # first lap: a personal best
    assert tm.sector_colour(30.5, None, None) == "purple"  # nothing to compare with


def test_tower_rows_rank_and_format():
    entries = [
        tm.TowerEntry("AI", 87.458, theme.SELF, True),
        tm.TowerEntry("VER", 84.892, (1, 2, 3, 255)),
        tm.TowerEntry("NONE", None, (9, 9, 9, 255)),
        tm.TowerEntry("RUS", 85.819, (4, 5, 6, 255)),
    ]
    rows = tm.tower_rows(entries)
    assert [r.name for r in rows] == ["VER", "RUS", "AI", "NONE"]
    assert [r.position for r in rows] == ["1", "2", "3", ""]
    assert [r.value for r in rows] == ["1:24.892", "+0.927", "+2.566", "-:--.---"]
    assert [r.is_self for r in rows] == [False, False, True, False]
    assert tm.tower_rows([]) == []


def test_live_positions():
    assert tm.live_positions([100.0, 300.0, 200.0]) == [3, 1, 2]


def test_lap_telemetry_splits_bests_and_display():
    tel = tm.LapTelemetry(100.0, (0.0, 30.0, 70.0), n_bins=11)
    assert tel.n_sectors == 3

    def drive_lap(speed, t0=0.0):
        dt = 0.25
        t, p = 0.0, 0.0
        while p + speed * dt < 100.0:
            t += dt
            p += speed * dt
            tel.record(p, t, speed * 3.6)
        tel.complete_lap(100.0 / speed)

    drive_lap(10.0)  # 3, 4, 3 s
    assert tel.last == pytest.approx([3.0, 4.0, 3.0])
    assert tel.best == pytest.approx([3.0, 4.0, 3.0])
    # Between laps the last lap is shown, compared with the bests from before it.
    times, before = tel.displayed()
    assert times == pytest.approx([3.0, 4.0, 3.0]) and all(math.isnan(b) for b in before)
    # A slower first sector on the next lap is yellow against the personal best.
    tel.record(0.0, 0.0, 0.0)
    tel.record(35.0, 4.2, 0.0)  # crosses 30 m at 3.6 s
    times, before = tel.displayed()
    assert times[0] == pytest.approx(3.6) and math.isnan(times[1])
    assert tm.sector_colour(times[0], before[0], before[0]) == "yellow"
    assert np.isfinite(tel.speed).sum() == 2  # bins at 0 and 35 m
    tel.discard_lap()
    assert all(math.isnan(v) for v in tel.current)
    assert tel.best == pytest.approx([3.0, 4.0, 3.0])


# --------------------------------------------------------------------- replay HUD data


def _trace(label, speed, length, meta=None):
    from f1sim.app.replay import Trace

    t = np.arange(0.0, length / speed + 0.5, 0.05)
    s = np.minimum(speed * t, length + 10.0)
    ang = s / 150.0
    return Trace(
        label,
        t,
        150 * np.cos(ang),
        150 * np.sin(ang),
        s,
        np.full(t.size, speed),
        float(t[-1]),
        meta=meta,
    )


@pytest.fixture(scope="module")
def circle_track() -> Track:
    x, y = circle(150.0, 300)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="circle", name="Circle")
    length = data.length_m
    return Track(replace(data, sectors_s=(0.0, length / 3, 2 * length / 3)))


def test_replay_timing_reference_tower_gap_and_sectors(circle_track):
    from f1sim.app.replay import ReplaySession, ReplayTiming, default_self_index

    length = circle_track.length
    ghost = {"driver": "VER", "year": 2025, "team": "Red Bull Racing"}
    traces = [
        _trace("VER '25", 50.0, length, ghost),
        _trace("RUS '24", 49.0, length, {"driver": "RUS", "year": 2024}),
        _trace("AI", 48.0, length),
    ]
    me = default_self_index(traces)
    assert me == 2  # the first trace that isn't a ghost
    session = ReplaySession(traces, length, follow=me)
    colours = theme.trace_colours(["Red Bull Racing", "", None], [True, True, False])
    timing = ReplayTiming(session, length, circle_track.data.sectors_s, colours)
    assert timing.lap_times[0] == pytest.approx(length / 50.0, abs=1e-6)
    assert timing.reference(me) == 0  # fastest other
    assert timing.reference(0) == 1
    rows = timing.tower(me)
    assert [r.name for r in rows] == ["VER '25", "RUS '24", "AI"]
    assert rows[2].is_self and rows[2].value.startswith("+")
    assert rows[0].value == tm.format_lap(length / 50.0)

    session.clock.seek_to(10.0)
    poses = session.poses()
    gap, s_q = timing.gap(me, 0, poses)
    assert s_q == pytest.approx(min(poses[me].s, poses[0].s))
    assert gap == pytest.approx(s_q / 48.0 - s_q / 50.0, abs=1e-3)  # behind: positive
    assert math.isnan(timing.gap(me, None, poses)[0])

    times, states = timing.sectors_at(me, poses[me].s)  # ~480 m: first sector only
    third = length / 3
    assert times[0] == pytest.approx(third / 48.0, abs=1e-3)
    assert math.isnan(times[1]) and states[1:] == ["unset", "unset"]
    assert states[0] == "green"  # slower than VER's sector, a personal best
    times, states = timing.sectors_at(0, length)
    assert states == ["purple"] * 3


def test_load_trace_reads_ghost_sidecar(tmp_path):
    from f1sim.app.replay import load_trace

    p = tmp_path / "2025_silverstone_Q_VER.npz"
    np.savez(p, t=np.arange(3.0), x=np.zeros(3), y=np.arange(3.0), gear=np.array([3, 4, 5]))
    plain = load_trace(p)
    assert plain.meta is None and plain.label == p.stem and not plain.is_ghost
    assert plain.gear == pytest.approx([3, 4, 5])
    doc = {
        "format": "f1sim.ghost",
        "version": 1,
        "driver": "VER",
        "year": 2025,
        "team": "Red Bull Racing",
    }
    p.with_suffix(".json").write_text(json.dumps(doc), encoding="utf-8")
    g = load_trace(p)
    assert g.is_ghost and g.meta["team"] == "Red Bull Racing" and "format" not in g.meta
    assert g.label == "VER ’25"
    assert load_trace(p, "mine").label == "mine"
    p.with_suffix(".json").write_text("{not json", encoding="utf-8")
    assert load_trace(p).meta is None


# ---------------------------------------------------------------------- drive HUD data


def test_drive_session_records_telemetry_and_tower(circle_track):
    from f1sim.app.drive import DriveSession, drive_sector_display, drive_tower

    session = DriveSession(circle_track)
    session.clock_running = True
    length = circle_track.length
    v, dt = 50.0, 0.1
    for k in range(1, int(1.5 * length / (v * dt))):
        p = circle_track.point_at(k * v * dt)
        session.state.x[:] = p[0]
        session.state.y[:] = p[1]
        session.state.v[:] = v
        session.update_progress(dt)
    assert session.laps == 1
    last = session.telemetry.last
    assert last == pytest.approx([length / 3 / v] * 3, abs=0.02)
    times, states = drive_sector_display(session, None)
    assert np.isfinite(times[0]) and states[0] in ("purple", "green")

    rows = drive_tower(session, None)
    assert [r.name for r in rows] == ["You", "Last lap", "This lap"]
    assert rows[0].is_self and rows[0].position == "1"
    assert rows[1].value == tm.format_lap(session.last_lap)

    session.reset()  # the running lap is discarded, the bests stay
    assert all(math.isnan(x) for x in session.telemetry.current)
    assert session.telemetry.best == pytest.approx(last)


def test_drive_ghost_gap_and_purple_sectors(circle_track, tmp_path):
    from f1sim.app.drive import DriveGhost, DriveSession, drive_sector_display, drive_tower

    length = circle_track.length
    tr = _trace("x", 60.0, length)
    p = tmp_path / "ghost.npz"
    np.savez(p, t=tr.t, x=tr.x, y=tr.y, s=tr.s, speed=tr.speed)
    doc = {"format": "f1sim.ghost", "version": 1, "driver": "HAM", "year": 2024, "team": "Mercedes"}
    p.with_suffix(".json").write_text(json.dumps(doc), encoding="utf-8")
    ghost = DriveGhost.load(p, circle_track)
    assert ghost.label == "HAM ’24" and ghost.long_label == "Hamilton 2024"
    assert ghost.colour == theme.TEAM_COLOURS["mercedes"]
    assert ghost.lap_time == pytest.approx(length / 60.0, abs=1e-3)
    assert ghost.gap(12.0, 600.0) == pytest.approx(12.0 - 10.0, abs=1e-3)

    session = DriveSession(circle_track)
    rows = drive_tower(session, ghost)
    assert [r.name for r in rows[:2]] == ["HAM ’24", "You"]
    assert rows[1].position == "" and rows[3].value == "-:--.---"  # clock not running
    session.telemetry.current = [length / 3 / 70.0, NAN, NAN]  # faster than the ghost
    _, states = drive_sector_display(session, ghost)
    assert states == ["purple", "unset", "unset"]
    session.telemetry.current = [length / 3 / 50.0, NAN, NAN]
    _, states = drive_sector_display(session, ghost)
    assert states[0] == "green"
