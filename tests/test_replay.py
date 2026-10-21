"""Replay viewer logic (M3-L), offline: no window is opened."""

import numpy as np
import pytest

from _geometry_helpers import circle
from f1sim.app import replay
from f1sim.app.replay import (
    SPEED_STEPS,
    PlaybackClock,
    PreparedTrace,
    ReplaySession,
    Trace,
    delta_at_s,
    hermite,
    interp_heading,
    interp_position,
    load_trace,
    smooth_heading,
    unwrap_s,
    with_track_s,
)
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _circle_lap(radius=150.0, speed=50.0, rate_hz=50.0, laps=1.0, scale=1.0):
    """Counter-clockwise run around a circle at constant speed, starting at (radius, 0)."""
    period = 2.0 * np.pi * radius / speed
    t = np.arange(0.0, laps * period + 1e-9, 1.0 / rate_hz)
    ang = speed / radius * t
    return Trace(
        label="c",
        t=t * scale,
        x=radius * np.cos(ang),
        y=radius * np.sin(ang),
        s=radius * ang,
        speed=np.full(t.size, speed / scale),
        lap_time_s=float(t[-1] * scale),
    )


@pytest.fixture(scope="module")
def circle_track() -> Track:
    x, y = circle(150.0, 300)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="circle", name="Circle")
    return Track(data)


# ----------------------------------------------------------------------------- loading


def test_load_trace_with_all_columns(tmp_path):
    t = np.arange(0.0, 5.0, 0.05)
    p = tmp_path / "2024_silverstone_Q_HAM.npz"
    cols = dict(
        t=t,
        x=10.0 * t,
        y=np.zeros_like(t),
        s=10.0 * t,
        speed=np.full(t.size, 10.0),
        throttle=np.ones_like(t),
        brake=np.zeros_like(t),
        gear=np.full(t.size, 7, dtype=np.int64),
    )
    np.savez(p, **cols)
    tr = load_trace(p)
    assert tr.label == "2024_silverstone_Q_HAM"
    assert tr.has_s
    np.testing.assert_array_equal(tr.s, cols["s"])
    np.testing.assert_array_equal(tr.speed, cols["speed"])
    assert tr.lap_time_s == pytest.approx(t[-1] - t[0])
    assert load_trace(p, "HAM").label == "HAM"


def test_load_trace_minimal_columns(tmp_path):
    t = np.linspace(2.0, 12.0, 201)  # not starting at 0
    p = tmp_path / "run7" / "best_lap.npz"
    p.parent.mkdir()
    np.savez(p, t=t, x=3.0 * t, y=4.0 * t)  # 5 m/s along a diagonal
    tr = load_trace(p)
    assert tr.label == "run7 best"
    assert not tr.has_s and tr.s.size == 0
    np.testing.assert_allclose(tr.speed, 5.0)
    assert tr.lap_time_s == pytest.approx(10.0)


def test_load_trace_rejects_bad_files(tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, t=np.arange(3.0), x=np.zeros(3))
    with pytest.raises(ValueError, match="missing column"):
        load_trace(p)
    np.savez(p, t=np.array([0.0, 1.0, 1.0]), x=np.zeros(3), y=np.zeros(3))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_trace(p)
    np.savez(p, t=np.arange(3.0), x=np.zeros(3), y=np.zeros(4))
    with pytest.raises(ValueError, match="expected 3"):
        load_trace(p)


def test_with_track_s_projects_and_unwraps(circle_track):
    # 1.3 laps starting just before the line: s must be continuous, with the lead-in negative.
    radius, speed = 150.0, 40.0
    t = np.arange(0.0, 1.3 * 2 * np.pi * radius / speed, 0.05)
    ang = -0.2 + speed / radius * t
    tr = Trace("p", t, radius * np.cos(ang), radius * np.sin(ang), np.empty(0), t * 0 + speed, 1.0)
    filled = with_track_s(tr, circle_track)
    assert filled.has_s
    assert np.all(np.diff(filled.s) > 0.0)
    np.testing.assert_allclose(filled.s, radius * ang, atol=0.5)
    assert with_track_s(filled, circle_track) is filled  # already has s


def test_unwrap_s():
    length = 1000.0
    s_true = np.arange(-30.0, 2100.0, 7.0)
    np.testing.assert_allclose(unwrap_s(np.mod(s_true, length), length), s_true, atol=1e-9)
    np.testing.assert_allclose(unwrap_s(s_true + length, length), s_true, atol=1e-9)


# ------------------------------------------------------------------------ interpolation


def test_interpolation_at_and_between_samples():
    t = np.array([0.0, 0.5, 1.0, 2.0])
    x = np.array([0.0, 10.0, 15.0, 35.0])
    y = np.array([5.0, 5.0, 0.0, -10.0])
    px, py = interp_position(t, x, y, t)
    np.testing.assert_array_equal(px, x)
    np.testing.assert_array_equal(py, y)
    px, py = interp_position(t, x, y, np.array([0.25, 0.75, 1.5]))
    np.testing.assert_allclose(px, [5.0, 12.5, 25.0])
    np.testing.assert_allclose(py, [5.0, 2.5, -5.0])
    px, py = interp_position(t, x, y, np.array([-1.0, 3.0]))  # clamped to the ends
    np.testing.assert_allclose(px, [0.0, 35.0])


def test_hermite_is_exact_at_samples_and_cubic_between():
    t = np.array([0.0, 0.3, 1.0, 1.5])
    f = t**3 - 2 * t  # a cubic: Hermite with exact slopes reproduces it everywhere
    df = 3 * t**2 - 2
    np.testing.assert_allclose(hermite(t, f, df, t), f, atol=1e-12)
    tq = np.linspace(-0.5, 2.0, 41)
    inside = np.clip(tq, t[0], t[-1])
    np.testing.assert_allclose(hermite(t, f, df, tq), inside**3 - 2 * inside, atol=1e-12)


def test_pose_of_a_20hz_trace_moves_smoothly_between_samples():
    """At 60 fps, linear interpolation of a 20 Hz lap jumps velocity at every sample (the
    tick seen on ghosts in fast corners); the Hermite pose stays near the true 2.5 g."""
    radius, speed = 100.0, 50.0
    tr = _circle_lap(radius, speed, rate_hz=20.0)
    p = PreparedTrace(tr, 2 * np.pi * radius)
    tc = np.arange(0.5, tr.t[-1] - 0.5, 1 / 60)
    poses = [p.pose(t) for t in tc]
    x = np.array([q.x for q in poses])
    y = np.array([q.y for q in poses])
    acc = np.hypot(np.diff(x, 2), np.diff(y, 2)) * 3600.0
    true_acc = speed**2 / radius
    assert np.abs(acc - true_acc).max() < 0.05 * true_acc
    lx, ly = interp_position(p.t, tr.x, tr.y, tc)
    lin_acc = np.hypot(np.diff(lx, 2), np.diff(ly, 2)) * 3600.0
    assert lin_acc.max() > 2.0 * true_acc  # what the old linear pose did
    heading = np.unwrap([q.heading for q in poses])
    yaw_rate = np.diff(heading) * 60.0
    assert np.abs(yaw_rate - speed / radius).max() < 0.02


def test_interp_heading_goes_the_short_way_across_pi():
    t = np.array([0.0, 1.0])
    h = np.unwrap(np.array([np.pi - 0.1, -np.pi + 0.1]))
    mid = float(interp_heading(t, h, 0.5))
    assert abs(_wrap(mid - np.pi)) < 1e-12
    assert abs(_wrap(float(interp_heading(t, h, 0.25)) - (np.pi - 0.05))) < 1e-12


def test_heading_smoothing_on_noisy_circle():
    radius, speed, rate = 100.0, 60.0, 50.0
    tr = _circle_lap(radius, speed, rate, laps=2.2)  # crosses +-pi several times
    rng = np.random.default_rng(1)
    x = tr.x + rng.normal(0.0, 0.03, tr.x.size)
    y = tr.y + rng.normal(0.0, 0.03, tr.y.size)
    truth = speed / radius * tr.t + 0.5 * np.pi  # CCW tangent

    raw = np.arctan2(np.gradient(y, tr.t), np.gradient(x, tr.t))
    smoothed = smooth_heading(tr.t, x, y)
    err_raw = np.abs(_wrap(raw - truth))
    err = np.abs(_wrap(smoothed - truth))
    # Ends excluded: the window shrinks there (one 0.2 s window each side).
    inner = slice(int(0.2 * rate), -int(0.2 * rate))
    assert np.degrees(err_raw[inner].max()) > 1.0  # the noise matters unsmoothed...
    assert np.degrees(err[inner].max()) < 1.0  # ...and not after smoothing
    assert np.degrees(err.max()) < 2.0
    # No jumps across +-pi: steps match the true turn rate.
    step = np.abs(_wrap(np.diff(smoothed)))
    assert step.max() < 3.0 * speed / radius / rate
    assert np.all(np.abs(smoothed) <= np.pi)


def test_heading_of_a_standing_start_has_no_jump():
    t = np.arange(0.0, 4.0, 0.02)
    xpos = np.where(t < 1.0, 0.0, 0.5 * 5.0 * (t - 1.0) ** 2)  # still, then accelerating along -y
    h = smooth_heading(t, np.zeros_like(t), -xpos)
    np.testing.assert_allclose(h, -0.5 * np.pi, atol=1e-9)


# ------------------------------------------------------------------------------- delta


def test_delta_at_equal_s_for_time_scaled_trace():
    a = _circle_lap(laps=1.0)
    b = _circle_lap(laps=1.0, scale=1.02)
    s_q = np.linspace(0.0, a.s[-1], 50)
    d = delta_at_s(a.t, a.s, b.t, b.s, s_q)
    t1 = np.interp(s_q, a.s, a.t)
    np.testing.assert_allclose(d, 0.02 * t1, atol=1e-9)
    assert d[-1] == pytest.approx(0.02 * a.lap_time_s)
    assert np.isnan(delta_at_s(a.t, a.s, b.t, b.s, a.s[-1] + 1.0))  # beyond the traces
    # Reversed roles: trace 2 ahead gives a negative delta.
    assert delta_at_s(b.t, b.s, a.t, a.s, s_q[-1]) == pytest.approx(-0.02 * a.lap_time_s)


def test_delta_ignores_stalls_and_backward_steps():
    t = np.arange(6.0)
    s1 = np.array([0.0, 10.0, 10.0, 9.0, 20.0, 30.0])  # stall, then a small step back
    s2 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    assert delta_at_s(t, s1, t, s2, 20.0) == pytest.approx(2.0 - 4.0)


def test_session_aligns_lead_in_and_reports_delta():
    length = 2 * np.pi * 150.0
    a = _circle_lap()
    b = _circle_lap(scale=1.02)
    # b starts 2 s of driving before the line, with s wrapped into [0, length).
    lead = 2.0
    tb = np.concatenate([np.arange(-lead, 0.0, 0.02), b.t]) + 100.0  # arbitrary time origin
    sb = np.concatenate([np.arange(-lead, 0.0, 0.02) * 50.0 / 1.02, b.s])
    ang = sb / 150.0
    b2 = Trace("b", tb, 150 * np.cos(ang), 150 * np.sin(ang), np.mod(sb, length), tb * 0, 0.0)
    sess = ReplaySession([a, b2], length)
    assert sess.clock.start == pytest.approx(-lead)
    assert sess.clock.end == pytest.approx(b.t[-1])
    # At the line both cars are at s = 0.
    sess.clock.seek_to(0.0)
    p = sess.poses()
    assert p[0].s == pytest.approx(0.0, abs=1e-9) and p[1].s == pytest.approx(0.0, abs=1e-6)
    sess.clock.seek_to(sess.clock.end)
    assert sess.delta() == pytest.approx(0.02 * a.lap_time_s, abs=1e-6)
    # Follow target cycles through the traces, then the free camera.
    assert sess.follow == 0
    sess.cycle_follow()
    assert sess.follow == 1
    sess.cycle_follow()
    assert sess.follow is None
    sess.cycle_follow()
    assert sess.follow == 0


# ------------------------------------------------------------------------------- clock


def test_playback_clock_clamps_and_steps():
    c = PlaybackClock(-1.0, 10.0)
    assert c.time == -1.0 and c.speed == 1.0 and not c.paused
    c.update(0.1)
    assert c.time == pytest.approx(-0.9)
    c.update(-1.0)  # negative frame times are ignored
    assert c.time == pytest.approx(-0.9)
    c.update(5.0)  # long frames are capped
    assert c.time == pytest.approx(-0.9 + replay.MAX_FRAME_S)

    for _ in range(10):
        c.faster()
    assert c.speed == SPEED_STEPS[-1] == 8.0
    for _ in range(10):
        c.slower()
    assert c.speed == SPEED_STEPS[0] == 0.25
    c.update(0.2)
    assert c.time == pytest.approx(-0.9 + replay.MAX_FRAME_S + 0.05)

    c.seek_to(9.0)
    c.faster()  # 0.5x
    for _ in range(100):
        c.update(0.1)
    assert c.time == 10.0 and c.at_end  # held at the end
    c.seek(-2.0)
    assert c.time == pytest.approx(8.0)
    c.seek(-100.0)
    assert c.time == -1.0

    c.toggle_pause()
    c.update(0.1)
    assert c.time == -1.0 and c.paused
    c.seek(2.0)  # seeking works while paused
    assert c.time == pytest.approx(1.0)
    c.restart()
    assert c.time == -1.0 and not c.paused


def test_playback_clock_rejects_reversed_range():
    with pytest.raises(ValueError):
        PlaybackClock(5.0, 1.0)


# --------------------------------------------------------------------------------- run


def test_run_reports_missing_trace_file(tmp_path, capsys):
    replay.run("silverstone", [tmp_path / "nope.npz"])
    assert "not found" in capsys.readouterr().out


def test_run_reports_unbuilt_track(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "a.npz"
    np.savez(p, t=np.arange(3.0), x=np.zeros(3), y=np.zeros(3))
    replay.run("silverstone", [p])
    assert "hasn't been built" in capsys.readouterr().out
