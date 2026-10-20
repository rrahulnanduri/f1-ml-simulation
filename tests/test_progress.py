import dataclasses

import numpy as np
import pytest

from _geometry_helpers import circle, stadium, teardrop
from f1sim.data.track_builder import build_track
from f1sim.sim.progress import (
    DoneReason,
    ProgressEvents,
    ProgressParams,
    ProgressState,
    ProgressTracker,
)
from f1sim.track.track import Track

DT = 1.0 / 60.0
W_LEFT = 5.0
W_RIGHT = 6.0

# Wrong way and stall switched off, for tests about laps and sectors only.
NO_TERMINATION = ProgressParams(wrong_way_time_s=1e9, stall_time_s=1e9)


def make_track(xy, sectors=None) -> Track:
    x, y = xy
    data = build_track(
        x, y, np.full(x.size, W_RIGHT), np.full(x.size, W_LEFT), track_id="t", name="T"
    )
    if sectors is not None:
        data = dataclasses.replace(data, sectors_s=tuple(f * data.length_m for f in sectors))
    return Track(data)


@pytest.fixture(scope="module")
def circle_track() -> Track:
    return make_track(circle(200.0, 400), sectors=(0.0, 0.3, 0.7))


@pytest.fixture(scope="module")
def plain_circle() -> Track:
    return make_track(circle(200.0, 400))


@pytest.fixture(scope="module")
def oval() -> Track:
    return make_track(stadium(200.0, 60.0), sectors=(0.0, 0.25, 0.6))


@pytest.fixture(scope="module")
def small_drop() -> Track:
    return make_track(teardrop(25.0, 45.0, 80.0), sectors=(0.0, 0.4, 0.75))


def xy_at(track: Track, s, n=0.0):
    """Points at arc length ``s`` (wrapping) and lateral offset ``n`` (positive left)."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    n = np.broadcast_to(np.asarray(n, dtype=np.float64), s.shape)
    p = track.point_at(s)
    h = track.heading_at(s)
    return p[:, 0] - np.sin(h) * n, p[:, 1] + np.cos(h) * n


def run(tracker: ProgressTracker, track: Track, s_of_t, steps: int, dt: float = DT, n=0.0):
    """Reset at ``s_of_t(0)`` and drive ``steps`` updates; returns the list of events."""
    tracker.reset(*xy_at(track, s_of_t(0.0), n))
    events = []
    for k in range(1, steps + 1):
        events.append(tracker.update(*xy_at(track, s_of_t(k * dt), n), dt))
    return events


def stacked(events, name):
    return np.array([getattr(e, name) for e in events])


# ------------------------------------------------------------------------------- laps


@pytest.mark.parametrize("fixture", ["plain_circle", "circle_track", "oval"])
def test_exact_lap_count_and_interpolated_lap_times(fixture, request):
    track = request.getfixturevalue(fixture)
    v = 73.3
    lap_t = track.length / v
    steps = int(5.5 * lap_t / DT)
    tr = ProgressTracker(track, 1)
    ev = run(tr, track, lambda t: v * t, steps)
    assert tr.state.lap[0] == 5
    times = stacked(ev, "lap_time")[:, 0]
    done = stacked(ev, "lap_completed")[:, 0]
    assert done.sum() == 5
    np.testing.assert_allclose(times[done], lap_t, atol=1e-3)
    assert np.max(np.abs(times[done] - lap_t)) < 1e-6  # interpolation is exact here
    assert np.all(np.isnan(times[~done]))
    assert tr.state.last_lap_time[0] == pytest.approx(lap_t, abs=1e-6)
    assert tr.state.best_lap_time[0] == pytest.approx(lap_t, abs=1e-6)
    assert not tr.state.done[0]
    np.testing.assert_allclose(tr.state.progress[0], v * steps * DT, rtol=1e-9)


def test_lap_times_for_several_cars_at_different_speeds(circle_track):
    track = circle_track
    v = np.array([40.0, 55.5, 61.0, 90.0])
    tr = ProgressTracker(track, v.size, NO_TERMINATION)
    tr.reset(*xy_at(track, np.zeros(v.size)))
    best = np.full(v.size, np.nan)
    for k in range(1, int(3.2 * track.length / v.min() / DT)):
        e = tr.update(*xy_at(track, v * k * DT), DT)
        best = np.where(e.lap_completed, e.lap_time, best)
    np.testing.assert_array_equal(tr.state.lap, np.floor(v * k * DT / track.length))
    np.testing.assert_allclose(best, track.length / v, atol=1e-6)
    np.testing.assert_allclose(tr.state.best_lap_time, track.length / v, atol=1e-6)


def test_best_lap_is_the_minimum(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    length = track.length

    # First lap at 50 m/s, then 80 m/s, then 60 m/s.
    def s_of_t(t):
        t1, t2 = length / 50.0, length / 80.0
        if t <= t1:
            return 50.0 * t
        if t <= t1 + t2:
            return length + 80.0 * (t - t1)
        return 2 * length + 60.0 * (t - t1 - t2)

    total = length / 50 + length / 80 + length / 60
    run(tr, track, s_of_t, int(total / DT) + 5)
    assert tr.state.lap[0] == 3
    # The speed changes on the line, inside a step, so the linear interpolation is ~ms off.
    assert tr.state.best_lap_time[0] == pytest.approx(length / 80.0, abs=5e-3)
    assert tr.state.last_lap_time[0] == pytest.approx(length / 60.0, abs=5e-3)


# ---------------------------------------------------------------------- no fake laps


def test_zigzag_across_the_line_never_counts(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    # +-15 m across the start line, 0.5 Hz, for 2 minutes.
    ev = run(tr, track, lambda t: 15.0 * np.sin(np.pi * t), int(120 / DT))
    assert tr.state.lap[0] == 0
    assert not stacked(ev, "lap_completed").any()
    assert (stacked(ev, "sector_completed") == -1).all()
    assert np.isnan(tr.state.last_lap_time[0])


def test_back_and_forth_over_most_of_a_lap_never_counts_then_a_real_lap_does(circle_track):
    track = circle_track
    length = track.length
    v = 50.0
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    # Forward 0.9 L, back to -0.05 L, three times, then forward 1.5 L.
    legs = [0.9, -0.05, 0.9, -0.05, 0.9, -0.05, 1.5]
    pos = [0.0]
    for target in legs:
        a = pos[-1]
        b = target * length
        k = int(np.ceil(abs(b - a) / (v * DT)))
        pos.extend(np.linspace(a, b, k + 1)[1:])
    tr.reset(*xy_at(track, pos[0]))
    laps = []
    for k, s in enumerate(pos[1:], start=1):
        e = tr.update(*xy_at(track, s), DT)
        if e.lap_completed[0]:
            laps.append((k, e.lap_time[0]))
    assert tr.state.lap[0] == 1
    assert len(laps) == 1
    # The lap is timed from the last forward crossing of the line to the next one.
    assert laps[0][1] == pytest.approx(length / v, abs=0.02)


def test_starting_just_behind_the_line_does_not_count(circle_track):
    track = circle_track
    length = track.length
    v = 60.0
    tr = ProgressTracker(track, 1)
    s_start = length - 3.0
    ev = run(tr, track, lambda t: s_start + v * t, int((3.0 + 1.2 * length) / v / DT))
    done = stacked(ev, "lap_completed")[:, 0]
    assert done.sum() == 1  # the immediate crossing doesn't count, the full lap does
    # ... and that lap is a full line-to-line lap.
    assert stacked(ev, "lap_time")[done][0, 0] == pytest.approx(length / v, abs=1e-6)


def test_reset_mid_lap_first_crossing_starts_the_timed_lap(oval):
    track = oval
    v = 70.0
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    s0 = 0.4 * track.length
    ev = run(tr, track, lambda t: s0 + v * t, int(1.7 * track.length / v / DT))
    assert tr.state.lap[0] == 1
    times = stacked(ev, "lap_time")[:, 0]
    assert times[~np.isnan(times)] == pytest.approx([track.length / v], abs=1e-6)


# ---------------------------------------------------------------------------- sectors


@pytest.mark.parametrize("fixture", ["circle_track", "oval", "small_drop"])
def test_sector_times_sum_to_lap_time(fixture, request):
    track = request.getfixturevalue(fixture)
    v = 41.0
    tr = ProgressTracker(track, 1)
    # 3.5 laps: past the first boundary (0.25-0.4 L), short of the second (0.6-0.75 L).
    ev = run(tr, track, lambda t: v * t, int(3.5 * track.length / v / DT))
    sec = stacked(ev, "sector_completed")[:, 0]
    sec_t = stacked(ev, "sector_time")[:, 0]
    lap_done = stacked(ev, "lap_completed")[:, 0]
    lap_t = stacked(ev, "lap_time")[:, 0]
    assert lap_done.sum() == 3
    starts = np.asarray(track.data.sectors_s)
    lens = np.diff(np.append(starts, track.length))
    assert list(sec[sec >= 0]) == [0, 1, 2] * 3 + [0]
    np.testing.assert_allclose(sec_t[sec >= 0][:9], np.tile(lens / v, 3), atol=1e-6)
    np.testing.assert_array_equal(np.isnan(sec_t), sec < 0)
    # Last sector and the lap finish on the same step; the three sectors add up to the lap.
    np.testing.assert_array_equal(sec[lap_done], 2)
    k_laps = np.flatnonzero(lap_done)
    sec_steps = np.flatnonzero(sec >= 0)
    for lap_i, k in enumerate(k_laps):
        three = sec_steps[3 * lap_i : 3 * lap_i + 3]
        assert three[-1] == k
        assert sec_t[three].sum() == pytest.approx(lap_t[k], abs=1e-9)
    assert tr.state.sector[0] == 1


def test_sector_index_returns_to_zero_after_a_lap(circle_track):
    track = circle_track
    v = 50.0
    tr = ProgressTracker(track, 1)
    tr.reset(*xy_at(track, 0.0))
    seen = []
    for k in range(1, int(1.1 * track.length / v / DT)):
        e = tr.update(*xy_at(track, v * k * DT), DT)
        seen.append(int(tr.state.sector[0]))
        if e.lap_completed[0]:
            assert tr.state.sector[0] == 0
    assert seen[0] == 0 and 1 in seen and 2 in seen and seen[-1] == 0


def test_no_sectors_means_sector_zero_and_no_sector_events(plain_circle):
    track = plain_circle
    tr = ProgressTracker(track, 1)
    ev = run(tr, track, lambda t: 60.0 * t, int(2.2 * track.length / 60.0 / DT))
    assert tr.state.lap[0] == 2
    assert (stacked(ev, "sector_completed") == -1).all()
    assert (tr.state.sector == 0).all()


def test_reset_in_a_later_sector_sets_the_index_and_skips_the_partial_sector(circle_track):
    track = circle_track
    v = 50.0
    tr = ProgressTracker(track, 1)
    s0 = 0.5 * track.length  # inside sector 1 (0.3-0.7 L)
    tr.reset(*xy_at(track, s0))
    assert tr.state.sector[0] == 1
    ev = run(tr, track, lambda t: s0 + v * t, int(0.75 * track.length / v / DT))
    sec = stacked(ev, "sector_completed")[:, 0]
    assert list(sec[sec >= 0]) == [2]  # sector 1 was partial; sector 2 is complete
    sec_t = stacked(ev, "sector_time")[:, 0]
    assert sec_t[sec == 2][0] == pytest.approx(0.3 * track.length / v, abs=1e-6)
    assert tr.state.sector[0] == 0
    assert tr.state.lap[0] == 0


def test_invalid_sectors_rejected(plain_circle):
    data = dataclasses.replace(plain_circle.data, sectors_s=(0.0, 900.0, 800.0))
    with pytest.raises(ValueError, match="sectors_s"):
        ProgressTracker(Track(data), 1)


# ------------------------------------------------------------------------ termination


@pytest.fixture(scope="module")
def straight_s(oval) -> float:
    """An ``s`` in the middle of a straight (constant normals, exact offsets)."""
    k = np.abs(oval.data.curvature)
    i = int(np.argmin(np.where(k < 1e-6, np.abs(oval.data.s - 0.75 * oval.length), np.inf)))
    return float(oval.data.s[i])


@pytest.mark.parametrize(
    ("n", "off"),
    [
        (W_LEFT - 1.0 - 0.02, False),
        (W_LEFT - 1.0 + 0.02, True),
        (-(W_LEFT - 1.0 + 0.5), False),  # past the left threshold, but on the wider right side
        (-(W_RIGHT - 1.0 - 0.02), False),
        (-(W_RIGHT - 1.0 + 0.02), True),
    ],
)
def test_off_track_threshold_on_both_sides(oval, straight_s, n, off):
    track = oval
    s0 = straight_s
    tr = ProgressTracker(track, 1)
    tr.reset(*xy_at(track, s0, 0.0))
    e = tr.update(*xy_at(track, s0 + 0.5, n), DT)
    assert tr.state.n[0] == pytest.approx(n, abs=1e-6)
    assert bool(e.newly_done[0]) is off
    assert tr.state.done_reason[0] == (DoneReason.OFF_TRACK if off else DoneReason.NONE)


def test_off_track_uses_car_half_width_param(oval, straight_s):
    tr = ProgressTracker(oval, 1, ProgressParams(car_half_width_m=0.0))
    tr.reset(*xy_at(oval, straight_s))
    tr.update(*xy_at(oval, straight_s + 0.5, W_LEFT - 0.02), DT)
    assert not tr.state.done[0]
    tr.update(*xy_at(oval, straight_s + 1.0, W_LEFT + 0.02), DT)
    assert tr.state.done_reason[0] == DoneReason.OFF_TRACK


def test_wrong_way_triggers_after_the_time_and_not_before(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1)
    ev = run(tr, track, lambda t: 100.0 - 5.0 * t, 70)
    newly = stacked(ev, "newly_done")[:, 0]
    assert np.flatnonzero(newly).tolist() == [59]  # step 60 = exactly 1.0 s
    assert tr.state.done_reason[0] == DoneReason.WRONG_WAY
    assert tr.state.t[0] == pytest.approx(1.0)


def test_wrong_way_timer_restarts_when_going_forward(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1, ProgressParams(stall_time_s=1e9))

    # 0.9 s backwards, 0.1 s forwards, repeated: never 1 s in a row.
    def s_of_t(t):
        cyc, ph = divmod(t, 1.0)
        return 300.0 + cyc * (-4.5 + 0.5) + (-5.0 * ph if ph <= 0.9 else -4.5 + 5.0 * (ph - 0.9))

    run(tr, track, s_of_t, 300)
    assert not tr.state.done[0]
    # Slow reversing above the -1 m/s threshold is not wrong-way either.
    run(tr, track, lambda t: 300.0 - 0.9 * t, 300)
    assert not tr.state.done[0]


def test_stall_triggers_for_a_parked_car(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1)
    ev = run(tr, track, lambda t: 50.0, 400)
    newly = stacked(ev, "newly_done")[:, 0]
    assert np.flatnonzero(newly).tolist() == [299]  # exactly stall_time_s after the reset
    assert tr.state.done_reason[0] == DoneReason.STALL


def test_stall_triggers_for_a_crawling_car_and_after_a_stop(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1)
    run(tr, track, lambda t: 50.0 + 0.5 * t, 400)  # 2.5 m per 5 s
    assert tr.state.done_reason[0] == DoneReason.STALL
    assert tr.state.t[0] == pytest.approx(5.0)

    # Drive 10 s at 30 m/s, then stop: flagged within stall_time_s of stopping.
    run(tr, track, lambda t: 30.0 * min(t, 10.0), int(20 / DT))
    assert tr.state.done_reason[0] == DoneReason.STALL
    assert 10.0 + 4.0 <= tr.state.t[0] <= 10.0 + 5.0 + 1e-9


def test_no_stall_for_a_slow_but_moving_car(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 1)
    run(tr, track, lambda t: 1.2 * t, int(30 / DT))  # 6 m per 5 s
    assert not tr.state.done[0]


# --------------------------------------------------------------------- state handling


def test_done_cars_stay_frozen(oval, straight_s):
    track = oval
    tr = ProgressTracker(track, 2)
    tr.reset(*xy_at(track, [straight_s, straight_s]))
    e = tr.update(*xy_at(track, [straight_s + 1.0] * 2, [10.0, 0.0]), DT)
    assert e.newly_done.tolist() == [True, False]
    frozen = {f.name: getattr(tr.state, f.name)[0].copy() for f in dataclasses.fields(tr.state)}
    for k in range(2, 200):
        e = tr.update(*xy_at(track, [straight_s + 50.0 * k * DT] * 2, [0.0, 0.0]), DT)
        assert not e.newly_done[0]
        assert not e.lap_completed[0] and e.sector_completed[0] == -1
    for name, val in frozen.items():
        np.testing.assert_array_equal(getattr(tr.state, name)[0], val, err_msg=name)
    assert tr.state.done_reason[0] == DoneReason.OFF_TRACK
    assert tr.state.t[1] == pytest.approx(199 * DT)


def test_masked_reset_touches_only_masked_cars(circle_track):
    track = circle_track
    tr = ProgressTracker(track, 3)
    v = np.array([50.0, -5.0, 60.0])
    tr.reset(*xy_at(track, np.zeros(3)))
    for k in range(1, int(1.2 * track.length / 50.0 / DT)):
        tr.update(*xy_at(track, v * k * DT), DT)
    assert tr.state.done.tolist() == [False, True, False]
    assert tr.state.lap[0] == 1 and tr.state.lap[2] == 1
    before = {f.name: getattr(tr.state, f.name).copy() for f in dataclasses.fields(tr.state)}
    x, y = xy_at(track, [123.0, 456.0, 789.0])
    tr.reset(x, y, mask=np.array([False, True, False]))
    st = tr.state
    for name, val in before.items():
        np.testing.assert_array_equal(getattr(st, name)[[0, 2]], val[[0, 2]], err_msg=name)
    assert st.s[1] == pytest.approx(456.0, abs=1e-6)
    assert not st.done[1] and st.done_reason[1] == DoneReason.NONE
    assert st.lap[1] == 0 and st.t[1] == 0.0 and st.progress[1] == 0.0
    assert np.isnan(st.last_lap_time[1]) and np.isnan(st.best_lap_time[1])
    assert st.sector[1] == 1  # 456 m is in sector 1 (0.3-0.7 L)


def test_vectorised_matches_individual_cars(small_drop):
    track = small_drop
    rng = np.random.default_rng(7)
    n_cars = 256
    dt = 1.0 / 15.0  # coarse steps (4.7 m at 70 m/s) keep the 256 single-car runs quick
    steps = 120
    v = rng.uniform(-8.0, 70.0, n_cars)
    v[:8] = 0.0
    amp = rng.uniform(0.0, 7.0, n_cars)
    freq = rng.uniform(0.05, 0.5, n_cars)
    s0 = rng.uniform(0.0, track.length, n_cars)
    s0[:32] = 0.0
    t = np.arange(steps + 1)[:, None] * dt
    s_path = s0 + v * t
    n_path = amp * np.sin(2 * np.pi * freq * t)
    xs, ys = xy_at(track, s_path.ravel(), n_path.ravel())
    xs, ys = xs.reshape(s_path.shape), ys.reshape(s_path.shape)

    names = [f.name for f in dataclasses.fields(ProgressEvents)]

    def run_cars(cols):
        tr = ProgressTracker(track, len(cols))
        tr.reset(xs[0, cols], ys[0, cols])
        out = {name: [] for name in names}
        for k in range(1, steps + 1):
            e = tr.update(xs[k, cols], ys[k, cols], dt)
            for name in names:
                out[name].append(getattr(e, name))
        return tr.state, {name: np.array(v) for name, v in out.items()}  # (steps, cars)

    all_state, all_ev = run_cars(np.arange(n_cars))
    assert all_state.done.any() and not all_state.done.all()
    assert (all_state.lap > 0).any()
    assert (all_ev["sector_completed"] >= 0).any()
    one = [run_cars(np.array([i])) for i in range(n_cars)]
    for f in dataclasses.fields(ProgressState):
        got = np.concatenate([getattr(st, f.name) for st, _ in one])
        np.testing.assert_array_equal(got, getattr(all_state, f.name), err_msg=f.name)
    for name in names:
        got = np.concatenate([ev[name] for _, ev in one], axis=1)
        np.testing.assert_array_equal(got, all_ev[name], err_msg=name)


@pytest.mark.parametrize("jump", [30.0, 300.0, -400.0])
def test_jumps_are_reprojected_globally(circle_track, jump):
    track = circle_track
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    tr.reset(*xy_at(track, 100.0))
    tr.update(*xy_at(track, 101.0), DT)
    tr.update(*xy_at(track, 101.0 + jump), DT)
    expected = (101.0 + jump) % track.length
    assert tr.state.s[0] == pytest.approx(expected, abs=1e-6)
    assert tr.state.progress[0] == pytest.approx(1.0 + jump, abs=1e-6)
    # And the hint is good again: plain driving continues from there.
    tr.update(*xy_at(track, 102.0 + jump), DT)
    assert tr.state.s[0] == pytest.approx((102.0 + jump) % track.length, abs=1e-6)


def test_jump_across_line_and_sector_boundary_in_one_step(circle_track):
    track = circle_track
    length = track.length
    b1 = track.data.sectors_s[1]
    tr = ProgressTracker(track, 1, NO_TERMINATION)
    run(tr, track, lambda t: 50.0 * t, int((length - 20.0) / 50.0 / DT))
    assert tr.state.lap[0] == 0 and tr.state.sector[0] == 2
    # Teleport from just before the line to past the first sector boundary: the crossings
    # are handled in order (line first, then the boundary), so the lap counts and the car
    # ends up in sector 1. Only one sector event fits in a step: the later one, sector 0.
    e = tr.update(*xy_at(track, b1 + 10.0), DT)
    assert e.lap_completed[0]
    assert tr.state.lap[0] == 1
    assert tr.state.sector[0] == 1
    assert e.sector_completed[0] == 0
    # Crossing times are interpolated along the jump: lap end before the sector-0 end.
    assert 0.0 < e.sector_time[0] < DT


def test_update_before_reset_and_bad_inputs(circle_track):
    tr = ProgressTracker(circle_track, 2)
    x, y = xy_at(circle_track, [0.0, 1.0])
    with pytest.raises(RuntimeError, match="reset"):
        tr.update(x, y, DT)
    tr.reset(x, y)
    with pytest.raises(ValueError, match="dt"):
        tr.update(x, y, 0.0)
    with pytest.raises(ValueError, match="shape"):
        tr.update(x[:1], y[:1], DT)
    with pytest.raises(ValueError, match="mask"):
        tr.reset(x, y, mask=[True])


def test_events_empty_and_all_done(circle_track):
    ev = ProgressEvents.empty(3)
    assert ev.sector_completed.tolist() == [-1, -1, -1]
    tr = ProgressTracker(circle_track, 1)
    x, y = xy_at(circle_track, 0.0, 20.0)
    tr.reset(x, y)
    assert tr.update(x, y, DT).newly_done[0]
    e = tr.update(x, y, DT)
    assert not e.newly_done[0] and tr.state.done[0]


def test_params_validation():
    with pytest.raises(ValueError):
        ProgressParams(lap_min_fraction=0.0)
    with pytest.raises(ValueError):
        ProgressParams(window=0)
