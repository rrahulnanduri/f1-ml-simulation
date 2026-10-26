"""The training view's focus overlay: f1sim.app.focus and the pure parts of the layer."""

import math
from pathlib import Path

import numpy as np
import pytest

from f1sim.app import focus as fo
from f1sim.app.training import ghost_label
from f1sim.app.ui import focus_layer as fl
from f1sim.app.ui import layout as lay

MARKS = np.array([0.0, 1000.0, 2500.0, 4000.0])


def test_fade_eases_in_while_held_and_out_after():
    f = fo.Fade()
    f.update(fo.FADE_IN_S / 2, held=True)
    assert 0.0 < f.value < 1.0
    f.update(1.0, held=True)
    assert f.value == 1.0
    f.update(fo.FADE_OUT_S / 2, held=False)
    assert 0.0 < f.value < 1.0
    f.update(1.0, held=False)
    assert f.value == 0.0


def test_sector_marks():
    np.testing.assert_allclose(
        fo.sector_marks((0.0, 1830.4, 4284.2), 5887.4), [0, 1830.4, 4284.2, 5887.4]
    )
    # Not three sectors: thirds
    np.testing.assert_allclose(fo.sector_marks((), 300.0), [0, 100, 200, 300])


def test_lap_timer_times_the_lap_from_progress():
    # Starts 80 m before the line at 40 m/s and drives at a steady 40 m/s
    timer = fo.LapTimer(start_s=4000.0 - 80.0, start_v=40.0, length=4000.0, marks=MARKS)
    assert timer.lead == pytest.approx(80.0)
    assert timer.lap_clock(0.0) == pytest.approx(-2.0)  # 2 s to the line at the start speed
    dt = 1.0 / 30.0
    for k in range(int(110 / dt) + 1):
        t = k * dt
        timer.feed(t, 40.0 * t)
    assert timer.t_line == pytest.approx(2.0)
    np.testing.assert_allclose(timer.splits, [25.0, 62.5, 100.0])
    np.testing.assert_allclose(timer.sector_times, [25.0, 37.5, 37.5])
    assert timer.lap_time == pytest.approx(100.0)
    assert timer.lap_clock(12.0) == pytest.approx(10.0)
    assert timer.lap_s(80.0 + 500.0) == pytest.approx(500.0)


def test_lap_timer_before_the_end():
    timer = fo.LapTimer(start_s=0.0, start_v=50.0, length=4000.0, marks=MARKS)
    assert timer.t_line == 0.0  # starts on the line
    timer.feed(0.0, 0.0)
    timer.feed(30.0, 1200.0)
    assert timer.sector_times[0] == pytest.approx(25.0)
    assert math.isnan(timer.sector_times[1]) and math.isnan(timer.lap_time)


def test_revealed_and_bar_widths():
    times = (25.0, 37.5, 37.5)
    shown = fo.revealed(times, 30.0)
    assert shown[0] == 25.0 and math.isnan(shown[1]) and math.isnan(shown[2])
    assert fo.revealed(times, 100.0) == times
    assert fo.bar_widths(times, MARKS) == pytest.approx((0.25, 0.375, 0.375))
    # No reference: the sectors' lengths
    assert fo.bar_widths((), MARKS) == pytest.approx((0.25, 0.375, 0.375))


def test_callout():
    widths = (0.25, 0.375, 0.375)
    c = fo.callout(
        "AI",
        40.0,
        1600.0,
        (25.0, math.nan, math.nan),
        math.nan,
        MARKS,
        widths,
        pole_times=(24.5, 37.0, 37.0),
    )
    assert not c.finished and c.clock == 40.0
    assert c.fills == pytest.approx((1.0, 0.4, 0.0))
    assert c.deltas[0] == pytest.approx(0.5) and math.isnan(c.deltas[1])
    done = fo.callout("AI", 120.0, 4100.0, (25.0, 37.5, 37.5), 100.0, MARKS, widths)
    assert done.finished and done.clock == 100.0 and done.fills == (1.0, 1.0, 1.0)
    assert done.deltas is None


def test_card_rect_sides_and_window():
    r, elbow = fl.card_rect(800, 400, 1, 1600, 900)
    assert r.x > 800 and r.y > 400 and elbow == (800 + fl.ELBOW, 400 + fl.ELBOW)
    r, _ = fl.card_rect(800, 400, -1, 1600, 900)
    assert r.right < 800 and r.top < 400
    # Kept inside the window
    r, _ = fl.card_rect(1590, 880, 1, 1600, 900)
    assert r.right <= 1600 - lay.MARGIN and r.top <= 900 - lay.MARGIN
    r, _ = fl.card_rect(5, 5, -1, 1600, 900, h=fl.CARD_H + fl.DELTA_H)
    assert r.x >= lay.MARGIN and r.y >= lay.BOTTOM_Y and r.h == fl.CARD_H + fl.DELTA_H


def test_ghost_label():
    p = Path("2024_silverstone_Q_RUS.npz")
    assert ghost_label(p, {"driver": "RUS"}, "x") == "Russell, 2024 pole"
    assert ghost_label(Path("2023_monza_R_VER.npz"), None, "x") == "Verstappen, 2023 R"
