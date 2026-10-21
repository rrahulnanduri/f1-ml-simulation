"""The ticker's scramble/decode state machine: pure, deterministic, no window."""

import pytest

from f1sim.app.ui import scramble as sc


def _same_kinds(a: str, b: str) -> bool:
    """Same length, letters where letters were, digits where digits were, the rest equal."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x.isdigit() != y.isdigit() or x.isalpha() != y.isalpha():
            return False
        if not x.isalnum() and x != y:
            return False
    return True


def test_a_static_value_never_animates():
    s = sc.Scramble("SILVERSTONE")
    assert not s.animating(0.0)
    assert s.text(0.0) == s.text(123.4) == "SILVERSTONE"
    assert not s.set("SILVERSTONE", 5.0)  # unchanged: nothing starts
    assert not s.animating(5.0)


def test_a_change_scrambles_then_settles_after_the_duration():
    s = sc.Scramble("14:32", seed=3)
    assert s.set("14:33", 10.0)
    assert s.animating(10.0)
    start = s.text(10.0)
    assert start != "14:33" and _same_kinds(start, "14:33")
    assert start[2] == ":"  # punctuation never scrambles
    assert s.text(10.0 + sc.DURATION_S) == "14:33"
    assert not s.animating(10.0 + sc.DURATION_S)
    assert sc.DURATION_S == pytest.approx(0.25)


def test_characters_settle_left_to_right():
    s = sc.Scramble("", seed=1)
    s.set("REPLAY 2X", 0.0)
    n = len("REPLAY 2X")
    times = [sc.settle_time(i, n) for i in range(n)]
    assert times == sorted(times) and times[-1] == pytest.approx(sc.DURATION_S)
    assert times[0] >= sc.HOLD * sc.DURATION_S  # everything scrambles for a moment first
    mid = 0.5 * (times[3] + times[4])
    shown = s.text(mid)
    assert shown[:4] == "REPL"  # settled
    assert _same_kinds(shown, "REPLAY 2X")


def test_scrambling_is_deterministic_and_steps_not_every_frame():
    a, b = sc.Scramble("", seed=7), sc.Scramble("", seed=7)
    a.set("21 OCT 2026", 1.0)
    b.set("21 OCT 2026", 1.0)
    assert a.text(1.01) == b.text(1.01)
    # Within one step the random characters hold still (cheap: no change, no re-layout).
    assert a.text(1.0) == a.text(1.0 + 0.9 * sc.STEP_S)
    # And they do change from step to step.
    frames = {a.text(1.0 + k * sc.STEP_S) for k in range(3)}
    assert len(frames) > 1


def test_scramble_char_keeps_the_kind():
    for r in range(50):
        assert sc.scramble_char("7", r).isdigit()
        assert sc.scramble_char("Q", r).isupper()
        assert sc.scramble_char("q", r).islower()
        assert sc.scramble_char("·", r) == "·"
        assert sc.scramble_char("×", r) == "×"


def test_a_line_only_scrambles_the_value_that_changed():
    line = sc.ScrambleLine(3)
    assert line.set(["21 OCT 2026", "14:32", "SILVERSTONE"], 0.0)
    assert line.text(1.0) == "21 OCT 2026 · 14:32 · SILVERSTONE"
    assert not line.set(["21 OCT 2026", "14:32", "SILVERSTONE"], 2.0)
    assert not line.animating(2.0)
    assert line.set(["21 OCT 2026", "14:33", "SILVERSTONE"], 3.0)
    shown = line.text(3.0)
    date, clock, track = shown.split(" · ")
    assert date == "21 OCT 2026" and track == "SILVERSTONE"  # untouched
    assert clock != "14:33" and _same_kinds(clock, "14:33")
    assert line.text(3.0 + sc.DURATION_S) == "21 OCT 2026 · 14:33 · SILVERSTONE"
    with pytest.raises(ValueError):
        line.set(["one"], 4.0)


def test_empty_values_are_left_out():
    line = sc.ScrambleLine(3)
    line.set(["A", "", "C"], 0.0)
    assert line.text(1.0) == "A · C"
