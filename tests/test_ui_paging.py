"""Paging card state machine and easing (pure, times passed in)."""

import pytest

from f1sim.app.ui.paging import ANIM_S, WHEEL_DEBOUNCE_S, Pager, ease_out_cubic


def test_ease_out_cubic():
    assert ease_out_cubic(0.0) == 0.0
    assert ease_out_cubic(1.0) == 1.0
    assert ease_out_cubic(-1.0) == 0.0 and ease_out_cubic(2.0) == 1.0
    assert ease_out_cubic(0.5) == pytest.approx(0.875)  # fast start, gentle stop
    us = [i / 20 for i in range(21)]
    vals = [ease_out_cubic(u) for u in us]
    assert vals == sorted(vals)


def test_click_cycles_and_animates():
    p = Pager(3)
    assert p.page == 0 and p.position(0.0) == 0.0
    p.next(10.0)
    assert p.page == 1
    assert p.animating(10.0)
    assert p.position(10.0) == pytest.approx(0.0)
    mid = p.position(10.0 + ANIM_S / 2)
    assert 0.5 < mid < 1.0  # ease-out: more than half way at half time
    assert p.position(10.0 + ANIM_S) == 1.0
    assert not p.animating(10.0 + ANIM_S)
    p.next(20.0)
    p.next(21.0)
    assert p.page == 0  # wraps from the last page to the first
    assert p.position(21.0 + ANIM_S) == 0.0


def test_interrupted_slide_starts_from_where_it_is():
    p = Pager(3)
    p.go_to(2, 0.0)
    pos = p.position(0.1)
    p.go_to(0, 0.1)
    assert p.position(0.1) == pytest.approx(pos)
    assert p.position(0.1 + ANIM_S) == 0.0


def test_wheel_pages_once_per_notch_with_debounce_and_no_wrap():
    p = Pager(3)
    assert p.wheel(-1, 0.0)  # wheel down -> next
    assert p.page == 1
    assert not p.wheel(-1, 0.1)  # within the debounce
    assert p.page == 1
    assert p.wheel(-1, WHEEL_DEBOUNCE_S + 0.01)
    assert p.page == 2
    assert not p.wheel(-1, 5.0)  # no wrap past the last page
    assert p.page == 2
    assert p.wheel(+1, 6.0)  # wheel up -> previous
    assert p.page == 1
    assert not p.wheel(0, 9.0)


def test_pager_rejects_no_pages_and_clamps():
    with pytest.raises(ValueError):
        Pager(0)
    p = Pager(3, page=7)
    assert p.page == 2
    p.go_to(-4, 0.0)
    assert p.page == 0
