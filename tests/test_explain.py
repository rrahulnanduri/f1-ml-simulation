"""The "How it works" drawer: its content (f1sim.app.explain) and geometry, no window."""

import math

import pytest

from f1sim.app import explain as ex
from f1sim.app.ui import drawer as dr
from f1sim.app.ui import layout as lay
from f1sim.sim.car_model import CarParams


def _text(page: ex.Page) -> str:
    out = []
    for b in page.blocks:
        out.append(getattr(b, "text", "") or f"{getattr(b, 'name', '')} {getattr(b, 'value', '')}")
        if isinstance(b, ex.Bar):
            out.append(b.label)
    return " ".join(out)


LIVE = ex.Facts(
    gen=86,
    stage=2,
    population=64,
    sigma=0.0005,
    best_index=0,
    best_lap_s=85.411,
    best_progress_m=5887.0,
    episode_s=259.3,
    track_length_m=5887.0,
    run_best_s=85.411,
    t_ref_s=86.433,
    pole_s=85.819,
    pole_label="Russell, 2024 pole",
)


def test_four_pages_with_tabs():
    pages = ex.pages(LIVE)
    assert len(pages) == len(ex.PAGE_TABS) == 4
    assert all(p.title and p.blocks for p in pages)


def test_pages_quote_this_run():
    learn, score, results, physics = ex.pages(LIVE)
    assert "generation 86, 64 cars" in _text(learn)
    # The best car's score worked out, matching the trainer's formula
    want = ex.lap_score(5887.0, 259.3, 85.411)
    assert f"{want:,.0f} points" in _text(score)
    assert want == pytest.approx(5887.0 + 1000.0 + 50.0 * (259.3 - 85.411))
    assert "0.408 s faster than the real lap (-0.5%)" in _text(results)
    bars = [b for b in results.blocks if isinstance(b, ex.Bar)]
    assert [b.lap_s for b in bars] == [85.411, 85.819, 86.433] and bars[0].highlight
    assert "750 kW / 800 kg" in _text(physics)


def test_pages_before_any_generation():
    facts = ex.Facts(copying=True)
    learn, score, results, _ = ex.pages(facts)
    assert "copying the scripted driver" in _text(learn)
    assert "No completed lap yet" in _text(results)
    assert not any(isinstance(b, ex.Bar) for b in results.blocks)
    assert ex.opening_page(facts) == 0
    no_pole = ex.pages(ex.Facts(gen=3, run_best_s=90.0))[2]
    assert "f1sim ghost" in _text(no_pole)


def test_crashed_best_car_in_stage_1():
    facts = ex.Facts(
        gen=2, best_index=4, best_progress_m=812.0, best_crashed=True, track_length_m=5000.0
    )
    assert "Car 5, drove 812 m, less 10% for crashing" in _text(ex.pages(facts)[1])


def test_opening_page():
    assert ex.opening_page(LIVE) == 2  # a lap to compare: the results
    assert ex.opening_page(ex.Facts(gen=1)) == 1  # no lap yet: how scoring works


def test_physics_numbers():
    car = CarParams()
    v = ex.top_speed_mps(car)
    assert car.power_w / (car.mass_kg * v) == pytest.approx(car.drag_coeff * v * v)
    assert 320 < v * 3.6 < 340
    assert ex.grip_g(car, 0.0) == car.grip_base_g
    assert ex.pct(85.0, 85.0 * 1.01) == "-1.0%"
    assert ex.fmt_lap(85.411) == "1:25.411" and ex.fmt_lap(math.nan) == "-:--.---"


def test_reference_poles_are_only_checked_ones():
    assert ex.REFERENCE_POLES == {"silverstone": (85.819, "Russell, 2024 pole")}


# -------------------------------------------------------------------------------- drawer


def test_drawer_slides_in_from_the_right_edge():
    shut, half, out = (dr.drawer_rect(1600, 900, f) for f in (0.0, 0.5, 1.0))
    assert shut.x >= 1600 and out.right == pytest.approx(1600 - lay.MARGIN)
    assert out.x < half.x < shut.x and out.w == shut.w == dr.WIDTH
    assert out.y == lay.BOTTOM_Y and out.top == pytest.approx(900 - lay.MARGIN)
    assert dr.drawer_rect(400, 600, 1.0).w < dr.WIDTH  # narrower on a small window
    # The handle peeks in from the right edge when the drawer is shut
    h = dr.handle_rect(shut, 1600)
    assert 0 < h.x < h.right < 1600
    assert dr.handle_rect(out, 1600).right < out.x


def test_slide_eases_both_ways():
    s = dr.Slide()
    assert s.frac(0.0) == 0.0
    s.set(True, 10.0)
    assert 0.0 < s.frac(10.0 + dr.ANIM_S / 2) < 1.0 and s.frac(11.0) == 1.0
    s.set(False, 20.0)
    s.set(True, 20.0 + dr.ANIM_S / 2)  # turned round mid-slide: no jump
    mid = s.frac(20.0 + dr.ANIM_S / 2)
    assert 0.0 < mid < 1.0
    assert s.frac(30.0) == 1.0


def test_wrap_words():
    measure = len  # one unit per character
    assert dr.wrap_words("aa bb cc dd", 5, measure) == ["aa bb", "cc dd"]
    assert dr.wrap_words("aaaaaaa bb", 5, measure) == ["aaaaaaa", "bb"]  # too long: own line
    assert dr.wrap_words("", 5, measure) == []
    lines = dr.wrap_words("one two three four five six seven", 9, measure)
    assert all(len(ln) <= 9 for ln in lines) and " ".join(lines).split() == (
        "one two three four five six seven".split()
    )


def test_bar_fraction():
    laps = [85.4, 85.8, 86.4]
    assert dr.bar_fraction(86.4, laps) == 1.0
    assert dr.bar_fraction(85.4, laps) == pytest.approx(0.3)
    assert dr.bar_fraction(85.8, laps) == pytest.approx(0.3 + 0.7 * 0.4)
    assert dr.bar_fraction(90.0, [90.0]) == 1.0
