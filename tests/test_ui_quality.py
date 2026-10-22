"""Rendering quality tiers and the frame-time governor (f1sim.app.ui.quality): offline."""

import pytest

from f1sim.app.ui import quality as q


@pytest.mark.parametrize(
    ("renderer", "setting", "expected"),
    [
        ("NVIDIA GeForce RTX 5070 Laptop GPU/PCIe/SSE2", None, ("high", True)),
        ("AMD Radeon RX 7800 XT", "auto", ("high", True)),
        ("Intel(R) UHD Graphics 620", None, ("medium", True)),
        ("Mesa Intel(R) Iris(R) Xe Graphics", "", ("medium", True)),
        ("llvmpipe (LLVM 15.0.7, 256 bits)", None, ("low", True)),
        ("GDI Generic", None, ("low", True)),
        ("NVIDIA GeForce RTX 5070", "low", ("low", False)),  # forced: no governor
        ("Intel(R) UHD Graphics 620", "HIGH", ("high", False)),
    ],
)
def test_the_starting_tier(renderer, setting, expected):
    assert q.initial_tier(renderer, setting) == expected


def test_tiers_spend_less_as_they_go_down():
    low, medium, high = q.TIERS
    assert [t.name for t in q.TIERS] == ["low", "medium", "high"]
    for worse, better in ((low, medium), (medium, high)):
        assert worse.floor_scale < better.floor_scale
        assert worse.car_lamps <= better.car_lamps and worse.shadow_cars <= better.shadow_cars
        assert worse.msaa <= better.msaa and worse.blur_div >= better.blur_div


def _run(gov, dt, seconds):
    steps = []
    for _ in range(int(seconds / dt)):
        if gov.feed(dt):
            steps.append(gov.tier.name)
    return steps


def test_slow_frames_step_the_tier_down_and_fast_ones_keep_it():
    gov = q.Governor("high")
    assert _run(gov, 1 / 60, 30.0) == []  # 60 fps: fine
    assert _run(gov, 1 / 30, 4.0) == ["medium"]  # 30 fps: down once the window fills
    assert _run(gov, 1 / 30, 30.0) == ["low"]  # still slow: down again, after settling
    assert _run(gov, 1 / 10, 30.0) == []  # nothing below low
    assert gov.tier.name == "low"


def test_the_governor_ignores_start_up_stalls_and_pauses():
    gov = q.Governor("high")
    assert _run(gov, 1 / 30, q.Governor.SETTLE_S - 0.5) == []  # still settling
    gov = q.Governor("high")
    assert _run(gov, 0.5, 60.0) == []  # stalls (loads, drags), not slowness
    gov.paused = True  # minimised
    assert _run(gov, 1 / 20, 60.0) == []
    forced = q.Governor("high", auto=False)
    assert _run(forced, 1 / 20, 60.0) == []


def test_setup_reads_the_environment(monkeypatch):
    monkeypatch.setenv(q.ENV, "medium")
    gov = q.setup("NVIDIA GeForce RTX 5070")
    assert q.current is gov and q.tier().name == "medium" and not gov.auto
    monkeypatch.delenv(q.ENV)
    assert q.setup("NVIDIA GeForce RTX 5070").tier.name == "high"
