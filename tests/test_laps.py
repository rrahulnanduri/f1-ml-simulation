import numpy as np
import pytest

from f1sim.data import laps


@pytest.mark.parametrize(
    ("year", "source", "expected"),
    [
        (2024, "auto", "openf1"),
        (2023, "auto", "openf1"),
        (2022, "auto", "fastf1"),
        (2019, "fastf1", "fastf1"),
        (2025, "fastf1", "fastf1"),
    ],
)
def test_resolve_source(year, source, expected):
    assert laps.resolve_source(year, source) == expected


def test_openf1_before_2023_is_refused():
    with pytest.raises(ValueError, match="no data before 2023"):
        laps.resolve_source(2022, "openf1")


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="Unknown lap source"):
        laps.resolve_source(2024, "ergast")


@pytest.mark.parametrize(("year", "module"), [(2024, "openf1_source"), (2019, "fastf1_source")])
def test_reference_car_data_uses_the_same_source_and_lap(monkeypatch, year, module):
    import importlib

    src = importlib.import_module(f"f1sim.data.{module}")
    calls = []
    sentinel = laps.CarTrace(*(np.zeros(3) for _ in range(5)))
    if module == "openf1_source":
        monkeypatch.setattr(src, "find_session", lambda y, tid, kind: calls.append((y, tid, kind)))
    else:
        monkeypatch.setattr(src, "load_session", lambda y, loc, kind: calls.append((y, loc, kind)))

    def fake_car(session, driver):
        calls.append(driver)
        return sentinel

    monkeypatch.setattr(src, "fastest_lap_car_data", fake_car)
    got = laps.reference_car_data("silverstone", year, kind="Q", driver="HAM")
    assert got is sentinel
    place = "silverstone" if module == "openf1_source" else "Silverstone"
    assert calls == [(year, place, "Q"), "HAM"]


def test_fastf1_source_still_exports_lapxy():
    from f1sim.data import fastf1_source

    assert fastf1_source.LapXY is laps.LapXY
