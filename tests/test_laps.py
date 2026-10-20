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


def test_fastf1_source_still_exports_lapxy():
    from f1sim.data import fastf1_source

    assert fastf1_source.LapXY is laps.LapXY
