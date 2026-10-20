import re

import pytest

from f1sim.data import catalog
from f1sim.data.tum_source import TUM_TRACKS

# FastF1 3.7 event-schedule Locations (testing excluded), as fetched on 2026-10-20.
LOCATIONS_2024 = [
    "Sakhir", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami", "Imola", "Monaco",
    "Montréal", "Barcelona", "Spielberg", "Silverstone", "Budapest", "Spa-Francorchamps",
    "Zandvoort", "Monza", "Baku", "Marina Bay", "Austin", "Mexico City", "São Paulo",
    "Las Vegas", "Lusail", "Yas Island",
]  # fmt: skip
LOCATIONS_2025 = [
    "Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah", "Miami Gardens", "Imola", "Monaco",
    "Barcelona", "Montréal", "Spielberg", "Silverstone", "Spa-Francorchamps", "Budapest",
    "Zandvoort", "Monza", "Baku", "Marina Bay", "Austin", "Mexico City", "São Paulo",
    "Las Vegas", "Lusail", "Yas Island",
]  # fmt: skip
# Per revamp-plan section 4.1
EXPECTED_TUM = {
    "sakhir", "melbourne", "suzuka", "shanghai", "catalunya", "montreal", "spielberg",
    "silverstone", "budapest", "spa", "zandvoort", "monza", "austin", "mexico_city",
    "sao_paulo", "yas_marina",
}  # fmt: skip


def test_ids_unique_and_well_formed():
    for key, entry in catalog.CATALOG.items():
        assert key == entry.id
        assert re.fullmatch(r"[a-z][a-z0-9_]*", entry.id)
        assert entry.name
        assert entry.fastf1_locations
        assert isinstance(entry.fastf1_locations, tuple)


def test_tum_names_exist_and_unique():
    tum_names = [e.tum_name for e in catalog.CATALOG.values() if e.tum_name is not None]
    assert len(tum_names) == len(set(tum_names))
    assert set(tum_names) <= set(TUM_TRACKS)


def test_tum_coverage_matches_plan():
    with_tum = {e.id for e in catalog.CATALOG.values() if e.tum_name is not None}
    assert with_tum == EXPECTED_TUM


def test_locations_unique_across_entries():
    seen = {}
    for e in catalog.CATALOG.values():
        for loc in e.fastf1_locations:
            key = catalog._norm(loc)
            assert key not in seen, f"{loc!r} used by {seen.get(key)} and {e.id}"
            seen[key] = e.id


@pytest.mark.parametrize("location", sorted(set(LOCATIONS_2024 + LOCATIONS_2025)))
def test_every_calendar_location_maps(location):
    entry = catalog.entry_for_location(location)
    assert entry is not None
    assert location in entry.fastf1_locations  # exact string, not just a fuzzy match


@pytest.mark.parametrize(
    ("location", "track_id"),
    [
        ("Silverstone", "silverstone"),
        ("  silverstone\n", "silverstone"),
        ("SPA-FRANCORCHAMPS", "spa"),
        ("Montréal", "montreal"),
        ("montreal", "montreal"),
        ("Sao Paulo", "sao_paulo"),
        ("Montréal", "montreal"),  # decomposed accent
        ("Yas Island", "yas_marina"),
        ("Yas Marina", "yas_marina"),
        ("Miami Gardens", "miami"),
        ("Monte Carlo", "monaco"),
    ],
)
def test_entry_for_location(location, track_id):
    assert catalog.entry_for_location(location).id == track_id


@pytest.mark.parametrize("location", ["", "Nowhere", "Silver stone", "Sepang"])
def test_entry_for_unknown_location(location):
    assert catalog.entry_for_location(location) is None


def test_get():
    e = catalog.get("silverstone")
    assert e.tum_name == "Silverstone"
    assert e.fastf1_locations == ("Silverstone",)
    assert e.availability == "tum"
    assert catalog.get("monaco").availability == "fastf1-spread"
    with pytest.raises(KeyError, match="Known ids"):
        catalog.get("nope")


def test_layout_risks_noted():
    for track_id in ("yas_marina", "catalunya", "melbourne", "zandvoort"):
        assert catalog.get(track_id).notes
    assert "DTM" in catalog.get("zandvoort").notes


def test_entries_are_frozen():
    with pytest.raises(AttributeError):
        catalog.get("spa").tum_name = "Monza"


@pytest.mark.network
@pytest.mark.parametrize("year", [2024, 2025])
def test_live_schedule_locations_all_mapped(year):
    from f1sim.data import fastf1_source

    events = fastf1_source.event_schedule(year)
    assert len(events) == 24
    missing = [e.location for e in events if catalog.entry_for_location(e.location) is None]
    assert missing == []
    assert [e.location for e in events] == (LOCATIONS_2024 if year == 2024 else LOCATIONS_2025)
