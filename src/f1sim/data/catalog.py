"""Track catalogue: f1sim track ids <-> TUM file names <-> FastF1 event ``Location`` values.

Covers every circuit on the 2024 and 2025 F1 calendars, plus Madrid from 2026.
``fastf1_locations`` holds the exact ``Location`` strings seen in FastF1's event schedules
for 2024-2026 (checked against FastF1 3.7); a circuit can appear under several names.
``tum_name`` is None where the TUM racetrack-database has no file for the circuit; those
tracks need the FastF1-spread width fallback (revamp-plan section 4.1).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_LAYOUT_CHECK = "The FastF1 alignment check decides per year whether the TUM layout is usable."


@dataclass(frozen=True)
class CatalogEntry:
    """One circuit. ``tum_name`` is a stem in ``tum_source.TUM_TRACKS`` or None."""

    id: str
    name: str
    tum_name: str | None
    fastf1_locations: tuple[str, ...]
    notes: str = ""

    @property
    def availability(self) -> str:
        """Geometry source: ``"tum"`` or ``"fastf1-spread"`` (docs/data-formats.md section 2)."""
        return "tum" if self.tum_name is not None else "fastf1-spread"


_ENTRIES: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "sakhir",
        "Bahrain International Circuit",
        "Sakhir",
        ("Sakhir", "Bahrain"),
        "'Bahrain' is the 2026 pre-season testing Location.",
    ),
    CatalogEntry("jeddah", "Jeddah Corniche Circuit", None, ("Jeddah",)),
    CatalogEntry(
        "melbourne",
        "Albert Park Circuit",
        "Melbourne",
        ("Melbourne",),
        "Reprofiled for 2022 (old T9-T10 chicane removed, several corners widened); the TUM "
        "file likely shows the older layout. " + _LAYOUT_CHECK,
    ),
    CatalogEntry("suzuka", "Suzuka International Racing Course", "Suzuka", ("Suzuka",)),
    CatalogEntry("shanghai", "Shanghai International Circuit", "Shanghai", ("Shanghai",)),
    CatalogEntry(
        "miami",
        "Miami International Autodrome",
        None,
        ("Miami", "Miami Gardens"),
        "Location is 'Miami' in 2024 and 'Miami Gardens' from 2025.",
    ),
    CatalogEntry(
        "imola",
        "Autodromo Enzo e Dino Ferrari",
        None,
        ("Imola",),
        "Not on the 2026 calendar.",
    ),
    CatalogEntry(
        "monaco",
        "Circuit de Monaco",
        None,
        ("Monaco", "Monte Carlo"),
        "Location is 'Monaco' in 2024-2025 and 'Monte Carlo' in 2026.",
    ),
    CatalogEntry("montreal", "Circuit Gilles Villeneuve", "Montreal", ("Montréal",)),
    CatalogEntry(
        "catalunya",
        "Circuit de Barcelona-Catalunya",
        "Catalunya",
        ("Barcelona",),
        "F1 dropped the final-sector chicane for 2023 (back to the fast T13-T14 sweep); the "
        "TUM file may include the chicane. " + _LAYOUT_CHECK,
    ),
    CatalogEntry("spielberg", "Red Bull Ring", "Spielberg", ("Spielberg",)),
    CatalogEntry("silverstone", "Silverstone Circuit", "Silverstone", ("Silverstone",)),
    CatalogEntry("budapest", "Hungaroring", "Budapest", ("Budapest",)),
    CatalogEntry("spa", "Circuit de Spa-Francorchamps", "Spa", ("Spa-Francorchamps",)),
    CatalogEntry(
        "zandvoort",
        "Circuit Zandvoort",
        "Zandvoort",
        ("Zandvoort",),
        "The TUM file is the DTM layout, before the 2020-21 rebuild for F1 (banked T3 and "
        "T14, revised final sector). " + _LAYOUT_CHECK,
    ),
    CatalogEntry("monza", "Autodromo Nazionale Monza", "Monza", ("Monza",)),
    CatalogEntry("baku", "Baku City Circuit", None, ("Baku",)),
    CatalogEntry(
        "marina_bay",
        "Marina Bay Street Circuit",
        None,
        ("Marina Bay",),
        "T16-T19 section removed for 2023.",
    ),
    CatalogEntry("austin", "Circuit of the Americas", "Austin", ("Austin",)),
    CatalogEntry(
        "mexico_city",
        "Autódromo Hermanos Rodríguez",
        "MexicoCity",
        ("Mexico City",),
    ),
    CatalogEntry(
        "sao_paulo",
        "Autódromo José Carlos Pace (Interlagos)",
        "SaoPaulo",
        ("São Paulo",),
    ),
    CatalogEntry("las_vegas", "Las Vegas Strip Circuit", None, ("Las Vegas",)),
    CatalogEntry("lusail", "Lusail International Circuit", None, ("Lusail",)),
    CatalogEntry(
        "yas_marina",
        "Yas Marina Circuit",
        "YasMarina",
        ("Yas Island", "Yas Marina"),
        "Reprofiled for 2021 (T5-T7 chicane and hairpin merged into one hairpin, T11-T14 "
        "chicanes replaced by one banked corner, hotel section opened up); the TUM file likely "
        "shows the pre-2021 layout. Location is 'Yas Island' in "
        "2024-2025 and 'Yas Marina' in 2026. " + _LAYOUT_CHECK,
    ),
    CatalogEntry(
        "madrid",
        "Madring",
        None,
        ("Madrid",),
        "New for 2026 (Spanish Grand Prix); not on the 2024-2025 calendars.",
    ),
)

CATALOG: dict[str, CatalogEntry] = {e.id: e for e in _ENTRIES}


def _norm(location: str) -> str:
    """Case-, accent- and surrounding-whitespace-insensitive key for a Location string."""
    decomposed = unicodedata.normalize("NFKD", location.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


_BY_LOCATION: dict[str, CatalogEntry] = {
    _norm(loc): e for e in _ENTRIES for loc in e.fastf1_locations
}


def get(track_id: str) -> CatalogEntry:
    """Catalogue entry for ``track_id``; KeyError listing the valid ids if unknown."""
    try:
        return CATALOG[track_id]
    except KeyError:
        raise KeyError(
            f"Unknown track id {track_id!r}. Known ids: {', '.join(sorted(CATALOG))}"
        ) from None


def entry_for_location(location: str) -> CatalogEntry | None:
    """Entry whose FastF1 ``Location`` matches (ignoring case, accents and outer whitespace)."""
    return _BY_LOCATION.get(_norm(location))
