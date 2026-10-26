"""Open the F1 Anatomy showcase in the default browser.

    python f1_anatomy/launch.py         # index page
    python f1_anatomy/launch.py aero    # one figure: chassis, power-unit, aero, drivetrain, corner

Standard library only, so it can be called from any menu without touching its dependencies.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

SITE = Path(__file__).resolve().parent / "site"
FIGURES = ("chassis", "power-unit", "aero", "drivetrain", "corner")


def page(name: str | None = None) -> Path:
    """Path of the built page for a figure, or the index when no name is given."""
    target = SITE / ("index.html" if not name else f"{name}.html")
    if not target.exists():
        raise FileNotFoundError(f"{target} is missing; run `node f1_anatomy/build.mjs` first")
    return target


def open_showcase(name: str | None = None) -> None:
    webbrowser.open(page(name).as_uri())


if __name__ == "__main__":
    open_showcase(sys.argv[1] if len(sys.argv) > 1 else None)
