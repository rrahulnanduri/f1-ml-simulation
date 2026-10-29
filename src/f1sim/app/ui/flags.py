"""Minimalist flags for the menu's track picker (pure Python, no window).

A flag is a :class:`Flag`: stripes (horizontal or vertical, optionally of unequal width) and a
few shapes on top, in unit coordinates (``x`` right, ``y`` up, both 0..1 over a 3:2 box). They
are drawn flat with rectangles and discs only, so they stay crisp at any size and add no
image files. :func:`flag_for` finds the flag of a track by its id.
"""

from __future__ import annotations

from dataclasses import dataclass

ASPECT = 1.5
"""Width over height of every flag."""


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    colour: str


@dataclass(frozen=True)
class Disc:
    x: float
    y: float
    r: float  # of the flag's height
    colour: str


@dataclass(frozen=True)
class Line:
    """A straight stroke between two points, ``width`` of the flag's height thick."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    colour: str


@dataclass(frozen=True)
class Flag:
    stripes: tuple[str, ...]
    vertical: bool = False  # stripes side by side (else stacked)
    weights: tuple[float, ...] = ()  # the stripes' relative widths (default: equal)
    shapes: tuple[Rect | Disc | Line, ...] = ()

    def bands(self) -> list[tuple[float, float, str]]:
        """``(start, end, colour)`` of each stripe along the stripe axis, 0..1 (from the left
        for vertical stripes, from the top for horizontal)."""
        w = self.weights or (1.0,) * len(self.stripes)
        total = sum(w)
        out, at = [], 0.0
        for wi, c in zip(w, self.stripes, strict=True):
            out.append((at / total, (at + wi) / total, c))
            at += wi
        return out


RED, WHITE, BLUE, GREEN = "#d0202e", "#f4f4f4", "#1f3f9a", "#1c8a3c"
YELLOW, BLACK = "#f2c316", "#141414"

_ITALY = Flag((GREEN, WHITE, RED), vertical=True)
_SPAIN = Flag((RED, YELLOW, RED), weights=(1.0, 2.0, 1.0))
_USA = Flag(
    (RED, WHITE, RED, WHITE, RED, WHITE, RED),
    shapes=(Rect(0.0, 4 / 7, 0.45, 1.0, BLUE),),
)

FLAGS: dict[str, Flag] = {
    "austin": _USA,
    "las_vegas": _USA,
    "miami": _USA,
    "baku": Flag(("#2a9fd6", RED, GREEN), shapes=(Disc(0.5, 0.5, 0.14, WHITE),)),
    "budapest": Flag((RED, WHITE, GREEN)),
    "catalunya": _SPAIN,
    "madrid": _SPAIN,
    "imola": _ITALY,
    "monza": _ITALY,
    "jeddah": Flag((GREEN,), shapes=(Rect(0.2, 0.42, 0.8, 0.5, WHITE),)),
    "lusail": Flag((WHITE, "#8a1538"), vertical=True, weights=(0.3, 0.7)),
    "marina_bay": Flag((RED, WHITE), shapes=(Disc(0.2, 0.75, 0.16, WHITE),)),
    "melbourne": Flag(
        ("#12286b",),
        shapes=(Rect(0.0, 0.5, 0.5, 1.0, "#b8202e"), Disc(0.25, 0.75, 0.12, WHITE)),
    ),
    "mexico_city": Flag(
        (GREEN, WHITE, RED), vertical=True, shapes=(Disc(0.5, 0.5, 0.12, "#7b4a22"),)
    ),
    "montreal": Flag(
        (RED, WHITE, RED),
        vertical=True,
        weights=(1.0, 2.0, 1.0),
        shapes=(Disc(0.5, 0.5, 0.17, RED),),
    ),
    "monaco": Flag((RED, WHITE)),
    "sakhir": Flag((WHITE, RED), vertical=True, weights=(0.3, 0.7)),
    "sao_paulo": Flag((GREEN,), shapes=(Disc(0.5, 0.5, 0.3, YELLOW), Disc(0.5, 0.5, 0.17, BLUE))),
    "shanghai": Flag((RED,), shapes=(Disc(0.2, 0.74, 0.15, YELLOW),)),
    "silverstone": Flag(
        (BLUE,),
        shapes=(  # the Union Jack: diagonals, then the cross over them
            Line(0.0, 0.0, 1.0, 1.0, 0.2, WHITE),
            Line(0.0, 1.0, 1.0, 0.0, 0.2, WHITE),
            Line(0.0, 0.0, 1.0, 1.0, 0.07, RED),
            Line(0.0, 1.0, 1.0, 0.0, 0.07, RED),
            Rect(0.0, 0.36, 1.0, 0.64, WHITE),
            Rect(0.4, 0.0, 0.6, 1.0, WHITE),
            Rect(0.0, 0.43, 1.0, 0.57, RED),
            Rect(0.45, 0.0, 0.55, 1.0, RED),
        ),
    ),
    "spa": Flag((BLACK, YELLOW, RED), vertical=True),
    "spielberg": Flag((RED, WHITE, RED)),
    "suzuka": Flag((WHITE,), shapes=(Disc(0.5, 0.5, 0.3, "#c8102e"),)),
    "yas_marina": Flag(
        (WHITE,),
        shapes=(
            Rect(0.25, 0.667, 1.0, 1.0, GREEN),
            Rect(0.25, 0.0, 1.0, 0.333, BLACK),
            Rect(0.0, 0.0, 0.25, 1.0, RED),
        ),
    ),
    "zandvoort": Flag(("#ae1c28", WHITE, "#21468b")),
}


def flag_for(track_id: str) -> Flag | None:
    """The flag of the country ``track_id`` is in, or None for a track with none."""
    return FLAGS.get(track_id)


def fit_scale(width: float, available: float, minimum: float = 0.6) -> float:
    """The scale (``minimum``..1) that makes text ``width`` wide fit in ``available``;
    ``minimum`` if even that is too wide (the caller then cuts the text)."""
    if width <= available or width <= 0.0:
        return 1.0
    return max(available / width, minimum)
