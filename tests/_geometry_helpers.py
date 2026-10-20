"""Synthetic closed centre lines for the track tests (no data files, no network)."""

from __future__ import annotations

import numpy as np
import shapely


def circle(radius: float, n: int = 200, ccw: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points on a circle centred at the origin, starting at ``(radius, 0)``."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if not ccw:
        t = -t
    return radius * np.cos(t), radius * np.sin(t)


def resample_ring(ring: shapely.LinearRing | shapely.LineString, spacing: float):
    """Points every ~``spacing`` metres along a closed shapely ring, counterclockwise."""
    line = shapely.LineString(ring.coords)
    n = max(8, int(round(line.length / spacing)))
    d = np.arange(n) * (line.length / n)
    pts = np.array([line.interpolate(v).coords[0] for v in d])
    x, y = pts[:, 0], pts[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:  # make it counterclockwise
        x, y = x[::-1].copy(), y[::-1].copy()
    return x, y


def stadium(straight: float, radius: float, spacing: float = 3.0):
    """Two straights of length ``straight`` joined by semicircles of ``radius``; CCW."""
    a = np.linspace(-np.pi / 2, np.pi / 2, 400)
    right = np.column_stack([straight / 2 + radius * np.cos(a), radius * np.sin(a)])
    left = np.column_stack([-straight / 2 - radius * np.cos(a), -radius * np.sin(a)])
    ring = shapely.LinearRing(np.vstack([right, left]))
    return resample_ring(ring, spacing)


def teardrop(r_small: float, r_big: float, dist: float, spacing: float = 2.0):
    """Convex 'belt' around a small and a big circle: a hairpin with diverging legs; CCW."""
    a = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    pts = np.vstack(
        [
            np.column_stack([r_small * np.cos(a), r_small * np.sin(a)]),
            np.column_stack([dist + r_big * np.cos(a), r_big * np.sin(a)]),
        ]
    )
    return resample_ring(shapely.MultiPoint(pts).convex_hull.exterior, spacing)


def figure_eight(scale: float = 100.0, n: int = 300):
    """A lemniscate-like loop that crosses itself at the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return scale * np.sin(t), scale * np.sin(t) * np.cos(t)
