"""The menu hero as a light stage: path-traced photos of the car, one per softbox position.

``scripts/hero/render_top.py --stage`` photographs the front of the car from above in Cycles
with a softbox on a grid of positions (``stage/light_<row>_<col>.png``), plus ``ambient.png``
(the car at rest under a dim, broad light). The menu blends the four photos around the
cursor (:func:`blend_weights`), so wherever the light is, the car has real soft shadows,
reflections of the softbox in its clear coat and occlusion: photographic, not shaded.

:func:`load` reads a stage folder into a :class:`LightStage`, fades the back edge of the photos
(they stop ``STAGE_LENGTH_M`` behind the nose) over ``FADE_M``, and caches the arrays as
``stage_cache.npz`` while the PNGs are unchanged. Grid positions and pixel coordinates use
the hero's map frame: metres right of the centre line, and down from the top edge.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1sim.app.ui import hero_maps

FADE_M = 0.35
CACHE_NAME = "stage_cache.npz"
CACHE_VERSION = 1


@dataclass(frozen=True)
class LightStage:
    images: np.ndarray  # (rows * cols, H, W, 3) uint8: photo for light (row, col) at row*cols+col
    ambient: np.ndarray  # (H, W, 4) uint8: the car at rest, alpha = coverage (faded at the back)
    xs: tuple[float, ...]  # light columns, metres right of the centre line
    ys: tuple[float, ...]  # light rows, metres down from the top edge
    px_per_m: float
    front_m: float

    @property
    def size_m(self) -> tuple[float, float]:
        h, w = self.ambient.shape[:2]
        return w / self.px_per_m, h / self.px_per_m


def stage_dir() -> Path:
    return hero_maps.render_dir() / "stage"


def _read_rgba(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGBA"))


def load(directory: Path | None = None) -> LightStage | None:
    """The light stage in ``directory`` (default :func:`stage_dir`), or None if it's missing
    or incomplete."""
    d = Path(directory) if directory is not None else stage_dir()
    info_path = d / "stage.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not info.get("complete", False):
        return None
    xs, ys = tuple(info["xs"]), tuple(info["ys"])
    paths = [d / f"light_{iy}_{ix}.png" for iy in range(len(ys)) for ix in range(len(xs))]
    amb_path = d / "ambient.png"
    if not amb_path.is_file() or not all(p.is_file() for p in paths):
        return None
    ppm, front = float(info["px_per_m"]), float(info["margin_m"])
    stamp = max(p.stat().st_mtime for p in [*paths, amb_path, info_path])
    cache = d / CACHE_NAME
    if cache.is_file() and cache.stat().st_mtime >= stamp:
        try:
            with np.load(cache, allow_pickle=False) as z:
                if int(z["version"]) == CACHE_VERSION:
                    return LightStage(z["images"], z["ambient"], xs, ys, ppm, front)
        except (OSError, ValueError, KeyError):
            pass
    ambient = _read_rgba(amb_path).copy()
    rows = ambient.shape[0]
    y = (np.arange(rows) + 0.5) / ppm  # metres down from the top edge
    t = np.clip((y - (rows / ppm - FADE_M)) / FADE_M, 0.0, 1.0)
    fade = 1.0 - t * t * (3.0 - 2.0 * t)
    ambient[..., 3] = (ambient[..., 3] * fade[:, None] + 0.5).astype(np.uint8)
    images = np.stack([_read_rgba(p)[..., :3] for p in paths])
    stage = LightStage(images, ambient, xs, ys, ppm, front)
    try:
        np.savez(cache, version=CACHE_VERSION, images=images, ambient=ambient)
    except OSError:
        pass
    return stage


def blend_weights(xs, ys, lx: float, ly: float) -> tuple[list[tuple[int, int, float]], float]:
    """The photos to blend for a light at ``(lx, ly)`` (map metres): four ``(row, col,
    weight)`` (bilinear on the grid, clamped to its edges) and how far outside the grid the
    light is, metres (0 inside). The menu fades the light out with that distance."""

    def locate(v, grid):
        if v <= grid[0]:
            return 0, 0.0
        if v >= grid[-1]:
            return len(grid) - 2, 1.0
        i = int(np.searchsorted(grid, v, side="right") - 1)
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])

    ix, fx = locate(lx, xs)
    iy, fy = locate(ly, ys)
    out = [
        (iy, ix, (1 - fy) * (1 - fx)),
        (iy, ix + 1, (1 - fy) * fx),
        (iy + 1, ix, fy * (1 - fx)),
        (iy + 1, ix + 1, fy * fx),
    ]
    dx = max(xs[0] - lx, 0.0, lx - xs[-1])
    dy = max(ys[0] - ly, 0.0, ly - ys[-1])
    return out, math.hypot(dx, dy)
