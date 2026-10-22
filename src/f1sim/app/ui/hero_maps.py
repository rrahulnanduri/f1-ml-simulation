"""The menu hero's lighting maps: from top-down renders of a real model, or the drawn car.

:class:`HeroMaps` is what the menu's shader lights:

- ``albedo`` ``(H, W, 4)`` uint8: colour and coverage (alpha).
- ``normal`` ``(H, W, 3)`` uint8: ``128 + 127 n``; x right, y towards the nose, z up.
- ``surface`` ``(H, W, 4)`` uint8: roughness, metallic, height / ``height_scale_m``, and
  ambient occlusion.
- ``coat`` ``(H, W)`` uint8: clear-coat weight (glossy lacquer over the paint).

Row 0 is the nose end. ``px_per_m`` and ``size_m`` place the maps in metres; the car's front
is ``front_m`` below the top edge.

:func:`load_rendered` reads ``scripts/hero/render_top.py``'s output (``albedo``, ``normal``,
``surface``, ``ao`` PNGs and ``meta.json``), keeps the car from the nose to ``FRONT_LENGTH_M``
behind it (the hero never shows more) and fades the cut over ``FADE_M``. It caches the result
next to the renders as ``hero_maps.npz`` and reuses it while the PNGs are unchanged.
:func:`from_drawn` converts the code-drawn car (:mod:`f1sim.app.ui.car_top`), the fallback
when there are no renders.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1sim.app.ui import car_top

FRONT_LENGTH_M = 4.05
FADE_M = 0.35
RENDER_DIR_ENV = "F1SIM_HERO_DIR"
DEFAULT_RENDER_DIR = Path(__file__).resolve().parents[4] / "assets_src" / "sf26" / "render"
"""Where ``scripts/hero/render_top.py`` writes the SF-26 renders (git-ignored)."""
CACHE_NAME = "hero_maps.npz"
CACHE_VERSION = 1
PASSES = ("albedo", "normal", "surface", "ao")


@dataclass(frozen=True)
class HeroMaps:
    albedo: np.ndarray
    normal: np.ndarray
    surface: np.ndarray
    coat: np.ndarray
    px_per_m: float
    front_m: float
    height_scale_m: float
    source: str  # "render" or "drawn"

    @property
    def size_m(self) -> tuple[float, float]:
        """``(width, length)`` covered by the maps, metres."""
        h, w = self.albedo.shape[:2]
        return w / self.px_per_m, h / self.px_per_m


def _to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _read_png(path: Path) -> np.ndarray:
    """An RGBA PNG (8 or 16 bit) as float32 0..1."""
    from PIL import Image

    im = Image.open(path)
    a = np.asarray(im)
    if a.dtype == np.uint16 or a.max() > 255:
        out = a.astype(np.float32) / 65535.0
    else:
        out = a.astype(np.float32) / 255.0
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=-1)
    if out.shape[-1] == 3:
        out = np.concatenate([out, np.ones((*out.shape[:2], 1), np.float32)], axis=-1)
    return out


def _fade(rows: int, px_per_m: float, front_m: float) -> np.ndarray:
    """Per-row alpha: 1 down to ``FRONT_LENGTH_M - FADE_M`` behind the nose, then to 0."""
    y = (np.arange(rows) + 0.5) / px_per_m - front_m
    t = np.clip((y - (FRONT_LENGTH_M - FADE_M)) / FADE_M, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def render_dir() -> Path:
    return Path(os.environ.get(RENDER_DIR_ENV) or DEFAULT_RENDER_DIR)


def load_rendered(directory: Path | None = None) -> HeroMaps | None:
    """The renders in ``directory`` (default :func:`render_dir`) as maps, or None if any
    pass or ``meta.json`` is missing."""
    d = Path(directory) if directory is not None else render_dir()
    pngs = [d / f"{p}.png" for p in PASSES]
    meta_path = d / "meta.json"
    if not meta_path.is_file() or not all(p.is_file() for p in pngs):
        return None
    stamp = max(p.stat().st_mtime for p in [*pngs, meta_path])
    cache = d / CACHE_NAME
    if cache.is_file() and cache.stat().st_mtime >= stamp:
        try:
            with np.load(cache, allow_pickle=False) as z:
                if int(z["version"]) == CACHE_VERSION:
                    return HeroMaps(
                        z["albedo"],
                        z["normal"],
                        z["surface"],
                        z["coat"],
                        float(z["px_per_m"]),
                        float(z["front_m"]),
                        float(z["height_scale_m"]),
                        "render",
                    )
        except (OSError, ValueError, KeyError):
            pass
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ppm = float(meta["px_per_m"])
    front = float(meta["margin_m"])
    rows = int(round((front + FRONT_LENGTH_M) * ppm))
    albedo, normal, surface, ao = (_read_png(p)[:rows] for p in pngs)
    fade = _fade(albedo.shape[0], ppm, front)[:, None]
    alpha = albedo[..., 3] * fade
    albedo = np.concatenate([albedo[..., :3], alpha[..., None]], axis=-1)
    n = normal[..., :3] * 2.0 - 1.0
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    n[alpha <= 0.0] = (0.0, 0.0, 1.0)
    surf = np.concatenate([surface[..., :3], ao[..., :1]], axis=-1)
    maps = HeroMaps(
        albedo=_to_u8(albedo),
        normal=_to_u8(0.5 + 0.5 * n),
        surface=_to_u8(surf),
        coat=_to_u8(ao[..., 1]),
        px_per_m=ppm,
        front_m=front,
        height_scale_m=float(meta.get("height_scale_m", 1.5)),
        source="render",
    )
    try:
        np.savez(
            cache,
            version=CACHE_VERSION,
            albedo=maps.albedo,
            normal=maps.normal,
            surface=maps.surface,
            coat=maps.coat,
            px_per_m=ppm,
            front_m=front,
            height_scale_m=maps.height_scale_m,
        )
    except OSError:
        pass
    return maps


def from_drawn(maps: car_top.CarMaps, height_scale_m: float = 1.2) -> HeroMaps:
    """The code-drawn car as hero maps (rough from its gloss, no metal, no occlusion)."""
    spec, gloss = maps.material[..., 0], maps.material[..., 1]
    surface = np.stack(
        [
            1.0 - 0.9 * gloss,
            np.zeros_like(gloss),
            maps.height / height_scale_m,
            np.ones_like(gloss),
        ],
        axis=-1,
    )
    return HeroMaps(
        albedo=_to_u8(maps.albedo),
        normal=_to_u8(0.5 + 0.5 * maps.normal),
        surface=_to_u8(surface),
        coat=_to_u8(spec * gloss),
        px_per_m=maps.px_per_m,
        front_m=0.0,
        height_scale_m=height_scale_m,
        source="drawn",
    )


def load() -> HeroMaps:
    """The rendered model when its renders exist, else the drawn car."""
    return load_rendered() or from_drawn(car_top.build_maps())
