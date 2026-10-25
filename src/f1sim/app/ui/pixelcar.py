"""Top-down pixel-art F1 cars, generated at runtime from a livery.

:data:`CAR_ART` is a 2026-rules car as a 16 x 40 character map, nose up, mirror-symmetric:
a three-element front wing with the nose tip on it, narrow front tyres with yellow sidewalls,
the halo round a small cockpit, undercut sidepods tapering into a coke-bottle floor, the
airbox, wide rear tyres, the diffuser, a two-element rear wing with endplates and the rain
light, with one-pixel highlight (spine) and shadow (flanks) lines for shape.
:func:`car_pixels` paints it in a :class:`~f1sim.app.ui.theme.Livery` (pure numpy),
optionally with a 1 px light outline (ghosts), and :func:`sprite_scale` picks the on-screen
size: the car's real 5.6 m, snapped to whole physical pixels per art pixel, with a minimum so
it stays readable zoomed out (:func:`fixed_scale` instead keeps one size at every zoom, for
the training view).

:class:`CarSprites` (arcade, created with a window) turns the pixels into cached textures
and draws all cars as one ``SpriteList`` with nearest-neighbour sampling, so the pixels stay
crisp at any integer scale and rotation.
"""

from __future__ import annotations

import math

import numpy as np

from f1sim.app.ui import theme
from f1sim.app.ui.layout import Rect, stack_labels

LABEL_H = 12.0
"""Height of a car label's box (10 px caps plus spacing) when stacking overlapping labels."""

# Legend: . clear, T tyre, t tyre highlight (tread), Y tyre sidewall (Pirelli yellow),
# D carbon (suspension, gearbox), F floor, K cockpit, H halo, P primary, h primary highlight,
# p primary shadow, W wing (secondary), s secondary shadow, S secondary, E endplate (accent),
# A accent, L rain light. Each row is the left half; the right half is its mirror image.
_CAR_HALF: tuple[str, ...] = (
    "EWWWWWWP",  # 0  front wing: slats
    "EsssssPP",  # 1
    "EPPPPPPP",  # 2  main plane; nose tip
    ".......P",  # 3  nose
    ".TTT...P",  # 4  front tyre
    ".YtT...h",  # 5  (yellow sidewall outside, tread highlight)
    ".YtTDDDh",  # 6  front suspension
    ".YtT..ph",  # 7
    ".YtTDDph",  # 8
    ".YtT..ph",  # 9
    ".TTT..Ph",  # 10
    ".....pPh",  # 11
    ".....pPA",  # 12 number
    "....pPPA",  # 13
    "....pPPH",  # 14 halo pillar
    "...pPPHH",  # 15
    "...pPPHK",  # 16 cockpit
    "..SpPPHK",  # 17 sidepod inlets
    ".SSpPPHK",  # 18
    "SSPPPPHH",  # 19 halo hoop
    "SPPPPpPK",  # 20 airbox
    "SPhPPpPh",  # 21 sidepods
    "SPhPPpPh",  # 22
    "SPhPPpPh",  # 23
    "FsPPPpPh",  # 24 undercut, floor
    "FFsPPpPh",  # 25
    "FFFsPpPh",  # 26
    "FFFFspPh",  # 27 coke bottle
    ".FFF.pPA",  # 28
    "TTTT.pPh",  # 29 rear tyre
    "YttT.pPh",  # 30
    "YttTDDPh",  # 31 rear suspension
    "YttT.pph",  # 32
    "YttTDDDD",  # 33 gearbox
    "TTTT.FDD",  # 34
    "...FFFFF",  # 35 diffuser
    ".EWWWWWW",  # 36 rear wing
    ".EPPPPPP",  # 37
    ".Essssss",  # 38 flap
    ".E.....L",  # 39 rain light
)
CAR_ART: tuple[str, ...] = tuple(h + h[::-1] for h in _CAR_HALF)
ART_W = len(CAR_ART[0])
ART_H = len(CAR_ART)

TYRE = (16, 16, 18, 255)
TYRE_HI = (58, 59, 64, 255)
CARBON = (30, 31, 35, 255)
COCKPIT = (6, 6, 8, 255)
HALO = (44, 45, 50, 255)
SIDEWALL = (232, 196, 40, 255)
FLOOR = (22, 23, 26, 255)
RAIN_LIGHT = (255, 40, 40, 255)


def shade(colour, amount: float) -> tuple[int, int, int, int]:
    """Lighten (``amount > 0``, towards white) or darken (``< 0``, towards black) by
    ``|amount|`` (0-1); alpha unchanged."""
    a = min(max(float(amount), -1.0), 1.0)
    target = 255.0 if a > 0 else 0.0
    rgb = [int(round(c + (target - c) * abs(a))) for c in colour[:3]]
    return (rgb[0], rgb[1], rgb[2], colour[3] if len(colour) > 3 else 255)


def palette(livery: theme.Livery) -> dict[str, tuple[int, int, int, int]]:
    """Colour of each :data:`CAR_ART` character for ``livery`` (``"."`` is transparent)."""
    return {
        ".": (0, 0, 0, 0),
        "T": TYRE,
        "t": TYRE_HI,
        "Y": SIDEWALL,
        "D": CARBON,
        "F": FLOOR,
        "L": RAIN_LIGHT,
        "K": COCKPIT,
        "H": HALO,
        "P": tuple(livery.primary),
        "h": shade(livery.primary, 0.32),
        "p": shade(livery.primary, -0.30),
        "W": tuple(livery.secondary),
        "s": shade(livery.secondary, -0.30),
        "S": tuple(livery.secondary),
        "E": tuple(livery.accent),
        "A": tuple(livery.accent),
    }


def car_pixels(livery: theme.Livery, outline=None) -> np.ndarray:
    """RGBA pixels ``(ART_H, ART_W, 4)`` uint8 of the car in ``livery``, nose up (row 0 is
    the front wing). With an ``outline`` colour the image grows by one pixel on every side
    and every clear pixel touching the car (8-neighbourhood) takes that colour."""
    pal = palette(livery)
    img = np.array([[pal[ch] for ch in row] for row in CAR_ART], dtype=np.uint8)
    if outline is None:
        return img
    out = np.zeros((ART_H + 2, ART_W + 2, 4), dtype=np.uint8)
    out[1:-1, 1:-1] = img
    # Outline the car's envelope: each row filled from its first to its last solid pixel, so
    # the ring follows the outer shape and the gaps between tyres and body stay see-through.
    solid = out[..., 3] > 0
    cols = np.arange(solid.shape[1])
    envelope = np.zeros_like(solid)
    for r in np.flatnonzero(solid.any(axis=1)):
        c = cols[solid[r]]
        envelope[r, c[0] : c[-1] + 1] = True
    ring = _dilate4(envelope) & ~envelope
    out[ring] = np.array(outline, dtype=np.uint8)
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """``mask`` grown by one pixel up, down, left and right (nothing wraps round)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def sprite_scale(
    pixels_per_metre: float,
    pixel_ratio: float = 1.0,
    length_m: float = theme.CAR_LENGTH_M,
    min_length_px: float = theme.CAR_MIN_LENGTH_PX,
) -> int:
    """Physical screen pixels per art pixel: the car's real length (``length_m`` at
    ``pixels_per_metre`` logical px/m, ``pixel_ratio`` physical px per logical px) rounded to
    a whole multiple of the art, but at least ``min_length_px`` logical pixels long."""
    world = length_m * pixels_per_metre * pixel_ratio / ART_H
    k_min = math.ceil(min_length_px * pixel_ratio / ART_H - 1e-9)
    return max(int(round(world)), k_min, 1)


def fixed_scale(length_px: float, pixel_ratio: float = 1.0) -> int:
    """Physical screen pixels per art pixel for a car about ``length_px`` logical pixels long
    at any zoom (a whole multiple of the art, at least 1)."""
    return max(int(round(length_px * pixel_ratio / ART_H)), 1)


def label_offset(heading: float, length_px: float, width_px: float) -> float:
    """Half the screen-space width of the rotated car's bounding box: how far right of its
    centre a label can start without touching it."""
    return 0.5 * (length_px * abs(math.cos(heading)) + width_px * abs(math.sin(heading)))


# ------------------------------------------------------------------------- arcade side

_textures: dict[tuple, object] = {}


def car_texture(livery: theme.Livery, outlined: bool):
    """An ``arcade.Texture`` of the car with the nose pointing right (+x), cached per
    livery."""
    key = (livery, outlined)
    tex = _textures.get(key)
    if tex is None:
        import arcade
        from PIL import Image

        px = car_pixels(livery, theme.CAR_OUTLINE if outlined else None)
        px = np.ascontiguousarray(np.rot90(px, k=-1))  # nose up -> nose right
        colours = (livery.primary, livery.secondary, livery.accent)
        name = "-".join(f"{c:02x}" for col in colours for c in col)
        tex = arcade.Texture(Image.fromarray(px, "RGBA"), hash=f"f1sim-car-{name}-{outlined}")
        _textures[key] = tex
    return tex


class CarSprites:
    """All cars of a viewer as one ``SpriteList``, drawn with nearest-neighbour filtering."""

    def __init__(self, liveries: list[theme.Livery], outlined: list[bool]):
        import arcade

        self.sprites = [
            arcade.Sprite(car_texture(lv, ol)) for lv, ol in zip(liveries, outlined, strict=True)
        ]
        self.list = arcade.SpriteList()
        self._order: tuple[int, ...] = ()
        self.set_order(list(range(len(self.sprites))))

    def set_order(self, order: list[int]) -> None:
        """Draw order, back to front (the self car last)."""
        if tuple(order) == self._order:
            return
        self.list.clear()
        for i in order:
            self.list.append(self.sprites[i])
        self._order = tuple(order)

    def place(self, i: int, sx: float, sy: float, heading: float, k: int, ratio: float) -> None:
        """Car ``i`` centred on screen ``(sx, sy)`` (window pixels) at ``k`` physical pixels
        per art pixel. The centre snaps to the physical pixel grid, so an axis-aligned car's
        pixels land exactly on screen pixels."""
        sp = self.sprites[i]
        sp.scale = k / ratio
        sp.position = (round(sx * ratio) / ratio, round(sy * ratio) / ratio)
        sp.angle = -math.degrees(heading)

    def draw(self) -> None:
        self.list.draw(pixelated=True)


class CarLayer:
    """The cars of a viewer with their micro labels, drawn in screen space (under the pixel
    camera) at the sprite size :func:`sprite_scale` picks for the zoom. With
    ``only_highlight_label`` only the highlighted car is labelled (a whole field of labels
    would bury the track); with ``fixed_length_px`` the cars keep that length on screen at
    every zoom (:func:`fixed_scale`)."""

    def __init__(
        self,
        labels: list[str],
        liveries: list[theme.Livery],
        outlined: list[bool],
        fixed_length_px: float | None = None,
    ):
        self.sprites = CarSprites(liveries, outlined)
        self.labels = [theme.micro(label, theme.CAR_LABEL) for label in labels]
        self.only_highlight_label = False
        self.fixed_length_px = fixed_length_px

    def draw(self, camera, poses, highlight: int | None, pixel_ratio: float) -> None:
        """``poses``: ``(x, y, heading)`` per car in world metres; the ``highlight`` car is
        drawn on top with a red label."""
        if self.fixed_length_px is None:
            k = sprite_scale(camera.pixels_per_metre, pixel_ratio)
        else:
            k = fixed_scale(self.fixed_length_px, pixel_ratio)
        length = ART_H * k / pixel_ratio
        width = ART_W * k / pixel_ratio
        n = len(poses)
        order = [i for i in range(n) if i != highlight] + (
            [highlight] if highlight is not None else []
        )
        self.sprites.set_order(order)
        xs, ys = camera.world_to_screen([p[0] for p in poses], [p[1] for p in poses])
        for i, (_, _, heading) in enumerate(poses):
            self.sprites.place(i, float(xs[i]), float(ys[i]), heading, k, pixel_ratio)
        self.sprites.draw()
        labelled = range(n)
        if self.only_highlight_label:
            labelled = order = [] if highlight is None else [highlight]
        # Labels right of each car, caps centred on it; overlapping ones stack downwards
        # (the highlighted car's label keeps its place).
        boxes: dict[int, Rect] = {}
        for i in labelled:
            dx = label_offset(poses[i][2], length, width) + 6.0
            x, y = float(xs[i]) + dx, float(ys[i]) - 3.5
            boxes[i] = Rect(x, y - 2.0, self.labels[i].content_width, LABEL_H)
        first = [highlight] if highlight is not None else []
        rest = sorted((i for i in boxes if i != highlight), key=lambda i: -boxes[i].y)
        placing = first + rest
        stacked = dict(
            zip(
                placing,
                stack_labels([boxes[i] for i in placing], list(range(len(placing))), LABEL_H),
                strict=True,
            )
        )
        for i in order:
            text = self.labels[i]
            colour = theme.ACCENT if i == highlight else theme.CAR_LABEL
            if tuple(text.color) != tuple(colour):
                text.color = colour
            pos = (round(boxes[i].x), round(stacked[i] + 2.0))
            if (text.x, text.y) != pos:
                text.position = pos
            text.draw()
