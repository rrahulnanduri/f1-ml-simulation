"""A generic, unbranded top-down F1 car, drawn from code into lighting maps for the menu hero.

The car is modelled as a height field: each part has a plan-view shape (a signed distance in
metres) and a height profile (a dome, a cylinder for the tyres, a tube, a flat plate). Where
parts overlap, the higher one wins, like looking down on the real thing. The result is a set
of maps that :mod:`f1sim.app.menu` lights per pixel with a point light at the cursor:

- ``albedo`` ``(H, W, 4)``: colour and coverage (alpha).
- ``normal`` ``(H, W, 3)``: surface direction in ``-1..1``, from the height field's gradient.
- ``material`` ``(H, W, 2)``: specular strength ``0..1`` and glossiness ``0..1``.
- ``height`` ``(H, W)``: metres above the ground.

The proportions follow the current ground-effect cars (2022 on): a 2.0 m front wing with
swept multi-element flaps, a narrow nose, 720 mm front tyres, halo, sidepods with
forward inlets, and an airbox and engine cover behind the driver. Only the front part is
built, as the hero shows the car from the nose to just behind the driver:
``FRONT_LENGTH_M`` of it, fading out over the last ``FADE_M``.

Car frame: ``x`` to the car's right, ``y`` from the front wing's leading edge towards the
rear (so the nose points up when row 0 is drawn at the top), metres. There are no logos,
numbers or team colours; the only colour besides white, silver and carbon is the app's red
accent on the wing endplates and the nose tip. Pure numpy and scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

PX_PER_M = 480.0
"""Map resolution. The hero draws the car at up to ~450 physical px/m on a 2x display."""
HALF_WIDTH_M = 1.06
FRONT_LENGTH_M = 4.05
FADE_M = 0.35
VERSION = 1
"""Bump when the drawing changes (the maps are cached by version)."""

# Colours (linear-ish 0..1) and materials: (specular strength, glossiness)
WHITE = (0.84, 0.85, 0.87)
SILVER = (0.55, 0.57, 0.60)
CARBON = (0.055, 0.057, 0.062)
CARBON_LIGHT = (0.09, 0.093, 0.10)
TYRE = (0.065, 0.065, 0.07)
COCKPIT = (0.02, 0.02, 0.022)
ACCENT = (0.92, 0.05, 0.25)
VISOR = (0.03, 0.03, 0.035)
PAINT = (0.75, 0.80)
CARBON_MAT = (0.35, 0.55)
RUBBER = (0.16, 0.12)
MATTE = (0.10, 0.25)
GLASS = (0.95, 0.95)

LIGHT_CORE_M = 0.75
"""The cursor light: intensity ``1 / (1 + (r / core)^2)`` with distance ``r`` in plan..."""
LIGHT_REACH_M = 1.8
"""... faded to nothing at this distance."""
LIGHT_HEIGHT_M = 1.6
AMBIENT = 0.012
"""Light everywhere, so the unlit car is just visible."""
DIFFUSE_GAIN = 1.1
TWILL_PERIOD_M = 0.012
"""Carbon weave: diagonal twill stripes this far apart, tilting the normals slightly."""
TWILL_TILT = 0.07

FRONT_AXLE_Y = 1.18
WHEEL_X = 0.84
TYRE_RADIUS = 0.36
TYRE_HALF_WIDTH = 0.15


@dataclass(frozen=True)
class CarMaps:
    albedo: np.ndarray  # (H, W, 4) float32 0..1
    normal: np.ndarray  # (H, W, 3) float32 -1..1
    material: np.ndarray  # (H, W, 2) float32 0..1
    height: np.ndarray  # (H, W) float32, m
    px_per_m: float

    @property
    def size_m(self) -> tuple[float, float]:
        """``(width, length)`` covered by the maps, metres."""
        h, w = self.height.shape
        return w / self.px_per_m, h / self.px_per_m


# ---------------------------------------------------------------------- 2D distances


def _sd_box(x, y, cx, cy, hx, hy, r=0.0):
    """Signed distance to a rounded rectangle (half sizes ``hx``, ``hy``, corner ``r``)."""
    qx = np.abs(x - cx) - (hx - r)
    qy = np.abs(y - cy) - (hy - r)
    out = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
    return out - r


def _sd_segment(x, y, ax, ay, bx, by):
    """Distance to the segment a-b, and the position ``t`` (0..1) along it."""
    px, py = x - ax, y - ay
    dx, dy = bx - ax, by - ay
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - t * dx, py - t * dy), t


def _sd_polyline(x, y, pts):
    d = np.full(np.shape(x), np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:], strict=True):
        d = np.minimum(d, _sd_segment(x, y, ax, ay, bx, by)[0])
    return d


def _taper(y, y0, y1, w0, w1):
    """Half-width that goes linearly from ``w0`` at ``y0`` to ``w1`` at ``y1``."""
    t = np.clip((y - y0) / (y1 - y0), 0.0, 1.0)
    return w0 + (w1 - w0) * t


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- canvas


class _Canvas:
    """The maps being drawn, with a z-buffer: a part paints where it's higher."""

    def __init__(self, px_per_m: float):
        self.ppm = px_per_m
        w = int(round(2 * HALF_WIDTH_M * px_per_m))
        h = int(round(FRONT_LENGTH_M * px_per_m))
        xs = (np.arange(w) + 0.5) / px_per_m - HALF_WIDTH_M
        ys = (np.arange(h) + 0.5) / px_per_m
        self.X, self.Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        self.height = np.zeros((h, w), dtype=np.float32)
        self.cover = np.zeros((h, w), dtype=np.float32)
        self.albedo = np.zeros((h, w, 3), dtype=np.float32)
        self.mat = np.zeros((h, w, 2), dtype=np.float32)

    def window(self, x0, x1, y0, y1):
        """Index slices of the region ``x0..x1, y0..y1`` (metres), for speed."""
        c0 = max(int((x0 + HALF_WIDTH_M) * self.ppm) - 2, 0)
        c1 = min(int((x1 + HALF_WIDTH_M) * self.ppm) + 3, self.X.shape[1])
        r0 = max(int(y0 * self.ppm) - 2, 0)
        r1 = min(int(y1 * self.ppm) + 3, self.X.shape[0])
        return slice(r0, r1), slice(c0, c1)

    def decal(self, sl, sd, dh, colour, material):
        """Recolour what's already drawn inside ``sd <= 0`` and lower or raise it by ``dh``:
        grooves, slots and panel details on top of a part."""
        inside = (sd <= 0.0) & (self.cover[sl] > 0.0)
        if not inside.any():
            return
        self.height[sl][inside] += dh
        self.albedo[sl][inside] = colour
        self.mat[sl][inside] = material

    def paint(self, sl, sd, height, colour, material):
        """Paint inside ``sd <= 0`` (in window ``sl``) where ``height`` beats the buffer.

        ``colour`` is an RGB tuple or an ``(h, w, 3)`` array; edges get a half-pixel of
        coverage so the outline isn't jagged.
        """
        cover = np.clip(0.5 - sd * self.ppm, 0.0, 1.0)
        win = cover > 0.0
        hb = self.height[sl]
        top = win & (height >= hb)
        if not top.any():
            return
        hb[top] = np.asarray(height, dtype=np.float32)[top] if np.ndim(height) else height
        cb = self.cover[sl]
        cb[top] = np.maximum(cb[top], cover[top])
        col = np.broadcast_to(np.asarray(colour, dtype=np.float32), (*sd.shape, 3))
        self.albedo[sl][top] = col[top]
        self.mat[sl][top] = material


# ------------------------------------------------------------------------- the parts


def _front_wing(c: _Canvas) -> None:
    sl = c.window(-1.0, 1.0, 0.0, 0.85)
    X, Y = c.X[sl], c.Y[sl]
    ax = np.abs(X)
    # Centre section under the nose: the mainplane, flat
    sd = _sd_box(X, Y, 0.0, 0.17, 0.30, 0.11, 0.04)
    c.paint(
        sl, sd, 0.08 + 0.01 * np.sin(np.pi * np.clip((Y - 0.06) / 0.22, 0, 1)), CARBON, CARBON_MAT
    )
    # Four stacked flaps outboard, their leading edges swept back towards the tips
    for i in range(4):
        y_le = 0.07 + 0.11 * i + 0.20 * np.clip((ax - 0.22) / 0.76, 0.0, 1.0) ** 1.3
        chord = 0.095 - 0.008 * i
        mid = y_le + 0.5 * chord
        sd = np.maximum(np.abs(Y - mid) - 0.5 * chord, ax - 0.975)
        sd = np.maximum(sd, 0.18 - ax)
        camber = np.sin(np.pi * np.clip((Y - y_le) / chord, 0.0, 1.0))
        h = 0.09 + 0.035 * i + 0.012 * camber
        col = ACCENT if i == 3 else CARBON
        c.paint(sl, sd, h, col if i == 3 else CARBON_LIGHT if i % 2 else CARBON, CARBON_MAT)
    # Endplates (thin, tall) with the red accent on top
    for s in (-1.0, 1.0):
        sd = _sd_box(X, Y, s * 0.985, 0.47, 0.014, 0.33, 0.01)
        c.paint(sl, sd, 0.30, ACCENT, PAINT)


def _nose_and_chassis(c: _Canvas) -> None:
    sl = c.window(-0.4, 0.4, 0.0, 2.3)
    X, Y = c.X[sl], c.Y[sl]
    hw = np.where(
        Y < 1.25,
        0.075 + (0.18 - 0.075) * _smooth01((Y - 0.02) / 1.23) ** 0.8,
        _taper(Y, 1.25, 2.15, 0.18, 0.29),
    )
    sd = np.maximum(np.abs(X) - hw, 0.02 - Y)
    sd = np.minimum(sd, np.hypot(X, Y - 0.08) - 0.075)  # rounded tip
    top = np.where(Y < 1.25, 0.20 + 0.20 * _smooth01(Y / 1.25), _taper(Y, 1.25, 2.15, 0.40, 0.62))
    dome = np.sqrt(np.clip(1.0 - (X / np.maximum(hw, 1e-3)) ** 2, 0.0, 1.0))
    h = 0.10 + (top - 0.10) * (0.35 + 0.65 * dome)
    col = np.where((Y < 0.16)[..., None], np.array(ACCENT, np.float32), np.array(WHITE, np.float32))
    c.paint(sl, sd, h, col, PAINT)


def _front_tyres_and_suspension(c: _Canvas) -> None:
    for s in (-1.0, 1.0):
        sl = c.window(s * WHEEL_X - 0.2, s * WHEEL_X + 0.2, 0.7, 1.65)
        X, Y = c.X[sl], c.Y[sl]
        sd = _sd_box(X, Y, s * WHEEL_X, FRONT_AXLE_Y, TYRE_HALF_WIDTH, TYRE_RADIUS, 0.05)
        dy = (Y - FRONT_AXLE_Y) / TYRE_RADIUS
        dx = (np.abs(X - s * WHEEL_X)) / TYRE_HALF_WIDTH
        # A cylinder along x, rounded at the shoulders
        h = TYRE_RADIUS * (1.0 + np.sqrt(np.clip(1.0 - dy * dy, 0.0, 1.0)))
        h = h - 0.05 * np.clip((dx - 0.7) / 0.3, 0.0, 1.0) ** 2
        c.paint(sl, sd, h, TYRE, RUBBER)
        # Wishbones and the push rod: tubes from the chassis to the upright
        sl = c.window(min(s * 0.15, s * 0.74), max(s * 0.15, s * 0.74), 0.88, 1.50)
        X, Y = c.X[sl], c.Y[sl]
        for (ax_, ay), (bx, by), hz in (
            ((0.16, 0.98), (0.70, 1.16), 0.44),
            ((0.18, 1.44), (0.70, 1.21), 0.44),
            ((0.18, 1.24), (0.70, 1.18), 0.40),
        ):
            d, _ = _sd_segment(X, Y, s * ax_, ay, s * bx, by)
            r = 0.022
            h = hz + np.sqrt(np.clip(r * r - d * d, 0.0, None))
            c.paint(sl, d - r, h, CARBON, CARBON_MAT)


def _floor_and_sidepods(c: _Canvas) -> None:
    sl = c.window(-0.85, 0.85, 2.0, FRONT_LENGTH_M)
    X, Y = c.X[sl], c.Y[sl]
    ax = np.abs(X)
    # Floor: a wide carbon plate, its edge curling up slightly
    hw = _taper(Y, 2.08, 2.5, 0.50, 0.80)
    sd = np.maximum(ax - hw, 2.06 - Y)
    edge = np.clip((ax - (hw - 0.06)) / 0.06, 0.0, 1.0)
    c.paint(sl, sd, 0.06 + 0.03 * edge, CARBON, CARBON_MAT)
    # Sidepods: inlet at the front, outer edge sweeping in (the "coke bottle")
    out = np.where(
        Y < 2.75,
        0.56 + 0.03 * _smooth01((Y - 2.16) / 0.25),
        0.59 - 0.29 * _smooth01((Y - 2.75) / 1.25),
    )
    inner = 0.27
    front = 2.16 + 0.06 * ((ax - inner) / 0.32) ** 2
    sd = np.maximum(np.maximum(ax - out, inner - ax - 0.05), front - Y)
    width = np.maximum(out - inner, 1e-3)
    u = np.clip((ax - inner) / width, 0.0, 1.0)  # 0 at the chassis, 1 at the outer edge
    crown = np.where(Y < 2.7, 0.55, _taper(Y, 2.7, 3.95, 0.55, 0.30))
    shoulder = np.sqrt(np.clip(1.0 - np.clip((u - 0.45) / 0.55, 0.0, 1.0) ** 2, 0.0, 1.0))
    nose_round = np.sqrt(np.clip((Y - front) / 0.12, 0.0, 1.0))
    h = 0.12 + (crown - 0.12) * shoulder * nose_round
    c.paint(sl, sd, h, WHITE, PAINT)
    # Inlet mouths: dark slots at the front of each sidepod
    sd_in = np.maximum(
        np.maximum(np.abs(Y - (front + 0.035)) - 0.03, ax - (out - 0.07)), inner + 0.06 - ax
    )
    c.paint(sl, sd_in, h + 0.001, COCKPIT, MATTE)


def _cockpit_and_driver(c: _Canvas) -> None:
    sl = c.window(-0.35, 0.35, 1.8, 3.0)
    X, Y = c.X[sl], c.Y[sl]
    # Cockpit opening: recessed and dark
    sd = _sd_box(X, Y, 0.0, 2.46, 0.24, 0.38, 0.14)
    c.paint(sl, sd, 0.64, COCKPIT, MATTE)
    # Headrest padding around the helmet
    sd = _sd_box(X, Y, 0.0, 2.65, 0.23, 0.17, 0.10)
    c.paint(sl, sd, 0.66, CARBON_LIGHT, MATTE)
    # Helmet: a sphere, white, with the visor at the front
    r = 0.135
    d = np.hypot(X, (Y - 2.50) * 0.95)
    h = 0.70 + np.sqrt(np.clip(r * r - d * d, 0.0, None))
    visor = ((Y < 2.46) & (np.abs(X) < 0.10))[..., None]
    col = np.where(visor, np.array(VISOR, np.float32), np.array(WHITE, np.float32))
    c.paint(sl, d - r, h, col, PAINT)
    c.paint(sl, np.where(visor[..., 0], d - r, 1.0), h + 0.001, VISOR, GLASS)
    # Steering wheel peeking out in front of the driver
    sd = _sd_box(X, Y, 0.0, 2.22, 0.11, 0.025, 0.02)
    c.paint(sl, sd, 0.68, CARBON, CARBON_MAT)
    # Mirrors on stalks
    for s in (-1.0, 1.0):
        d, _ = _sd_segment(X, Y, s * 0.27, 2.16, s * 0.48, 2.14)
        c.paint(sl, d - 0.012, 0.64, CARBON, CARBON_MAT)
    sl2 = c.window(-0.7, 0.7, 2.0, 2.3)
    X2, Y2 = c.X[sl2], c.Y[sl2]
    for s in (-1.0, 1.0):
        sd = _sd_box(X2, Y2, s * 0.54, 2.14, 0.075, 0.035, 0.02)
        c.paint(sl2, sd, 0.68, WHITE, PAINT)


def _halo(c: _Canvas) -> None:
    sl = c.window(-0.4, 0.4, 1.8, 3.0)
    X, Y = c.X[sl], c.Y[sl]
    pts = [(0.0, 1.90), (0.0, 2.15)]
    loop = [(0.0, 2.15)]
    for k in range(1, 13):
        a = math.pi / 2 * k / 12
        loop.append((0.275 * math.sin(a), 2.15 + 0.30 * (1 - math.cos(a))))
    loop.append((0.28, 2.82))
    r = 0.034
    for side in (1.0, -1.0):
        line = [(side * x, y) for x, y in loop]
        d = _sd_polyline(X, Y, line)
        h = (
            0.84
            + np.sqrt(np.clip(r * r - d * d, 0.0, None))
            - 0.10 * np.clip((Y - 2.45) / 0.37, 0, 1)
        )
        c.paint(sl, d - r, h, CARBON_LIGHT, CARBON_MAT)
    d = _sd_polyline(X, Y, pts)
    h = (
        0.62
        + 0.22 * np.clip((Y - 1.90) / 0.25, 0.0, 1.0)
        + np.sqrt(np.clip(r * r - d * d, 0, None))
    )
    c.paint(sl, d - r, h, CARBON_LIGHT, CARBON_MAT)


def _engine_cover(c: _Canvas) -> None:
    sl = c.window(-0.35, 0.35, 2.65, FRONT_LENGTH_M)
    X, Y = c.X[sl], c.Y[sl]
    hw = np.where(
        Y < 2.95, 0.17 + 0.08 * _smooth01((Y - 2.72) / 0.23), _taper(Y, 2.95, 4.0, 0.25, 0.12)
    )
    sd = np.maximum(np.abs(X) - hw, 2.74 - Y)
    dome = np.sqrt(np.clip(1.0 - (X / np.maximum(hw, 1e-3)) ** 2, 0.0, 1.0))
    top = np.where(
        Y < 3.00, 0.80 + 0.18 * _smooth01((Y - 2.74) / 0.2), _taper(Y, 3.00, 4.0, 0.98, 0.60)
    )
    h = 0.55 + (top - 0.55) * dome
    c.paint(sl, sd, h, WHITE, PAINT)
    # The airbox intake above the driver's head, and the shark fin
    sd = _sd_box(X, Y, 0.0, 2.79, 0.075, 0.035, 0.03)
    c.paint(sl, sd, h + 0.002, COCKPIT, MATTE)
    sd = np.maximum(np.abs(X) - 0.008, np.maximum(3.15 - Y, Y - 4.0))
    c.paint(sl, sd, h + 0.03, SILVER, PAINT)


def _details(c: _Canvas) -> None:
    """Cooling louvres on the sidepods and the brake-duct fairings inside the front wheels."""
    for s in (-1.0, 1.0):
        sl = c.window(s * 0.55, s * 0.25, 2.85, 3.45) if s < 0 else c.window(0.25, 0.55, 2.85, 3.45)
        X, Y = c.X[sl], c.Y[sl]
        for k in range(7):
            yk = 2.92 + 0.06 * k
            sd = _sd_box(X, Y, s * (0.405 - 0.012 * k), yk, 0.085 - 0.004 * k, 0.011, 0.008)
            c.decal(sl, sd, -0.012, COCKPIT, MATTE)
        sl = c.window(s * WHEEL_X - 0.3, s * WHEEL_X, FRONT_AXLE_Y - 0.3, FRONT_AXLE_Y + 0.3)
        if s < 0:
            sl = c.window(s * WHEEL_X, s * WHEEL_X + 0.3, FRONT_AXLE_Y - 0.3, FRONT_AXLE_Y + 0.3)
        X, Y = c.X[sl], c.Y[sl]
        cx = s * (WHEEL_X - TYRE_HALF_WIDTH - 0.04)
        sd = _sd_box(X, Y, cx, FRONT_AXLE_Y, 0.035, 0.24, 0.03)
        dome = np.sqrt(np.clip(1.0 - ((Y - FRONT_AXLE_Y) / 0.24) ** 2, 0.0, 1.0))
        c.paint(sl, sd, 0.30 + 0.16 * dome, CARBON, CARBON_MAT)


def build_maps(px_per_m: float = PX_PER_M) -> CarMaps:
    """Draw the car; see the module docstring. About a second at the default resolution."""
    c = _Canvas(px_per_m)
    _front_wing(c)
    _floor_and_sidepods(c)
    _nose_and_chassis(c)
    _cockpit_and_driver(c)
    _engine_cover(c)
    _halo(c)
    _front_tyres_and_suspension(c)
    _details(c)

    # Fade out towards the cut at the back
    fade = 1.0 - _smooth01((c.Y - (FRONT_LENGTH_M - FADE_M)) / FADE_M)
    alpha = c.cover * fade
    # Normals from the height field, lightly smoothed so edges read as small bevels
    hs = gaussian_filter(c.height, sigma=1.2)
    gy, gx = np.gradient(hs, 1.0 / px_per_m)
    n = np.stack([-gx, gy, np.ones_like(gx)], axis=-1)  # y grows downwards in the map
    # Carbon twill: diagonal stripes that tilt the normal a little and shade the colour
    carbon = np.isclose(c.mat[..., 0], CARBON_MAT[0]) & np.isclose(c.mat[..., 1], CARBON_MAT[1])
    wave = np.sin(2.0 * np.pi * (c.X + c.Y) / TWILL_PERIOD_M)
    tilt = TWILL_TILT * wave * carbon
    n[..., 0] += tilt
    n[..., 1] -= tilt
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    col = c.albedo * (1.0 + 0.12 * wave * carbon)[..., None]
    albedo = np.concatenate([col, alpha[..., None]], axis=-1)
    return CarMaps(albedo.astype(np.float32), n.astype(np.float32), c.mat, c.height, px_per_m)


def shade(
    maps: CarMaps, light_xy: tuple[float, float], light_z: float = LIGHT_HEIGHT_M
) -> np.ndarray:
    """A preview of the maps lit by a point light at ``light_xy`` (car metres), ``(H, W, 3)``
    0..1 on black: the same model as the menu's shader, for tests and quick looks."""
    h, w = maps.height.shape
    ys = (np.arange(h) + 0.5) / maps.px_per_m
    xs = (np.arange(w) + 0.5) / maps.px_per_m - HALF_WIDTH_M
    X, Y = np.meshgrid(xs, ys)
    lx, ly = light_xy
    # Light vector in map axes (x right, y down the map = towards the rear, z up)
    L = np.stack([lx - X, -(ly - Y), light_z - maps.height], axis=-1)
    dist = np.linalg.norm(L, axis=-1, keepdims=True)
    L = L / dist
    n = maps.normal
    ndl = np.clip(np.sum(n * L, axis=-1), 0.0, None)
    hv = L + np.array([0.0, 0.0, 1.0])
    hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
    spec_s, gloss = maps.material[..., 0], maps.material[..., 1]
    spec = spec_s * np.clip(np.sum(n * hv, axis=-1), 0.0, None) ** (8.0 + 120.0 * gloss)
    r_xy = np.hypot(X - lx, Y - ly)
    att = 1.0 / (1.0 + (r_xy / LIGHT_CORE_M) ** 2) * (1.0 - _smooth01(r_xy / LIGHT_REACH_M))
    col = maps.albedo[..., :3]
    out = col * (AMBIENT + DIFFUSE_GAIN * ndl[..., None] * att[..., None])
    out = out + (spec * att)[..., None]
    return np.clip(out * maps.albedo[..., 3:4], 0.0, 1.0)
