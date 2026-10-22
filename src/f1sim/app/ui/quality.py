"""Rendering quality, so the app stays smooth on slower machines.

Three tiers (:data:`TIERS`) set what the expensive passes spend:

- the menu floor's light (soft light and shadows) is computed in a buffer this share of the
  window's logical size and upsampled (``floor_scale``);
- how many trackside lamps light each car (``car_lamps``) and how many cars cast the cursor
  light's shadow (``shadow_cars``);
- the drive and replay scene's multisampling (``msaa``) and the size of its blur target
  (``blur_div``: the logical size over this).

The starting tier comes from ``F1SIM_QUALITY`` (``high``, ``medium``, ``low`` or ``auto``,
the default). In ``auto`` it comes from the GPU: integrated and mobile GPUs start at medium,
software renderers at low, the rest at high. Then a :class:`Governor`, fed every frame's
duration by the app window, steps the tier down when frames stay slow. It never steps back
up: stepping both ways would flicker between tiers.

Pure Python: the window feeds it; the screens read :func:`tier` every frame.
"""

from __future__ import annotations

import os
import statistics
from collections import deque
from dataclasses import dataclass

ENV = "F1SIM_QUALITY"


@dataclass(frozen=True)
class Tier:
    name: str
    floor_scale: float
    car_lamps: int
    shadow_cars: int
    msaa: int
    blur_div: int


TIERS = (
    Tier("low", 0.25, 2, 1, 0, 4),
    Tier("medium", 0.35, 4, 3, 2, 4),
    Tier("high", 0.5, 8, 6, 4, 2),
)
NAMES = tuple(t.name for t in TIERS)

SOFTWARE = ("llvmpipe", "softpipe", "swiftshader", "microsoft basic render", "gdi generic")
INTEGRATED = (
    "intel",
    "mali",
    "adreno",
    "powervr",
    "vivante",
    "videocore",
    "apple m1",
)
"""Renderer names (lower case, substrings) that start below the top tier."""


def initial_tier(renderer: str, setting: str | None = None) -> tuple[str, bool]:
    """``(tier name, auto)``: ``setting`` (``F1SIM_QUALITY``) if it names a tier (then the
    governor is off), else a guess from the GL ``renderer`` string."""
    setting = (setting or "auto").strip().lower()
    if setting in NAMES:
        return setting, False
    r = renderer.lower()
    if any(s in r for s in SOFTWARE):
        return "low", True
    if any(s in r for s in INTEGRATED):
        return "medium", True
    return "high", True


class Governor:
    """Steps the tier down when frames stay slow: the median of the last ``WINDOW`` frames
    over ``SLOW_S``, at least ``SETTLE_S`` after the start or the last step (loading and
    shader compiles make the first frames slow)."""

    WINDOW = 90
    SLOW_S = 1.0 / 45.0
    SETTLE_S = 3.0
    STALL_S = 0.25
    """A frame longer than this is a stall (a load, a drag), not slowness: ignored."""

    def __init__(self, name: str = "high", auto: bool = True):
        self.index = NAMES.index(name)
        self.auto = auto
        self.paused = False
        self._frames: deque[float] = deque(maxlen=self.WINDOW)
        self._since = 0.0

    @property
    def tier(self) -> Tier:
        return TIERS[self.index]

    def feed(self, dt: float) -> bool:
        """One frame took ``dt`` seconds. Returns True if the tier just stepped down."""
        if not self.auto or self.paused or self.index == 0 or dt <= 0.0:
            return False
        self._since += min(dt, self.STALL_S)
        if dt > self.STALL_S:
            return False
        self._frames.append(dt)
        if self._since < self.SETTLE_S or len(self._frames) < self.WINDOW:
            return False
        if statistics.median(self._frames) <= self.SLOW_S:
            return False
        self.index -= 1
        self._frames.clear()
        self._since = 0.0
        return True


current = Governor()
"""The app's governor (the window sets it up from its GL renderer: :func:`setup`)."""


def setup(renderer: str) -> Governor:
    """Start the app's governor from ``F1SIM_QUALITY`` and the GL ``renderer`` name."""
    global current
    current = Governor(*initial_tier(renderer, os.environ.get(ENV)))
    return current


def tier() -> Tier:
    return current.tier
