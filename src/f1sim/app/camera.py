"""2-D view transform between world metres and screen pixels (pure maths, no arcade).

World ``y`` is up and screen ``y`` is up (arcade's convention, origin at the bottom-left of
the viewport). The view is a uniform scale about a centre point::

    sx = viewport_w / 2 + (x - centre_x) * pixels_per_metre
    sy = viewport_h / 2 + (y - centre_y) * pixels_per_metre

This is exactly what an arcade ``Camera2D`` with ``position = centre``, ``zoom =
pixels_per_metre`` and its default viewport-sized, centred projection draws, so the drive viewer
hands these two numbers to a ``Camera2D`` each frame and the cached track geometry (in metres)
is transformed on the GPU.
"""

from __future__ import annotations

import numpy as np

MIN_PIXELS_PER_METRE = 0.01
"""Most zoomed-out scale: 100 m per pixel, a 100 km wide view at 1000 px."""
MAX_PIXELS_PER_METRE = 200.0
"""Most zoomed-in scale: 5 mm per pixel."""


class Camera:
    """Centre point (world metres) and scale (pixels per metre) for a viewport in pixels."""

    def __init__(self, viewport_w: int, viewport_h: int):
        self.viewport_w = max(1, int(viewport_w))
        self.viewport_h = max(1, int(viewport_h))
        self.centre_x = 0.0
        self.centre_y = 0.0
        self._ppm = 1.0

    # ----------------------------------------------------------------------- properties

    @property
    def pixels_per_metre(self) -> float:
        return self._ppm

    @property
    def centre(self) -> tuple[float, float]:
        return self.centre_x, self.centre_y

    def _set_ppm(self, ppm: float) -> None:
        self._ppm = float(np.clip(ppm, MIN_PIXELS_PER_METRE, MAX_PIXELS_PER_METRE))

    # ---------------------------------------------------------------------- operations

    def fit(self, xmin, ymin, xmax, ymax, padding_px: int = 20) -> None:
        """Centre the box and scale it to fill the viewport, leaving ``padding_px`` per side.

        The scale is clamped to the zoom range; a degenerate box (a point) keeps the scale.
        """
        xmin, ymin, xmax, ymax = (float(v) for v in (xmin, ymin, xmax, ymax))
        self.centre_x = 0.5 * (xmin + xmax)
        self.centre_y = 0.5 * (ymin + ymax)
        avail_w = max(1.0, self.viewport_w - 2.0 * padding_px)
        avail_h = max(1.0, self.viewport_h - 2.0 * padding_px)
        scales = [
            avail / size
            for avail, size in ((avail_w, xmax - xmin), (avail_h, ymax - ymin))
            if size > 0.0
        ]
        if scales:
            self._set_ppm(min(scales))

    def zoom_at(self, factor: float, screen_x: float, screen_y: float) -> None:
        """Multiply the scale by ``factor`` (clamped) keeping the world point under
        ``(screen_x, screen_y)`` at the same screen position."""
        if not factor > 0.0:
            raise ValueError(f"zoom factor must be > 0, got {factor!r}")
        wx, wy = self.screen_to_world(screen_x, screen_y)
        self._set_ppm(self._ppm * factor)
        # Solve screen_x = w/2 + (wx - cx) * ppm for the new centre.
        self.centre_x = float(wx) - (screen_x - 0.5 * self.viewport_w) / self._ppm
        self.centre_y = float(wy) - (screen_y - 0.5 * self.viewport_h) / self._ppm

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Drag the view by ``(dx_px, dy_px)`` pixels: the scene moves with the cursor, so a
        world point at screen ``p`` ends up at ``p + (dx, dy)``."""
        self.centre_x -= dx_px / self._ppm
        self.centre_y -= dy_px / self._ppm

    def follow(self, x: float, y: float) -> None:
        """Centre the view on the world point ``(x, y)``."""
        self.centre_x = float(x)
        self.centre_y = float(y)

    def follow_at(self, x: float, y: float, screen_x: float, screen_y: float) -> None:
        """Move the view so the world point ``(x, y)`` appears at ``(screen_x, screen_y)``
        (e.g. the middle of the area the HUD leaves free)."""
        self.centre_x = float(x) - (screen_x - 0.5 * self.viewport_w) / self._ppm
        self.centre_y = float(y) - (screen_y - 0.5 * self.viewport_h) / self._ppm

    def resize(self, viewport_w: int, viewport_h: int) -> None:
        """New viewport size; the world centre and the scale stay the same."""
        self.viewport_w = max(1, int(viewport_w))
        self.viewport_h = max(1, int(viewport_h))

    # ---------------------------------------------------------------------- transforms

    def world_to_screen(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """World metres -> screen pixels; vectorised over broadcastable ``x``, ``y``."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sx = 0.5 * self.viewport_w + (x - self.centre_x) * self._ppm
        sy = 0.5 * self.viewport_h + (y - self.centre_y) * self._ppm
        return sx, sy

    def screen_to_world(self, sx, sy) -> tuple[np.ndarray, np.ndarray]:
        """Screen pixels -> world metres; the exact inverse of :meth:`world_to_screen`."""
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        x = self.centre_x + (sx - 0.5 * self.viewport_w) / self._ppm
        y = self.centre_y + (sy - 0.5 * self.viewport_h) / self._ppm
        return x, y
