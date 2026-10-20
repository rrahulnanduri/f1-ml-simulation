"""Agent observations, spec v1 (docs/data-formats.md section 6).

Layout of each ``(OBS_SIZE,)`` row, float32:

====== ================================================== ===============================
index  feature                                            normalisation
====== ================================================== ===============================
0-8    wall distance along ``yaw + RAY_ANGLES_DEG``       ÷ ``RAY_RANGE_M``
9      speed ``v``                                        ÷ ``V_NORM``
10     heading error ``wrap(yaw - heading_at(s))``        ÷ pi
11     lateral offset ``n``                               ÷ ``w_left`` if n >= 0 else ``w_right``
12-16  curvature at ``s + LOOKAHEAD_M``                   × ``CURVATURE_SCALE``, clipped ±1
17-18  previous action (steer, pedal)                     as given
====== ================================================== ===============================

Changing any of this must bump ``OBS_SPEC_VERSION``.
"""

from __future__ import annotations

import numpy as np

from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track

OBS_SPEC_VERSION = 1
RAY_ANGLES_DEG = (-90, -60, -35, -15, 0, 15, 35, 60, 90)  # relative to yaw; + = left
RAY_RANGE_M = 200.0
LOOKAHEAD_M = (20.0, 50.0, 100.0, 150.0, 250.0)
V_NORM = 92.0
CURVATURE_SCALE = 50.0
OBS_SIZE = 19

_RAYS = slice(0, 9)
_SPEED = 9
_HEADING = 10
_LATERAL = 11
_CURVATURE = slice(12, 17)
_ACTION = slice(17, 19)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Sensors:
    """Builds the observation vectors of N cars on one track."""

    def __init__(self, track: Track, grid: SegmentGrid | None = None):
        self.track = track
        self.grid = grid if grid is not None else SegmentGrid.from_track(track)
        self._ray_offsets = np.deg2rad(np.asarray(RAY_ANGLES_DEG, dtype=np.float64))
        self._lookahead = np.asarray(LOOKAHEAD_M, dtype=np.float64)

    def observe(self, x, y, yaw, v, s, n, prev_action) -> np.ndarray:
        """Observations ``(N, OBS_SIZE)`` float32 for cars at ``(x, y)`` heading ``yaw``.

        ``x, y, yaw, v, s, n`` have shape ``(N,)`` (``s``, ``n`` from ``Track.project``) and
        ``prev_action`` ``(N, 2)`` (steer, pedal). See the module docstring for the layout.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must have shape (N,), got {x.shape}")
        cars = x.shape[0]
        named = {"y": y, "yaw": yaw, "v": v, "s": s, "n": n}
        arrs = {}
        for name, val in named.items():
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape != (cars,):
                raise ValueError(f"{name} must have shape ({cars},), got {arr.shape}")
            arrs[name] = arr
        action = np.asarray(prev_action, dtype=np.float64)
        if action.shape != (cars, 2):
            raise ValueError(f"prev_action must have shape ({cars}, 2), got {action.shape}")
        y, yaw, v, s, n = (arrs[k] for k in ("y", "yaw", "v", "s", "n"))

        obs = np.empty((cars, OBS_SIZE), dtype=np.float32)
        if cars == 0:
            return obs
        angles = yaw[:, None] + self._ray_offsets[None, :]
        obs[:, _RAYS] = self.grid.raycast(x, y, angles, RAY_RANGE_M) / RAY_RANGE_M
        obs[:, _SPEED] = v / V_NORM
        obs[:, _HEADING] = _wrap_angle(yaw - self.track.heading_at(s)) / np.pi
        w_left, w_right = self.track.widths_at(s)
        obs[:, _LATERAL] = np.where(n >= 0.0, n / w_left, n / w_right)
        kappa = self.track.curvature_at(s[:, None] + self._lookahead[None, :])
        obs[:, _CURVATURE] = np.clip(kappa * CURVATURE_SCALE, -1.0, 1.0)
        obs[:, _ACTION] = action
        return obs
