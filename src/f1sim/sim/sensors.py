"""Agent observations, spec v2 (docs/data-formats.md section 6).

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
19     offset from the racing line (+ = car left of it)   ÷ 5 m, clipped ±2
20     heading error to the racing line                   ÷ 0.5 rad
21-24  racing line's lateral position 10/30/60/100 m      ÷ 20 m, clipped ±2
       ahead along it, in the car frame (+ = left)
25-28  racing line's profile speed 0/50/100/200 m ahead   ÷ 92 m/s
====== ================================================== ===============================

v2 added 19-28 (:mod:`f1sim.sim.raceline`); 0-18 are v1 unchanged. Behaviour cloning showed
v1 doesn't locate the racing line, so no policy could drive it (D16 in the project log).
Changing any of this must bump ``OBS_SPEC_VERSION``.

The racing-line projection is hinted per car (O(1) per car), like the progress tracker's.
After a reset the World calls :meth:`Sensors.forget` so those cars are searched globally; a
hinted result more than ``REPROJECT_OFFSET_M`` off the line is redone globally too.
"""

from __future__ import annotations

import numpy as np

from f1sim.sim import raceline as rl
from f1sim.sim.car_model import CarParams
from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track

OBS_SPEC_VERSION = 2
RAY_ANGLES_DEG = (-90, -60, -35, -15, 0, 15, 35, 60, 90)  # relative to yaw; + = left
RAY_RANGE_M = 200.0
LOOKAHEAD_M = (20.0, 50.0, 100.0, 150.0, 250.0)
V_NORM = 92.0
CURVATURE_SCALE = 50.0
OBS_SIZE = 19 + rl.N_FEATURES
REPROJECT_OFFSET_M = 25.0

_RAYS = slice(0, 9)
_SPEED = 9
_HEADING = 10
_LATERAL = 11
_CURVATURE = slice(12, 17)
_ACTION = slice(17, 19)
_RACELINE = slice(19, 19 + rl.N_FEATURES)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Sensors:
    """Builds the observation vectors of N cars on one track.

    ``racing_line`` defaults to :func:`f1sim.sim.raceline.racing_line_for` the track and
    ``car`` (default parameters if None).
    """

    def __init__(
        self,
        track: Track,
        grid: SegmentGrid | None = None,
        racing_line: rl.RacingLine | None = None,
        car: CarParams | None = None,
    ):
        self.track = track
        self.grid = grid if grid is not None else SegmentGrid.from_track(track)
        self.racing_line = (
            racing_line if racing_line is not None else rl.racing_line_for(track, car)
        )
        self._hint: np.ndarray | None = None
        self._fresh: np.ndarray | None = None
        self._ray_offsets = np.deg2rad(np.asarray(RAY_ANGLES_DEG, dtype=np.float64))
        self._lookahead = np.asarray(LOOKAHEAD_M, dtype=np.float64)

    def forget(self, mask=None) -> None:
        """Search the racing line globally for these cars (all if None) next time."""
        if self._fresh is None or mask is None:
            self._hint = None
        else:
            self._fresh |= np.asarray(mask, dtype=bool)

    def _project_line(self, x, y):
        line = self.racing_line
        cars = x.size
        if self._hint is None or self._hint.size != cars:
            s_line, off, idx = line.project(x, y)
        else:
            s_line, off, idx = line.project(x, y, hint=self._hint)
            redo = self._fresh | (np.abs(off) > REPROJECT_OFFSET_M)
            if redo.any():
                g = line.project(x[redo], y[redo])
                s_line[redo], off[redo], idx[redo] = g
        self._hint = idx
        self._fresh = np.zeros(cars, dtype=bool)
        return s_line, off

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
        s_line, off = self._project_line(x, y)
        obs[:, _RACELINE] = self.racing_line.features(x, y, yaw, s_line, off)
        return obs
