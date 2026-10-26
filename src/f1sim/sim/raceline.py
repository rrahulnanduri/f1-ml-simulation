"""The racing line a car is told about: a path inside the track and its speed profile.

Observation spec v2 (:mod:`f1sim.sim.sensors`) adds where this line is relative to the car,
and the speed it allows ahead. Behaviour cloning (D16 in the project log) showed why: the v1
inputs (rays, centre-line offset, curvature) don't locate the racing line, so no policy could
drive it.

The line is the scripted reference driver's path (:mod:`f1sim.sim.scripted`):
- **Shape:** the circuit's TUM raceline, resampled and pulled ``INSIDE_MARGIN_M`` inside the
  off-track limit of the track the cars drive on. Catalogue circuits without one get a
  computed minimum-curvature line (:mod:`f1sim.sim.mincurv`, D37); other tracks (the tests'
  synthetic ones) use the centre line. :func:`raceline_for` picks.
- **Speed:** the quasi-steady-state speed profile on that path, with the driver's 1% curvature
  margin.

:func:`racing_line_for` builds one per (track, car) and caches it, so the Worlds of a
training run share it.
"""

from __future__ import annotations

import logging

import numpy as np

from f1sim.sim import scripted
from f1sim.sim.car_model import CarParams
from f1sim.track.track import Track

log = logging.getLogger(__name__)

AHEAD_M = (10.0, 30.0, 60.0, 100.0)
"""Distances along the line at which its lateral position (car frame) is observed, m."""
SPEED_AHEAD_M = (0.0, 50.0, 100.0, 200.0)
"""Distances along the line at which its profile speed is observed, m."""
N_FEATURES = 2 + len(AHEAD_M) + len(SPEED_AHEAD_M)
OFFSET_NORM_M = 5.0
HEADING_NORM_RAD = 0.5
AHEAD_NORM_M = 20.0
SPEED_NORM = 92.0
CLIP = 2.0


def tum_raceline(track_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The circuit's TUM raceline ``(x, y)`` (downloaded once into the cache), or None if the
    circuit has none or it can't be fetched."""
    from f1sim.data import catalog, tum_source

    try:
        tum_name = catalog.get(track_id).tum_name
        if tum_name is None:
            return None
        return tum_source.parse_tum_raceline(tum_source.fetch_tum_raceline(tum_name))
    except (KeyError, ValueError, OSError) as exc:
        log.info("no TUM raceline for %s (%s); using the centre line", track_id, exc)
        return None


def raceline_for(track: Track) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
    """``(raceline_xy, source)`` for ``track``: the TUM raceline (``"tum-raceline"``), else a
    minimum-curvature line inside ``track`` for a catalogue circuit (``"min-curvature"``),
    else ``(None, "centre-line")``."""
    from f1sim.data import catalog
    from f1sim.sim.mincurv import min_curvature_line

    tum = tum_raceline(track.data.id)
    if tum is not None:
        return tum, "tum-raceline"
    try:
        catalog.get(track.data.id)
    except KeyError:
        return None, "centre-line"
    return min_curvature_line(track), "min-curvature"


class RacingLine:
    """A closed path inside ``track`` with its speed profile for ``car``; see the module doc.

    ``raceline_xy=None`` uses the track's centre line. ``source`` says which line it is
    (default: ``"centre-line"`` without a raceline, ``"tum-raceline"`` with one).
    """

    def __init__(self, track: Track, car: CarParams, raceline_xy=None, source: str | None = None):
        self.track = track
        self.car = car
        if source is None:
            source = "centre-line" if raceline_xy is None else "tum-raceline"
        self.source = source
        self.path = scripted._reference_path(track, raceline_xy, scripted.INSIDE_MARGIN_M)
        self.v_profile = scripted._profile_on_path(
            self.path, car, curvature_margin=scripted.DRIVE_CURVATURE_MARGIN
        )
        self.ptrack = scripted._path_as_track(self.path)
        self.length = float(self.path.length)

    def speed_at(self, s_line) -> np.ndarray:
        """Profile speed (m/s) at arc length(s) ``s_line`` along the line (wraps)."""
        return scripted._interp_periodic(s_line, self.path.s, self.v_profile, self.length)

    def project(self, x, y, hint=None):
        """``(s_line, offset, idx)``: arc length along the line, signed offset from it
        (positive = car left of the line) and segment index (the next call's ``hint``)."""
        return self.ptrack.project(x, y, hint=hint)

    def features(self, x, y, yaw, s_line, offset) -> np.ndarray:
        """The ``N_FEATURES`` observation columns ``(N, N_FEATURES)`` for cars at ``x, y``
        heading ``yaw``, already projected onto the line (``s_line``, ``offset``)."""
        x, y, yaw = (np.asarray(a, dtype=np.float64) for a in (x, y, yaw))
        out = np.empty((x.size, N_FEATURES), dtype=np.float64)
        out[:, 0] = np.clip(offset / OFFSET_NORM_M, -CLIP, CLIP)
        h_err = (yaw - self.ptrack.heading_at(s_line) + np.pi) % (2.0 * np.pi) - np.pi
        out[:, 1] = h_err / HEADING_NORM_RAD
        sin, cos = np.sin(yaw), np.cos(yaw)
        for k, d in enumerate(AHEAD_M):
            p = np.atleast_2d(self.ptrack.point_at(s_line + d))
            lateral = -sin * (p[:, 0] - x) + cos * (p[:, 1] - y)
            out[:, 2 + k] = np.clip(lateral / AHEAD_NORM_M, -CLIP, CLIP)
        base = 2 + len(AHEAD_M)
        for k, d in enumerate(SPEED_AHEAD_M):
            out[:, base + k] = self.speed_at(s_line + d) / SPEED_NORM
        return out


_CACHE: dict[tuple[int, CarParams], tuple[Track, RacingLine]] = {}


def racing_line_for(track: Track, car: CarParams | None = None) -> RacingLine:
    """The cached :class:`RacingLine` of ``track``, on :func:`raceline_for`'s line."""
    car = car if car is not None else CarParams()
    key = (id(track.data), car)
    hit = _CACHE.get(key)
    if hit is not None and hit[0].data is track.data:
        return hit[1]
    xy, source = raceline_for(track)
    line = RacingLine(track, car, xy, source)
    _CACHE[key] = (track, line)  # holding the track keeps id(track.data) unique
    return line
