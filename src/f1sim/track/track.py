"""Track geometry: the built centre line with widths, and fast queries on it.

Conventions (docs/m1-interfaces.md): closed loop without a repeated end point, ``s[0] = 0``,
``length_m`` includes the closing segment, heading is ``atan2(dy, dx)``, the left normal is
``(-sin h, cos h)`` and the lateral offset ``n`` is positive to the left.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

_ARRAY_FIELDS = ("x", "y", "s", "w_left", "w_right", "heading", "curvature")
_DIRECTIONS = ("", "clockwise", "counterclockwise")

# Bound on the (points x segments) temporaries in a global projection: about 8 MB per array.
_PROJECT_CHUNK_ELEMS = 1_000_000


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _cross(ux, uy, vx, vy):
    return ux * vy - uy * vx


@dataclass(frozen=True)
class TrackData:
    """A built track: per-point centre line, widths, heading and curvature.

    Arrays are copied to read-only ``float64`` on construction.
    """

    id: str
    name: str
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray  # > 0 = turning left
    length_m: float
    resolution_m: float
    sectors_s: tuple[float, ...] = ()
    fastf1_location: str | None = None
    direction: str = ""  # "clockwise" | "counterclockwise"
    source: dict = field(default_factory=dict)
    fastf1_alignment: dict | None = None

    def __post_init__(self) -> None:
        m = None
        for name in _ARRAY_FIELDS:
            arr = np.array(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"TrackData.{name} must be 1-D, got shape {arr.shape}")
            if m is None:
                m = arr.size
            elif arr.size != m:
                raise ValueError(f"TrackData.{name} has {arr.size} points, expected {m}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"TrackData.{name} contains non-finite values")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if m is None or m < 3:
            raise ValueError("TrackData needs at least 3 points")
        object.__setattr__(self, "length_m", float(self.length_m))
        object.__setattr__(self, "resolution_m", float(self.resolution_m))
        object.__setattr__(self, "sectors_s", tuple(float(v) for v in self.sectors_s))
        s = self.s
        if abs(s[0]) > 1e-9:
            raise ValueError(f"TrackData.s must start at 0, got {s[0]}")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("TrackData.s must be strictly increasing")
        if not self.length_m > s[-1]:
            raise ValueError(
                f"TrackData.length_m ({self.length_m}) must exceed s[-1] ({s[-1]}): "
                "it includes the closing segment"
            )
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"TrackData.direction must be one of {_DIRECTIONS}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrackData):
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name in _ARRAY_FIELDS:
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    @property
    def n_points(self) -> int:
        """Number of centre-line points (= number of segments, the loop is closed)."""
        return int(self.x.size)


class Track:
    """Queries on a :class:`TrackData`: walls, interpolation along ``s`` and projection.

    Segment ``i`` joins point ``i`` to point ``i + 1`` (the last one joins back to point 0).
    Between points, position, widths and curvature are interpolated linearly in ``s`` and the
    heading as an angle. :meth:`project` uses the same model: the normal is interpolated
    linearly along each segment, so wall vertices map to ``n = w_left`` / ``-w_right``
    exactly and points along a wall to ``±widths_at(s)`` (exactly where the width is
    constant). For tracks from ``build_track`` (widths clipped so the normals within the
    walls never cross) the projection is one-to-one between the walls, inside bends too.
    """

    def __init__(self, data: TrackData):
        self.data = data
        self.length = float(data.length_m)
        self._pts = np.column_stack([data.x, data.y])
        self._m = self._pts.shape[0]
        nxt = np.roll(self._pts, -1, axis=0)
        self._d = nxt - self._pts  # segment vectors (M, 2)
        self._len2 = np.einsum("ij,ij->i", self._d, self._d)
        if np.any(self._len2 <= 0.0):
            raise ValueError("track has zero-length segments")
        self._s = data.s
        self._ds = np.diff(np.append(data.s, self.length))  # s-length of each segment
        self._nl = np.column_stack([-np.sin(data.heading), np.cos(data.heading)])
        self._dh = _wrap_angle(np.roll(data.heading, -1) - data.heading)

    # ----------------------------------------------------------------------------- walls

    def left_wall(self) -> np.ndarray:
        """Left wall vertices, ``(M, 2)``: ``p + nL * w_left``."""
        return self._pts + self._nl * self.data.w_left[:, None]

    def right_wall(self) -> np.ndarray:
        """Right wall vertices, ``(M, 2)``: ``p - nL * w_right``."""
        return self._pts - self._nl * self.data.w_right[:, None]

    # ------------------------------------------------------------------- interpolation

    def _locate(self, s) -> tuple[np.ndarray, np.ndarray]:
        """Segment index and fraction along it for arc length(s) ``s`` (wrapped)."""
        sm = np.mod(np.asarray(s, dtype=np.float64), self.length)
        sm = np.where(sm >= self.length, 0.0, sm)  # mod of a tiny negative can give length
        i = np.searchsorted(self._s, sm, side="right") - 1
        i = np.clip(i, 0, self._m - 1)
        t = (sm - self._s[i]) / self._ds[i]
        return i, t

    def _lerp(self, values: np.ndarray, s) -> np.ndarray:
        i, t = self._locate(s)
        j = (i + 1) % self._m
        return values[i] + t * (values[j] - values[i])

    def point_at(self, s) -> np.ndarray:
        """Centre-line point(s) at arc length ``s`` (wraps modulo length); shape ``(..., 2)``."""
        i, t = self._locate(s)
        return self._pts[i] + t[..., None] * self._d[i]

    def heading_at(self, s) -> np.ndarray:
        """Heading at ``s`` in ``[-pi, pi)``, interpolated as an angle."""
        i, t = self._locate(s)
        return _wrap_angle(self.data.heading[i] + t * self._dh[i])

    def curvature_at(self, s) -> np.ndarray:
        """Signed curvature at ``s`` (``> 0`` = turning left)."""
        return self._lerp(self.data.curvature, s)

    def widths_at(self, s) -> tuple[np.ndarray, np.ndarray]:
        """``(w_left, w_right)`` at ``s``."""
        return self._lerp(self.data.w_left, s), self._lerp(self.data.w_right, s)

    # ---------------------------------------------------------------------- projection

    def _nearest_global(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Index of the nearest segment (Euclidean) over the whole loop, chunked over points."""
        ax, ay = self._pts[:, 0], self._pts[:, 1]
        dx, dy = self._d[:, 0], self._d[:, 1]
        out = np.empty(px.size, dtype=np.intp)
        chunk = max(1, _PROJECT_CHUNK_ELEMS // self._m)
        for lo in range(0, px.size, chunk):
            hi = min(lo + chunk, px.size)
            rx = px[lo:hi, None] - ax[None, :]
            ry = py[lo:hi, None] - ay[None, :]
            t = np.clip((rx * dx + ry * dy) / self._len2, 0.0, 1.0)
            rx -= t * dx
            ry -= t * dy
            out[lo:hi] = np.argmin(rx * rx + ry * ry, axis=1)
        return out

    def _nearest_windowed(
        self, px: np.ndarray, py: np.ndarray, hint: np.ndarray, window: int
    ) -> np.ndarray:
        """Nearest segment (Euclidean) among ``hint ± window`` (wrapping)."""
        offs = np.arange(-window, window + 1)
        cand = (hint[:, None] + offs[None, :]) % self._m  # (P, 2W+1)
        a = self._pts[cand]
        d = self._d[cand]
        rx = px[:, None] - a[..., 0]
        ry = py[:, None] - a[..., 1]
        t = np.clip((rx * d[..., 0] + ry * d[..., 1]) / self._len2[cand], 0.0, 1.0)
        rx -= t * d[..., 0]
        ry -= t * d[..., 1]
        best = np.argmin(rx * rx + ry * ry, axis=1)
        return cand[np.arange(px.size), best]

    def _segment_fraction(self, idx: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Fraction ``t`` along segment ``idx`` whose interpolated normal passes through p.

        Solves ``(r - t d) x (N0 + t (N1 - N0)) = 0`` (a quadratic in ``t``) for the root near
        the plain chord projection.
        """
        a = self._pts[idx]
        d = self._d[idx]
        n0 = self._nl[idx]
        e = self._nl[(idx + 1) % self._m] - n0
        rx, ry = px - a[:, 0], py - a[:, 1]
        qa = -_cross(d[:, 0], d[:, 1], e[:, 0], e[:, 1])
        qb = _cross(rx, ry, e[:, 0], e[:, 1]) - _cross(d[:, 0], d[:, 1], n0[:, 0], n0[:, 1])
        qc = _cross(rx, ry, n0[:, 0], n0[:, 1])
        disc = np.sqrt(np.maximum(qb * qb - 4.0 * qa * qc, 0.0))
        q = -qb - np.where(qb >= 0.0, 1.0, -1.0) * disc
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(q) > 1e-300, 2.0 * qc / q, 0.0)
        return t

    def project(
        self, x, y, hint=None, window: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project points onto the centre line. Vectorised.

        Returns ``(s, n, idx)``: arc length in ``[0, length)``, signed lateral offset (positive
        to the left) and segment index. With ``hint=None`` the nearest segment is searched
        over the whole loop; with ``hint`` (array of segment indices, e.g. last step's ``idx``)
        only segments ``hint ± window`` are checked, so the cost per point is O(1).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x, y = np.broadcast_arrays(x, y)
        shape = x.shape
        px, py = x.ravel(), y.ravel()
        if hint is None:
            idx = self._nearest_global(px, py)
        else:
            h = np.broadcast_to(np.asarray(hint, dtype=np.intp), shape).ravel() % self._m
            idx = self._nearest_windowed(px, py, h, int(window))

        # The Euclidean pick is refined to the segment whose normal fan holds the point.
        eps = 1e-9
        for _ in range(4):
            t = self._segment_fraction(idx, px, py)
            step = np.where(t < -eps, -1, np.where(t > 1.0 + eps, 1, 0))
            if not np.any(step):
                break
            idx = (idx + step) % self._m
        t = np.clip(self._segment_fraction(idx, px, py), 0.0, 1.0)

        a = self._pts[idx]
        d = self._d[idx]
        n0 = self._nl[idx]
        nvec = n0 + t[:, None] * (self._nl[(idx + 1) % self._m] - n0)
        rel = np.column_stack([px, py]) - a - t[:, None] * d
        n = np.einsum("ij,ij->i", rel, nvec) / np.einsum("ij,ij->i", nvec, nvec)
        s = self._s[idx] + t * self._ds[idx]
        s = np.where(s >= self.length, s - self.length, s)
        return s.reshape(shape), n.reshape(shape), idx.reshape(shape)
