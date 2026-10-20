"""Uniform grid over wall segments for fast, vectorised ray casting.

Each grid cell lists the segments that touch it (a conservative rasterisation: every segment is
cut into pieces no longer than a cell, and each piece registers in every cell its slightly
padded bounding box overlaps, so no touching cell is ever missed). :meth:`SegmentGrid.raycast`
walks every ray through the cells it crosses, in order (an Amanatides-Woo style DDA computed
for all rays at once with numpy), tests only the segments in those cells, and stops a ray as
soon as its nearest hit is no farther than the exit of the last cell it visited. Rays are
processed a few cells at a time, all still-active rays together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from f1sim.track.track import Track

# Tolerances of the ray-segment test. A hit exactly at a segment end point counts (u in [0, 1]
# padded by _EPS_U), and a ray starting on a wall hits it at distance ~0 (t >= -_EPS_T).
_EPS_U = 1e-9
_EPS_T = 1e-9
# |cross(d, e)| at or below this (metres, |d| = 1) is treated as parallel.
_EPS_PARALLEL = 1e-12
# Rasterisation padding, in cells: segments within this of a cell boundary register on both
# sides, so rounding in the ray walk can never skip the cell holding a hit.
_RASTER_PAD = 1e-6
# Cells walked per batch step before checking which rays can stop: a short first step (most
# rays hit a wall within a cell or two), longer ones after (fewer numpy round trips).
_CHUNK_CELLS = (2, 4, 8)


class SegmentGrid:
    """Uniform grid of wall segments supporting batched ray casts.

    ``segments`` has shape ``(K, 2, 2)``: ``segments[k] = [[x0, y0], [x1, y1]]``.

    The default cell size, 10 m, was the fastest measured on Silverstone (2 m wall segments,
    roads ~12 m wide): about 5 segments per occupied cell, so the cells a ray walks hug it
    closely. 25 m cells hold ~3x as many segments; that matched 10 m for 64 cars x 9 rays but
    was about 1.7x slower for 1024 cars. Much smaller cells cost more batch steps per ray.
    """

    def __init__(self, segments: np.ndarray, cell_size_m: float = 10.0):
        seg = np.array(segments, dtype=np.float64)
        if seg.ndim != 3 or seg.shape[1:] != (2, 2) or seg.shape[0] < 1:
            raise ValueError(f"segments must have shape (K, 2, 2) with K >= 1, got {seg.shape}")
        if not np.all(np.isfinite(seg)):
            raise ValueError("segments contain non-finite values")
        cell = float(cell_size_m)
        if not (np.isfinite(cell) and cell > 0.0):
            raise ValueError(f"cell_size_m must be positive and finite, got {cell_size_m}")
        seg.setflags(write=False)
        self.segments = seg
        self.cell_size = cell

        p, q = seg[:, 0], seg[:, 1]
        lo = np.minimum(p, q).min(axis=0)
        hi = np.maximum(p, q).max(axis=0)
        margin = 0.5 * cell
        self.x0 = float(lo[0] - margin)
        self.y0 = float(lo[1] - margin)
        self.nx = int(np.ceil((hi[0] + margin - self.x0) / cell))
        self.ny = int(np.ceil((hi[1] + margin - self.y0) / cell))
        self.x1 = self.x0 + self.nx * cell
        self.y1 = self.y0 + self.ny * cell
        self.n_cells = self.nx * self.ny

        # Segment data for the ray test, one row per quantity: px, py, ex, ey.
        self._seg = np.ascontiguousarray(
            np.vstack([p[:, 0], p[:, 1], q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]])
        )
        self._build_cells()

    @classmethod
    def from_track(cls, track: Track, cell_size_m: float = 10.0) -> SegmentGrid:
        """Grid over both walls of ``track``, each a closed loop."""
        parts = []
        for wall in (track.left_wall(), track.right_wall()):
            w = np.asarray(wall, dtype=np.float64)
            parts.append(np.stack([w, np.roll(w, -1, axis=0)], axis=1))
        return cls(np.concatenate(parts, axis=0), cell_size_m)

    # --------------------------------------------------------------------- rasterisation

    def _build_cells(self) -> None:
        """CSR cell -> segment table; cell id ``ix * ny + iy``, plus one empty cell at the end."""
        seg, c = self.segments, self.cell_size
        p, q = seg[:, 0], seg[:, 1]
        length = np.hypot(*(q - p).T)
        pieces = np.maximum(1, np.ceil(length / c)).astype(np.int64)
        seg_id = np.repeat(np.arange(seg.shape[0]), pieces)
        k = np.arange(seg_id.size) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        npc = pieces[seg_id]
        d = q[seg_id] - p[seg_id]
        a = p[seg_id] + d * (k / npc)[:, None]
        b = p[seg_id] + d * ((k + 1) / npc)[:, None]
        origin = np.array([self.x0, self.y0])
        lo = (np.minimum(a, b) - origin) / c - _RASTER_PAD
        hi = (np.maximum(a, b) - origin) / c + _RASTER_PAD
        dims = np.array([self.nx, self.ny])
        i0 = np.clip(np.floor(lo).astype(np.int64), 0, dims - 1)
        i1 = np.clip(np.floor(hi).astype(np.int64), 0, dims - 1)
        span = i1 - i0
        cells, segs = [], []
        for di in range(int(span[:, 0].max()) + 1):
            for dj in range(int(span[:, 1].max()) + 1):
                m = (span[:, 0] >= di) & (span[:, 1] >= dj)
                cells.append((i0[m, 0] + di) * self.ny + (i0[m, 1] + dj))
                segs.append(seg_id[m])
        key = np.unique(np.concatenate(cells) * seg.shape[0] + np.concatenate(segs))
        cell_of = key // seg.shape[0]
        self._cell_segs = (key % seg.shape[0]).astype(np.int64)
        count = np.bincount(cell_of, minlength=self.n_cells + 1)
        self._cell_count = count.astype(np.int64)
        self._cell_start = (np.cumsum(count) - count).astype(np.int64)
        self.max_segments_per_cell = int(count.max())

    # --------------------------------------------------------------------------- raycast

    def raycast(self, ox, oy, angles, max_dist: float) -> np.ndarray:
        """Distance from ``(ox, oy)`` along each absolute angle to the first wall hit.

        ``ox``, ``oy`` have shape ``(N,)`` and ``angles`` ``(N, R)`` (radians). Returns
        ``(N, R)`` float64 distances, ``max_dist`` where nothing is hit within ``max_dist``
        (``nan`` for rays with a non-finite origin or angle). Origins may lie outside the grid.
        """
        ox = np.asarray(ox, dtype=np.float64)
        oy = np.asarray(oy, dtype=np.float64)
        ang = np.asarray(angles, dtype=np.float64)
        if ox.ndim != 1 or oy.shape != ox.shape:
            raise ValueError(f"ox and oy must both have shape (N,), got {ox.shape}, {oy.shape}")
        if ang.ndim != 2 or ang.shape[0] != ox.shape[0]:
            raise ValueError(f"angles must have shape (N={ox.shape[0]}, R), got {ang.shape}")
        max_dist = float(max_dist)
        if not (np.isfinite(max_dist) and max_dist > 0.0):
            raise ValueError(f"max_dist must be positive and finite, got {max_dist}")
        n, r = ang.shape
        out = np.full(n * r, max_dist)
        if out.size == 0:
            return out.reshape(n, r)
        rox = np.repeat(ox, r)
        roy = np.repeat(oy, r)
        th = ang.ravel()
        dx = np.cos(th)
        dy = np.sin(th)
        good = np.isfinite(rox) & np.isfinite(roy) & np.isfinite(th)
        out[~good] = np.nan

        t_in, t_out = self._clip_to_grid(rox, roy, dx, dy, max_dist)
        live = np.flatnonzero(good & (t_in <= t_out))
        if live.size:
            best = self._walk(rox[live], roy[live], dx[live], dy[live], t_in[live], t_out[live])
            out[live] = np.minimum(best, max_dist)
        return out.reshape(n, r)

    def _clip_to_grid(self, ox, oy, dx, dy, max_dist):
        """Parametric interval ``[t_in, t_out]`` of each ray inside the grid box and range."""
        t_in = np.zeros(ox.size)
        t_out = np.full(ox.size, max_dist)
        for o, d, lo, hi in ((ox, dx, self.x0, self.x1), (oy, dy, self.y0, self.y1)):
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo - o) / d
                tb = (hi - o) / d
            flat = d == 0.0
            inside = (o >= lo) & (o <= hi)
            tmin = np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
            tmax = np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
            t_in = np.maximum(t_in, tmin)
            t_out = np.minimum(t_out, tmax)
        return t_in, t_out

    def _walk(self, ox, oy, dx, dy, t_in, t_out) -> np.ndarray:
        """Nearest hit distance per ray (``inf`` if none), walking cells in batches.

        Each batch step, every still-active ray takes its next few cells (``_CHUNK_CELLS``; its
        upcoming x and y boundary crossings merged in order, as in a DDA), tests their
        segments, and stops once its nearest hit is no farther than the exit of the last cell
        visited (any nearer hit lies in a visited cell), or once it leaves the range or grid.
        """
        c = self.cell_size
        ix = np.clip(np.floor((ox + dx * t_in - self.x0) / c).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.floor((oy + dy * t_in - self.y0) / c).astype(np.int64), 0, self.ny - 1)
        sx = np.where(dx >= 0.0, 1, -1)
        sy = np.where(dy >= 0.0, 1, -1)
        # The ray crosses boundary line x0 + i c at t = cx0 + i * cx1 (never if parallel).
        with np.errstate(divide="ignore"):
            inv_x = 1.0 / dx
            inv_y = 1.0 / dy
        cx0 = np.where(dx == 0.0, np.inf, (self.x0 - ox) * inv_x)
        cx1 = np.where(dx == 0.0, 0.0, c * inv_x)
        cy0 = np.where(dy == 0.0, np.inf, (self.y0 - oy) * inv_y)
        cy1 = np.where(dy == 0.0, 0.0, c * inv_y)
        # Per-ray state, one row per quantity, compacted to the active rays after each batch.
        fs = np.vstack([ox, oy, dx, dy, cx0, cx1, cy0, cy1, t_in, t_out])
        ist = np.vstack([ix, iy, sx, sy, np.arange(ox.size)])
        best = np.full(ox.size, np.inf)
        step = 0
        while ist.shape[1]:
            b = _CHUNK_CELLS[min(step, len(_CHUNK_CELLS) - 1)]
            step += 1
            k = np.arange(b)
            ix, iy, sx, sy, ray = ist
            # Next b crossings on each axis; the first b of the merged list bound the b cells.
            tx = fs[4, :, None] + (ix + (sx > 0))[:, None] * fs[5, :, None]
            tx = tx + (sx[:, None] * k) * fs[5, :, None]
            ty = fs[6, :, None] + (iy + (sy > 0))[:, None] * fs[7, :, None]
            ty = ty + (sy[:, None] * k) * fs[7, :, None]
            t_all = np.concatenate([tx, ty], axis=1)
            order = np.argsort(t_all, axis=1, kind="stable")[:, :b]
            t_exit = np.take_along_axis(t_all, order, axis=1)
            n_x = np.cumsum(order < b, axis=1)  # x crossings among the first 1..b events
            cx = np.empty_like(n_x)
            cy = np.empty_like(n_x)
            cx[:, 0] = ix
            cy[:, 0] = iy
            cx[:, 1:] = ix[:, None] + sx[:, None] * n_x[:, :-1]
            cy[:, 1:] = iy[:, None] + sy[:, None] * (k[1:] - n_x[:, :-1])
            t_enter = np.empty_like(t_exit)
            t_enter[:, 0] = fs[8]
            t_enter[:, 1:] = t_exit[:, :-1]
            valid = (t_enter <= fs[9, :, None]) & (cx >= 0) & (cx < self.nx)
            valid &= (cy >= 0) & (cy < self.ny)
            cells = np.where(valid, cx * self.ny + cy, self.n_cells)
            hit = np.minimum(best[ray], self._test_cells(fs[:4], cells))
            best[ray] = hit
            # Step past the batch: the next cell is entered at the last crossing.
            ix = ix + sx * n_x[:, -1]
            iy = iy + sy * (b - n_x[:, -1])
            t_last = t_exit[:, -1]
            more = (hit > t_last) & (t_last <= fs[9]) & (ix >= 0) & (ix < self.nx)
            more &= (iy >= 0) & (iy < self.ny)
            fs = fs[:, more]
            fs[8] = t_last[more]
            ist = np.vstack([ix, iy, sx, sy, ray])[:, more]
        return best

    def _test_cells(self, rays: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Nearest hit per ray against the segments of its ``cells`` row; ``inf`` if none."""
        a = cells.shape[0]
        hit = np.full(a, np.inf)
        cnt = self._cell_count[cells]
        per_ray = cnt.sum(axis=1)
        total = int(per_ray.sum())
        if total == 0:
            return hit
        flat_cnt = cnt.ravel()
        offset = self._cell_start[cells.ravel()] - (np.cumsum(flat_cnt) - flat_cnt)
        seg_idx = self._cell_segs[np.arange(total) + np.repeat(offset, flat_cnt)]
        px, py, ex, ey = np.take(self._seg, seg_idx, axis=1)
        ox, oy, dx, dy = np.take(rays, np.repeat(np.arange(a), per_ray), axis=1)
        t = _ray_segment_t(ox, oy, dx, dy, px, py, ex, ey)
        has = per_ray > 0
        first = (np.cumsum(per_ray) - per_ray)[has]
        hit[has] = np.minimum.reduceat(t, first)
        return hit


def _ray_segment_t(ox, oy, dx, dy, px, py, ex, ey) -> np.ndarray:
    """Distance along unit rays ``o + t d`` to segments ``p + u e``; ``inf`` where missed.

    End points count as hits; a ray starting on a segment gets ``t = 0``. A segment collinear
    with the ray is hit at its nearest point ahead of the origin (``0`` if the origin is on it).
    """
    wx = px - ox
    wy = py - oy
    denom = dx * ey - dy * ex
    nu = wx * dy - wy * dx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / denom
        u = nu / denom
    ok = (u >= -_EPS_U) & (u <= 1.0 + _EPS_U) & (t >= -_EPS_T)
    t = np.where(ok, np.maximum(t, 0.0), np.inf)
    par = np.flatnonzero(np.abs(denom) <= _EPS_PARALLEL)
    if par.size:
        on_line = np.abs(nu[par]) <= _EPS_T
        ta = wx[par] * dx[par] + wy[par] * dy[par]
        tb = ta + ex[par] * dx[par] + ey[par] * dy[par]
        near = np.minimum(ta, tb)
        far = np.maximum(ta, tb)
        t[par] = np.where(on_line & (far >= -_EPS_T), np.maximum(near, 0.0), np.inf)
    return t
