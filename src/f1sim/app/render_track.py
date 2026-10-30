"""Static track geometry for drawing: asphalt, kerbs, run-off, the chequered start/finish strip,
the white lines and the sector ticks.

The pure-numpy part (no window) builds everything in world metres:

- :func:`road_triangles`: the asphalt between the white lines (``Track.widths_at``).
- :func:`kerb_runs`, :func:`corner_kerb_runs` and :func:`kerb_geometry`: kerbs just outside
  the white line, in alternating red and white blocks of about ``KERB_BLOCK_M`` along it, and
  a run-off band from the kerb out to the limit. Two sources of kerbs are merged:

  - **Data:** where the track limits (``Track.limits_at``, from real drivers' laps) are more
    than ``KERB_MIN_EXTRA_M`` beyond the white line; ``min(limit - white,
    KERB_MAX_WIDTH_M)`` wide.
  - **Corners** (:func:`corners`): an apex kerb on the inside and an exit kerb on the
    outside of every corner of the centre line, at least ``CORNER_KERB_WIDTH_M`` wide.
    Drivers don't run over every real kerb, so the data alone left 10 of Silverstone's 19
    corners without an apex or exit kerb. This is drawing only: the limits the cars drive
    on don't change.
- :func:`chequer_triangles`: a strip of ~1 m black and white squares across the road at
  ``s = 0``.
- :func:`sector_ticks` and :func:`boundary_lines`: where the sector and start lines cross the
  white lines.

:class:`TrackRenderer` (needs an open window) uploads it once. The filled shapes are two
cached :class:`TrackMesh` es, one with kerbs and one where the kerbs are plain run-off (used
zoomed out, below ``KERB_MIN_PPM``, where 3 m blocks would only flicker); a mesh
uploads its vertices once and is drawn with one call under the window's
current projection and view, set by an ``arcade.camera.Camera2D`` (``position`` = the
:class:`~f1sim.app.camera.Camera` centre, ``zoom`` = pixels per metre), so panning and zooming
never rebuild geometry. The white lines are a static vertex buffer drawn by a small shader that
extrudes each vertex along its miter by ``WHITE_LINE_PX / zoom``, so they stay
``WHITE_LINE_PX`` wide (anti-aliased) on screen at every zoom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from f1sim.app.ui import theme
from f1sim.track.track import Track

ROAD_COLOUR = theme.ROAD
CHEQUER_DARK = (12, 12, 13, 255)
CHEQUER_LIGHT = (236, 236, 238, 255)

# ---------------------------------------------------------------------------- asphalt


def road_triangles(track: Track) -> np.ndarray:
    """Triangles filling the road between the white lines, shape ``(2M, 3, 2)``.

    For each wall segment pair ``i -> j = i + 1`` (including the closing pair ``M-1 -> 0``)
    the quad ``left[i], right[i], right[j], left[j]`` is split along ``right[i]-left[j]``
    into ``(left[i], right[i], left[j])`` and ``(right[i], right[j], left[j])``.
    """
    left = track.left_wall()
    right = track.right_wall()
    left_n = np.roll(left, -1, axis=0)
    right_n = np.roll(right, -1, axis=0)
    tri_a = np.stack([left, right, left_n], axis=1)
    tri_b = np.stack([right, right_n, left_n], axis=1)
    out = np.empty((2 * left.shape[0], 3, 2), dtype=np.float64)
    out[0::2] = tri_a
    out[1::2] = tri_b
    return out


# ------------------------------------------------------------------------------ kerbs


def kerb_runs(
    extra: np.ndarray,
    s: np.ndarray,
    length: float,
    threshold: float = theme.KERB_MIN_EXTRA_M,
    min_len: float = 4.0,
    merge_gap: float = 4.0,
) -> list[tuple[float, float]]:
    """Stretches of a closed track where ``extra`` (limit minus white line, per vertex at arc
    lengths ``s``) exceeds ``threshold``: ``(s0, s1)`` with ``0 <= s0 < length`` and
    ``s0 < s1`` (``s1`` past ``length`` for a stretch over the line), ends interpolated to
    the threshold crossing. Gaps shorter than ``merge_gap`` are bridged and stretches shorter
    than ``min_len`` dropped. The whole loop is ``[(0, length)]``."""
    extra = np.asarray(extra, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    above = extra > threshold
    if not above.any():
        return []
    if above.all():
        return [(0.0, float(length))]
    m = extra.size
    k = int(np.argmin(above))  # a vertex below the threshold: start the walk there
    order = (k + np.arange(m + 1)) % m
    su = s[order] + np.where(np.arange(m + 1) + k >= m, length, 0.0)
    su[-1] = s[k] + length  # back to the start vertex, one lap on
    e = extra[order]
    a = above[order]

    def crossing(i: int) -> float:  # threshold crossing between walk positions i and i + 1
        e0, e1 = e[i], e[i + 1]
        f = (threshold - e0) / (e1 - e0) if e1 != e0 else 0.5
        return float(su[i] + min(max(f, 0.0), 1.0) * (su[i + 1] - su[i]))

    runs: list[list[float]] = []
    for i in range(m):
        if not a[i] and a[i + 1]:
            runs.append([crossing(i), np.nan])
        elif a[i] and not a[i + 1]:
            runs[-1][1] = crossing(i)
    merged: list[list[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][0] + length - merged[-1][1] < merge_gap:
        merged[-1][1] = merged[0][1] + length  # bridge the gap over the start of the walk
        merged.pop(0)
    out = []
    for s0, s1 in merged:
        if s1 - s0 < min_len:
            continue
        shift = np.floor(s0 / length) * length
        out.append((float(s0 - shift), float(s1 - shift)))
    return sorted(out)


def corners(track: Track) -> list[tuple[float, float, float, int]]:
    """The corners of a closed track: ``(s_start, s_apex, s_end, sign)`` with ``sign`` +1 for
    a left-hander, -1 for a right-hander, from the smoothed centre-line curvature.

    A corner is a stretch of one sign tighter than ``CORNER_MAX_RADIUS_M`` that turns at
    least ``CORNER_MIN_TURN_DEG`` in total; its apex is where it's tightest. A track that is
    one bend all the way round (a circle) has no corners. ``s_apex`` and ``s_end`` may exceed
    the length for a corner over the line.
    """
    kappa, spacing = _smoothed_curvature(track)
    d = track.data
    length = track.length
    k_min = 1.0 / theme.CORNER_MAX_RADIUS_M
    sign = np.where(kappa > k_min, 1, np.where(kappa < -k_min, -1, 0))
    if not np.any(sign == 0):
        return []
    m = sign.size
    start = int(np.argmax(sign == 0))  # walk from a straight, so no corner is split
    order = (start + np.arange(m)) % m
    s_walk = d.s[order] + np.where(order < start, length, 0.0)
    out = []
    k = 0
    while k < m:
        sg = sign[order[k]]
        if sg == 0:
            k += 1
            continue
        j = k
        while j < m and sign[order[j]] == sg:
            j += 1
        idx = order[k:j]
        turn = float(np.sum(np.abs(kappa[idx])) * spacing)
        if math.degrees(turn) >= theme.CORNER_MIN_TURN_DEG:
            su = s_walk[k:j]
            apex = int(np.argmax(np.abs(kappa[idx])))
            out.append((float(su[0]), float(su[apex]), float(su[-1] + spacing), int(sg)))
        k = j
    return out


def _smoothed_curvature(track: Track) -> tuple[np.ndarray, float]:
    from scipy.ndimage import gaussian_filter1d

    d = track.data
    spacing = track.length / d.s.size
    return gaussian_filter1d(d.curvature, theme.CORNER_SMOOTH_M / spacing, mode="wrap"), spacing


def corner_kerb_runs(track: Track, side: str) -> list[tuple[float, float]]:
    """Apex kerbs (inside) and exit kerbs (outside) of every corner on ``side``, as
    ``(s0, s1)`` runs like :func:`kerb_runs` (overlaps merged).

    The apex kerb covers the part of the corner tighter than ``APEX_FRACTION`` of its peak,
    padded by ``APEX_PAD_M``; the exit kerb runs from ``EXIT_START`` of the way from apex to
    corner end until ``EXIT_EXTRA_M`` past the end. Both are at least ``CORNER_KERB_MIN_M``.
    """
    kappa, spacing = _smoothed_curvature(track)
    kappa = np.abs(kappa)
    d = track.data
    length = track.length
    want = 1 if side == "left" else -1
    runs = []
    for s0, apex, s1, sg in corners(track):
        if sg == want:  # inside: the tightest part around the apex
            ss = np.arange(s0, s1, spacing)
            k = np.interp(np.mod(ss, length), d.s, kappa)
            tight = ss[k >= theme.APEX_FRACTION * k.max()]
            a0, a1 = tight.min() - theme.APEX_PAD_M, tight.max() + theme.APEX_PAD_M
            mid = 0.5 * (a0 + a1)
            half = max(0.5 * (a1 - a0), 0.5 * theme.CORNER_KERB_MIN_M)
            runs.append((mid - half, mid + half))
        else:  # outside: from past the apex to just after the exit
            e0 = apex + theme.EXIT_START * (s1 - apex)
            e1 = max(s1 + theme.EXIT_EXTRA_M, e0 + theme.CORNER_KERB_MIN_M)
            runs.append((e0, e1))
    out = []
    for a, b in _union_runs(runs, [], length):
        shift = math.floor(a / length) * length
        out.append((a - shift, b - shift))
    return sorted(out)


def _union_runs(a, b, length: float) -> list[tuple[float, float]]:
    """Union of two lists of ``(s0, s1)`` runs on a loop of ``length``."""
    merged: list[list[float]] = []
    for s0, s1 in sorted([*a, *b]):
        if merged and s0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s1)
        else:
            merged.append([s0, s1])
    if len(merged) > 1 and merged[-1][1] >= merged[0][0] + length:  # wraps onto the first
        merged[-1][1] = max(merged[-1][1], merged[0][1] + length)
        merged.pop(0)
    return [(float(s0), float(s1)) for s0, s1 in merged]


def _in_runs(ss: np.ndarray, runs, length: float) -> np.ndarray:
    m = np.zeros(ss.size, dtype=bool)
    for s0, s1 in runs:
        m |= np.mod(ss - s0, length) <= (s1 - s0)
    return m


def block_edges(run_length: float, block_m: float = theme.KERB_BLOCK_M) -> np.ndarray:
    """Edges of the kerb blocks along a kerb ``run_length`` long: a whole number of equal
    blocks as close to ``block_m`` as possible (at least one)."""
    n = max(1, int(round(run_length / block_m)))
    return np.linspace(0.0, run_length, n + 1)


def _lerp_loop(values: np.ndarray, s_vertices: np.ndarray, length: float, s) -> np.ndarray:
    """Per-vertex ``values`` (``(M,)`` or ``(M, k)``) linearly interpolated at arc lengths
    ``s`` on the closed loop (wrapped)."""
    sm = np.mod(np.asarray(s, dtype=np.float64), length)
    i = np.clip(np.searchsorted(s_vertices, sm, side="right") - 1, 0, s_vertices.size - 1)
    j = (i + 1) % s_vertices.size
    seg = np.where(j == 0, length - s_vertices[i], s_vertices[j] - s_vertices[i])
    t = (sm - s_vertices[i]) / seg
    if values.ndim > 1:
        t = t[:, None]
    return values[i] + t * (values[j] - values[i])


@dataclass(frozen=True)
class KerbGeometry:
    """Kerbs of one side of a track, in world metres.

    ``kerb`` ``(N, 3, 2)`` triangles with ``red`` ``(N,)`` (True = red block, False = white);
    ``runoff`` ``(K, 3, 2)`` triangles from the kerbs' outer edge to the limit; ``plain``
    ``(P, 3, 2)`` the whole band from the white line to the limit (the far-zoom version);
    ``runs`` the kerbs' ``(s0, s1)``; ``block_lengths`` the length of each block along the
    white line, run by run.
    """

    kerb: np.ndarray
    red: np.ndarray
    runoff: np.ndarray
    plain: np.ndarray
    runs: list[tuple[float, float]]
    block_lengths: list[np.ndarray]


def _quads(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Triangles of the strip between polylines ``a`` and ``b`` (both ``(n, 2)``), two per
    segment: ``(a_i, a_j, b_i)`` and ``(a_j, b_j, b_i)``."""
    t1 = np.stack([a[:-1], a[1:], b[:-1]], axis=1)
    t2 = np.stack([a[1:], b[1:], b[:-1]], axis=1)
    out = np.empty((2 * t1.shape[0], 3, 2))
    out[0::2] = t1
    out[1::2] = t2
    return out


def kerb_geometry(
    track: Track,
    side: str,
    block_m: float = theme.KERB_BLOCK_M,
    max_width: float = theme.KERB_MAX_WIDTH_M,
    threshold: float = theme.KERB_MIN_EXTRA_M,
    corner_kerbs: bool = True,
) -> KerbGeometry:
    """The kerbs on ``side`` (``"left"`` or ``"right"``) of ``track``.

    Each kerb starts on the white line and is ``min(limit - white, max_width)`` wide, and at
    least ``CORNER_KERB_WIDTH_M`` where there's a corner kerb (:func:`corner_kerb_runs`; off
    with ``corner_kerbs=False``). Its blocks are measured along the white line, so they look
    the same length on both sides of a bend. The kerb's inner edge lies exactly on the white
    line's polyline. A track without limits gets only its corner kerbs.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    empty = np.zeros((0, 3, 2))
    d = track.data
    zeros = np.zeros(d.s.size)
    if side == "left":
        white = track.left_wall()
        extra = d.limit_left - d.w_left if track.has_limits else zeros
        normal = track._nl  # outwards on the left (the walls are built along it)
    else:
        white = track.right_wall()
        extra = d.limit_right - d.w_right if track.has_limits else zeros
        normal = -track._nl
    length = track.length
    geo_runs = corner_kerb_runs(track, side) if corner_kerbs else []
    runs = _union_runs(kerb_runs(extra, d.s, length, threshold), geo_runs, length)
    if not runs:
        return KerbGeometry(empty, np.zeros(0, dtype=bool), empty, empty, [], [])
    kerb, red, runoff, plain, blocks = [], [], [], [], []
    for s0, s1 in runs:
        inner_s = d.s[(d.s > s0) & (d.s < s1)]
        inner_s = np.concatenate([inner_s, d.s[d.s + length < s1] + length])  # over the line
        base = np.unique(np.concatenate([[s0], inner_s[(inner_s > s0) & (inner_s < s1)], [s1]]))
        w = _lerp_loop(white, d.s, length, base)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(w, axis=0).T))])
        edges_a = block_edges(arc[-1], block_m)
        edges_s = np.interp(edges_a, arc, base)
        ss = np.unique(np.concatenate([base, edges_s]))
        w = _lerp_loop(white, d.s, length, ss)
        nrm = _lerp_loop(normal, d.s, length, ss)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        e = np.maximum(_lerp_loop(extra, d.s, length, ss), 0.0)
        k = np.minimum(e, max_width)
        if geo_runs:
            on_corner = _in_runs(ss, geo_runs, length)
            k = np.where(on_corner, np.maximum(k, theme.CORNER_KERB_WIDTH_M), k)
        k = np.maximum(k, 1e-3)
        outer = w + nrm * k[:, None]
        lim = w + nrm * np.maximum(e, k)[:, None]  # the limit, or the kerb's edge if beyond
        tris = _quads(w, outer)
        mid = 0.5 * (ss[:-1] + ss[1:])
        block = np.clip(np.searchsorted(edges_s, mid, side="right") - 1, 0, edges_s.size - 2)
        kerb.append(tris)
        red.append(np.repeat(block % 2 == 0, 2))
        wide = (e - k) > 1e-3
        seg = wide[:-1] | wide[1:]
        if seg.any():
            ro = _quads(outer, lim).reshape(-1, 2, 3, 2)[seg].reshape(-1, 3, 2)
            runoff.append(ro)
        plain.append(_quads(w, lim))
        blocks.append(np.diff(edges_a))

    def cat(parts):
        return np.concatenate(parts) if parts else empty

    return KerbGeometry(
        kerb=cat(kerb),
        red=np.concatenate(red) if red else np.zeros(0, dtype=bool),
        runoff=cat(runoff),
        plain=cat(plain),
        runs=runs,
        block_lengths=blocks,
    )


# ------------------------------------------------------------------- lines and strip


def _cross_line(track: Track, s: float) -> np.ndarray:
    """Line across the road at ``s``: ``[[right wall point], [left wall point]]``."""
    p = track.point_at(s)
    h = float(track.heading_at(s))
    w_left, w_right = track.widths_at(s)
    normal = np.array([-np.sin(h), np.cos(h)])
    return np.stack([p - normal * float(w_right), p + normal * float(w_left)])


def boundary_lines(track: Track) -> dict[str, np.ndarray]:
    """Start/finish and sector boundary lines across the road, from right wall to left wall.

    ``"start_finish"``: ``(2, 2)`` at ``s = 0``. ``"sectors"``: ``(k, 2, 2)``, one per entry of
    ``track.data.sectors_s[1:]`` (``k = 0`` when the track has no sectors).
    """
    sectors = [_cross_line(track, s) for s in track.data.sectors_s[1:]]
    return {
        "start_finish": _cross_line(track, 0.0),
        "sectors": np.array(sectors, dtype=np.float64).reshape(len(sectors), 2, 2),
    }


def sector_ticks(track: Track) -> np.ndarray:
    """``(k, 2, 2)``: for each sector boundary and side, the point on the white line and the
    unit direction pointing away from the road (left side first)."""
    out = []
    for s in track.data.sectors_s[1:]:
        h = float(track.heading_at(s))
        normal = np.array([-np.sin(h), np.cos(h)])
        right_pt, left_pt = _cross_line(track, s)
        out.append([left_pt, normal])
        out.append([right_pt, -normal])
    return np.array(out, dtype=np.float64).reshape(len(out), 2, 2)


def chequer_triangles(
    track: Track, square_m: float = theme.CHEQUER_M, rows: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """A chequered strip across the road centred on ``s = 0``: ``rows`` squares deep along
    the track and a whole number of ~``square_m`` squares across it. Returns triangles
    ``(2 * rows * cols, 3, 2)`` and whether each is a light square."""
    w_left, w_right = (float(v) for v in track.widths_at(0.0))
    cols = max(2, int(round((w_left + w_right) / square_m)))
    n_edges = np.linspace(-w_right, w_left, cols + 1)
    s_edges = (np.arange(rows + 1) - 0.5 * rows) * square_m
    p = track.point_at(s_edges)
    h = track.heading_at(s_edges)
    normal = np.column_stack([-np.sin(h), np.cos(h)])
    grid = p[:, None, :] + normal[:, None, :] * n_edges[None, :, None]  # (rows+1, cols+1, 2)
    tris, light = [], []
    for r in range(rows):
        for c in range(cols):
            a, b = grid[r, c], grid[r, c + 1]
            d, e = grid[r + 1, c], grid[r + 1, c + 1]
            tris += [[a, b, d], [b, e, d]]
            light += [(r + c) % 2 == 0] * 2
    return np.array(tris), np.array(light)


def miter_offsets(points: np.ndarray, closed: bool = True) -> np.ndarray:
    """Per-vertex extrusion vectors ``(n, 2)`` for a polyline of constant width: the unit
    bisector of the neighbouring segments' left normals, divided by the cosine to them (at
    most 2x long at sharp corners), so the strip's edges stay parallel to every segment."""
    p = np.asarray(points, dtype=np.float64)
    if closed:
        prev = np.roll(p, 1, axis=0)
        nxt = np.roll(p, -1, axis=0)
    else:
        prev = np.vstack([2 * p[0] - p[1], p[:-1]])
        nxt = np.vstack([p[1:], 2 * p[-1] - p[-2]])

    def unit(v):
        n = np.hypot(v[:, 0], v[:, 1])
        return v / np.where(n > 0, n, 1.0)[:, None]

    t0 = unit(p - prev)
    t1 = unit(nxt - p)
    n0 = np.column_stack([-t0[:, 1], t0[:, 0]])
    n1 = np.column_stack([-t1[:, 1], t1[:, 0]])
    m = unit(n0 + n1)
    cos = np.maximum(np.einsum("ij,ij->i", m, n1), 0.5)
    return m / cos[:, None]


def line_vertices(points: np.ndarray, closed: bool = True) -> np.ndarray:
    """Vertices ``(6 * segments, 5)`` of a constant-screen-width line: ``x, y`` (world),
    the miter vector and the side (+1 left, -1 right) of each corner of each segment's two
    triangles. The shader moves each vertex ``side * miter * half_width_px / zoom``."""
    p = np.asarray(points, dtype=np.float64)
    off = miter_offsets(p, closed)
    i = np.arange(p.shape[0] if closed else p.shape[0] - 1)
    j = (i + 1) % p.shape[0]
    corners = [(i, 1.0), (i, -1.0), (j, 1.0), (j, 1.0), (i, -1.0), (j, -1.0)]
    out = np.empty((i.size, 6, 5))
    for c, (idx, side) in enumerate(corners):
        out[:, c, 0:2] = p[idx]
        out[:, c, 2:4] = off[idx]
        out[:, c, 4] = side
    return out.reshape(-1, 5)


# ------------------------------------------------------------------------- GL side

_LINE_VS = """
#version 330
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
uniform float u_extent;   // half width + 1 px of anti-aliasing, screen pixels
uniform float u_ppm;      // screen pixels per world metre
in vec2 in_pos;
in vec2 in_off;
in float in_side;
out float v_d;
void main() {
    vec2 p = in_pos + in_off * in_side * u_extent / u_ppm;
    v_d = in_side * u_extent;
    gl_Position = window.projection * window.view * vec4(p, 0.0, 1.0);
}
"""

_LINE_FS = """
#version 330
uniform float u_half;
uniform vec4 u_colour;
in float v_d;
out vec4 f_color;
void main() {
    f_color = vec4(u_colour.rgb, u_colour.a * clamp(u_half + 0.5 - abs(v_d), 0.0, 1.0));
}
"""


def _colours(colours, n_tris: int) -> np.ndarray:
    """``(3 * n_tris, 4)`` vertex colours (0-255) from one colour or one per triangle."""
    c = np.asarray(colours, dtype=np.float32)
    if c.ndim == 1:
        c = np.broadcast_to(c, (n_tris, c.size))
    if c.shape[1] == 3:
        c = np.hstack([c, np.full((n_tris, 1), 255.0, np.float32)])
    return np.repeat(c, 3, axis=0)


def track_mesh_data(track: Track, kerbs: bool = True) -> np.ndarray:
    """The filled shapes as one ``(N, 6)`` float32 array of triangle vertices, ``x, y`` in
    metres then ``r, g, b, a`` (0-255), in drawing order: run-off, kerbs (red/white blocks, or
    plain run-off with ``kerbs=False``), asphalt and the chequered strip."""
    parts = []

    def add(tris, colours) -> None:
        tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 2)
        if len(tris):
            parts.append(np.hstack([tris.reshape(-1, 2), _colours(colours, len(tris))]))

    for side in ("left", "right"):
        g = kerb_geometry(track, side)
        if kerbs:
            add(g.runoff, theme.RUNOFF)
            add(g.kerb, [theme.KERB_RED if r else theme.KERB_WHITE for r in g.red])
        else:
            add(g.plain, theme.RUNOFF)
    add(road_triangles(track), ROAD_COLOUR)
    tris, light = chequer_triangles(track)
    add(tris, [CHEQUER_LIGHT if v else CHEQUER_DARK for v in light])
    return np.vstack(parts).astype(np.float32) if parts else np.zeros((0, 6), np.float32)


class TrackMesh:
    """The filled shapes, uploaded once and drawn in one call with arcade's shape-list
    program (so under the window's projection and view, like a ``ShapeElementList``). Built
    from numpy arrays: a ``ShapeElementList`` converted each of the ~77k vertex colours in
    Python, a third of a second of every screen's build (D51)."""

    def __init__(self, ctx, data: np.ndarray):
        from arcade.gl import BufferDescription

        self.ctx = ctx
        self.vertices = len(data)
        self.program = ctx.shape_element_list_program
        self._buf = ctx.buffer(data=np.ascontiguousarray(data, dtype=np.float32).tobytes())
        self._geo = ctx.geometry(
            [BufferDescription(self._buf, "2f 4f", ("in_vert", "in_color"))],
            mode=ctx.TRIANGLES,
        )

    def draw(self) -> None:
        if not self.vertices:
            return
        self.program["Position"] = 0.0, 0.0
        self.program["Angle"] = 0.0
        self.ctx.enable_only(self.ctx.BLEND)
        self._geo.render(self.program)
        self.ctx.disable(self.ctx.BLEND)


def build_track_shapes(track: Track, kerbs: bool = True, ctx=None) -> TrackMesh:
    """The filled shapes (:func:`track_mesh_data`) uploaded to ``ctx`` (default: the
    window's)."""
    if ctx is None:
        import arcade

        ctx = arcade.get_window().ctx
    return TrackMesh(ctx, track_mesh_data(track, kerbs))


class TrackRenderer:
    """The whole static track of a viewer, uploaded once. Draw it under the world camera."""

    def __init__(self, track: Track, ctx):
        from arcade.gl import BufferDescription

        self.track = track
        self.ctx = ctx
        self.near = build_track_shapes(track, kerbs=True, ctx=ctx)
        far = track.has_limits
        self.far = build_track_shapes(track, kerbs=False, ctx=ctx) if far else self.near
        verts = np.vstack(
            [line_vertices(track.left_wall()), line_vertices(track.right_wall())]
        ).astype("f4")
        self._line_buf = ctx.buffer(data=verts.tobytes())
        self._line_geo = ctx.geometry(
            [BufferDescription(self._line_buf, "2f 2f 1f", ["in_pos", "in_off", "in_side"])],
            mode=ctx.TRIANGLES,
        )
        self._line_prog = ctx.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        c = theme.WHITE_LINE
        self._line_prog["u_colour"] = (c[0] / 255, c[1] / 255, c[2] / 255, c[3] / 255)
        self._line_prog["u_half"] = 0.5 * theme.WHITE_LINE_PX
        self._line_prog["u_extent"] = 0.5 * theme.WHITE_LINE_PX + 1.0
        self.ticks = sector_ticks(track)

    def draw(self, pixels_per_metre: float) -> None:
        """Filled shapes and white lines, under a ``Camera2D`` at ``pixels_per_metre``."""
        shapes = self.near if pixels_per_metre >= theme.KERB_MIN_PPM else self.far
        shapes.draw()
        self._line_prog["u_ppm"] = float(pixels_per_metre)
        self.ctx.enable(self.ctx.BLEND)
        self._line_geo.render(self._line_prog)

    def draw_ticks(self, camera, glass) -> None:
        """The red sector ticks across the white lines, in screen pixels (under the pixel
        camera): ``SECTOR_TICK_PX`` wide and long, mostly outside the road."""
        width, length = theme.SECTOR_TICK_PX
        for (x, y), (dx, dy) in self.ticks:
            sx, sy = camera.world_to_screen(x, y)
            sx, sy = float(sx), float(sy)
            a = 0.3 * length
            b = 0.7 * length
            glass.stroke(
                sx - dx * a, sy - dy * a, sx + dx * b, sy + dy * b, width, theme.SECTOR_TICK
            )
