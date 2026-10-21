"""Static track geometry for drawing: asphalt, kerbs, run-off, the chequered start/finish strip,
the white lines and the sector ticks.

The pure-numpy part (no window) builds everything in world metres:

- :func:`road_triangles`: the asphalt between the white lines (``Track.widths_at``).
- :func:`kerb_runs` and :func:`kerb_geometry`: where the track limits (``Track.limits_at``,
  from real drivers' laps) are more than ``KERB_MIN_EXTRA_M`` beyond the white line, a kerb
  band ``min(limit - white, KERB_MAX_WIDTH_M)`` wide just outside the white line, in
  alternating red and white blocks of about ``KERB_BLOCK_M`` along the white line, and a
  run-off band from the kerb out to the limit. Tracks without limits have no kerbs.
- :func:`chequer_triangles`: a strip of ~1 m black and white squares across the road at
  ``s = 0``.
- :func:`sector_ticks` and :func:`boundary_lines`: where the sector and start lines cross the
  white lines.

:class:`TrackRenderer` (needs an open window) uploads it once. The filled shapes are two
cached ``ShapeElementList`` s, one with kerbs and one where the kerbs are plain run-off (used
zoomed out, below ``KERB_MIN_PPM``, where 3 m blocks would only flicker); a ``ShapeElementList``
uploads its vertices once and is drawn with one call per primitive mode under the window's
current projection and view, set by an ``arcade.camera.Camera2D`` (``position`` = the
:class:`~f1sim.app.camera.Camera` centre, ``zoom`` = pixels per metre), so panning and zooming
never rebuild geometry. The white lines are a static vertex buffer drawn by a small shader that
extrudes each vertex along its miter by ``WHITE_LINE_PX / zoom``, so they stay
``WHITE_LINE_PX`` wide (anti-aliased) on screen at every zoom.
"""

from __future__ import annotations

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
) -> KerbGeometry:
    """The kerbs on ``side`` (``"left"`` or ``"right"``) of ``track``; empty without limits.

    Each kerb starts on the white line and is ``min(limit - white, max_width)`` wide; its
    blocks are measured along the white line, so they look the same length on both sides of
    a bend. The kerb's inner edge lies exactly on the white line's polyline.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    empty = np.zeros((0, 3, 2))
    if not track.has_limits:
        return KerbGeometry(empty, np.zeros(0, dtype=bool), empty, empty, [], [])
    d = track.data
    if side == "left":
        white, limit = track.left_wall(), track.limit_left_wall()
        extra = d.limit_left - d.w_left
    else:
        white, limit = track.right_wall(), track.limit_right_wall()
        extra = d.limit_right - d.w_right
    length = track.length
    runs = kerb_runs(extra, d.s, length, threshold)
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
        lim = _lerp_loop(limit, d.s, length, ss)
        e = np.maximum(_lerp_loop(extra, d.s, length, ss), 1e-9)
        k = np.minimum(e, max_width)
        outer = w + (lim - w) * (k / e)[:, None]
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


def _flat(tris: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in np.asarray(tris).reshape(-1, 2)]


def _append(shapes, tris: np.ndarray, colours) -> None:
    from arcade import shape_list

    if len(tris) == 0:
        return
    pts = _flat(tris)
    if isinstance(colours, tuple):
        cols = [colours] * len(pts)
    else:
        cols = [c for c in colours for _ in range(3)]
    shapes.append(shape_list.create_triangles_filled_with_colors(pts, cols))


def build_track_shapes(track: Track, kerbs: bool = True):
    """One ``ShapeElementList`` in metres: run-off, kerbs (red/white blocks, or plain run-off
    with ``kerbs=False``), asphalt and the chequered strip."""
    from arcade import shape_list

    shapes = shape_list.ShapeElementList()
    for side in ("left", "right"):
        g = kerb_geometry(track, side)
        if kerbs:
            _append(shapes, g.runoff, theme.RUNOFF)
            colours = [theme.KERB_RED if r else theme.KERB_WHITE for r in g.red]
            _append(shapes, g.kerb, colours)
        else:
            _append(shapes, g.plain, theme.RUNOFF)
    _append(shapes, road_triangles(track), ROAD_COLOUR)
    tris, light = chequer_triangles(track)
    _append(shapes, tris, [CHEQUER_LIGHT if v else CHEQUER_DARK for v in light])
    return shapes


class TrackRenderer:
    """The whole static track of a viewer, uploaded once. Draw it under the world camera."""

    def __init__(self, track: Track, ctx):
        from arcade.gl import BufferDescription

        self.track = track
        self.ctx = ctx
        self.near = build_track_shapes(track, kerbs=True)
        self.far = build_track_shapes(track, kerbs=False) if track.has_limits else self.near
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
