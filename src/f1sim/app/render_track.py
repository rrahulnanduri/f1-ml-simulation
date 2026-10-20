"""Static track geometry for drawing: road triangles, start/finish and sector lines.

:func:`road_triangles` and :func:`boundary_lines` are pure numpy. :func:`build_track_shapes`
turns them into one arcade ``ShapeElementList`` **in world metres**.

How the static shapes are cached and transformed (arcade 3.3.3): a ``ShapeElementList``
uploads its vertices to one GPU buffer per primitive mode the first time it is drawn, and later
draws are a single ``render`` call per mode with no Python-side geometry. Its vertex shader
multiplies by the window's current projection and view matrices, so the viewer sets those with
an ``arcade.camera.Camera2D`` (``position`` = the :class:`~f1sim.app.camera.Camera` centre,
``zoom`` = pixels per metre) before drawing. Zooming and panning only change two uniforms; the
geometry is never rebuilt.

Lines are drawn in two ways so they read at every zoom: 1-pixel GL lines (walls, and the
start/sector lines) that stay visible when zoomed out, and, for the start and sector lines, a
quad of fixed world width that gives them body when zoomed in.

The batches are drawn in the order their primitive mode first appears: road triangles, wall
line strips, start/sector quads, then the 1-pixel start/sector lines on top.
"""

from __future__ import annotations

import numpy as np

from f1sim.track.track import Track

ROAD_COLOUR = (60, 60, 60, 255)
WALL_COLOUR = (255, 255, 255, 255)
START_COLOUR = (220, 30, 30, 255)
SECTOR_COLOUR = (40, 110, 255, 255)
LINE_BODY_WIDTH_M = 1.5
"""World width of the start/finish and sector line quads."""


def road_triangles(track: Track) -> np.ndarray:
    """Triangles filling the road between the walls, shape ``(2M, 3, 2)``.

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


def _points(a: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in np.asarray(a).reshape(-1, 2)]


def _line_quad(line: np.ndarray, width_m: float) -> np.ndarray:
    """Four triangle-strip corners of a quad of ``width_m`` centred on ``line`` (2, 2)."""
    a, b = line
    d = b - a
    length = float(np.hypot(*d))
    if length == 0.0:
        return np.array([a, a, b, b])
    off = np.array([-d[1], d[0]]) / length * (0.5 * width_m)  # perpendicular to the line
    return np.array([a + off, a - off, b + off, b - off])


def build_track_shapes(track: Track):
    """One ``arcade.shape_list.ShapeElementList`` with the whole static track, in metres.

    Build it once (needs an open window for the GL context) and draw it every frame under a
    ``Camera2D`` matching :class:`~f1sim.app.camera.Camera`.
    """
    from arcade import shape_list

    shapes = shape_list.ShapeElementList()

    tris = road_triangles(track)
    road_pts = _points(tris)
    shapes.append(
        shape_list.create_triangles_filled_with_colors(road_pts, [ROAD_COLOUR] * len(road_pts))
    )

    for wall in (track.left_wall(), track.right_wall()):
        shapes.append(shape_list.create_line_loop(_points(wall), WALL_COLOUR))

    lines = boundary_lines(track)
    coloured = [(lines["start_finish"], START_COLOUR)]
    coloured += [(line, SECTOR_COLOUR) for line in lines["sectors"]]
    for line, colour in coloured:
        quad = _points(_line_quad(line, LINE_BODY_WIDTH_M))
        shapes.append(shape_list.create_triangles_strip_filled_with_colors(quad, [colour] * 4))
    for line, colour in coloured:
        shapes.append(shape_list.create_lines(_points(line), colour))
    return shapes
