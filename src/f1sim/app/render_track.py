"""Static track geometry for drawing: road triangles and wall hairlines, plus the start/finish
and sector lines as data.

:func:`road_triangles` and :func:`boundary_lines` are pure numpy. :func:`build_track_shapes`
turns the road and walls into one arcade ``ShapeElementList`` **in world metres**, in the
"pit wall noir" look (:mod:`f1sim.app.ui.theme`): a ``#1A1C21`` road band between hairline
walls (white at 30 %).

How the static shapes are cached and transformed (arcade 3.3.3): a ``ShapeElementList``
uploads its vertices to one GPU buffer per primitive mode the first time it is drawn, and later
draws are a single ``render`` call per mode with no Python-side geometry. Its vertex shader
multiplies by the window's current projection and view matrices, so the viewer sets those with
an ``arcade.camera.Camera2D`` (``position`` = the :class:`~f1sim.app.camera.Camera` centre,
``zoom`` = pixels per metre) before drawing. Zooming and panning only change two uniforms; the
geometry is never rebuilt.

The walls are 1-pixel GL line loops, so they stay one screen pixel wide at any zoom. The
start/finish line is not part of the cached shapes: the viewers draw it in screen space as a
solid 2-3 px white line (:func:`f1sim.app.ui.glass.draw_start_line`). Sector lines are no
longer drawn; :func:`boundary_lines` still describes them.
"""

from __future__ import annotations

import numpy as np

from f1sim.track.track import Track

ROAD_COLOUR = (26, 28, 33, 255)  # theme.ROAD
WALL_COLOUR = (255, 255, 255, 77)  # theme.WALL: white at 30 %


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

    return shapes
