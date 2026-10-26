import importlib
import sys
from dataclasses import replace

import numpy as np
import pytest
import shapely

from _geometry_helpers import circle, stadium
from f1sim.app.render_track import boundary_lines, road_triangles
from f1sim.data.track_builder import build_track
from f1sim.track.track import Track


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _road_area(track: Track) -> float:
    """Area between the two wall polygons (one wall is inside the other)."""
    return abs(abs(_shoelace(track.left_wall())) - abs(_shoelace(track.right_wall())))


def _triangle_area(tris: np.ndarray) -> np.ndarray:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


@pytest.fixture(scope="module")
def circle_track() -> Track:
    x, y = circle(150.0, 300)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 5.0), track_id="circle", name="Circle")
    return Track(data)


@pytest.fixture(scope="module")
def stadium_track() -> Track:
    x, y = stadium(300.0, 60.0, 3.0)
    n = x.size
    w_right = 6.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, n, endpoint=False))
    data = build_track(x, y, w_right, np.full(n, 5.0), track_id="stadium", name="Stadium")
    track = Track(data)
    return Track(replace(track.data, sectors_s=(0.0, 0.3 * track.length, 0.71 * track.length)))


# ------------------------------------------------------------------------ road_triangles


@pytest.mark.parametrize("name", ["circle_track", "stadium_track"])
def test_road_triangles_cover_road_area(name, request):
    track = request.getfixturevalue(name)
    tris = road_triangles(track)
    m = track.data.n_points
    assert tris.shape == (2 * m, 3, 2)
    area = _triangle_area(tris).sum()
    assert area == pytest.approx(_road_area(track), rel=0.01)


def test_road_triangles_circle_matches_analytic_annulus(circle_track):
    area = _triangle_area(road_triangles(circle_track)).sum()
    r = 150.0
    assert area == pytest.approx(np.pi * ((r + 6.0) ** 2 - (r - 5.0) ** 2), rel=0.01)


def test_road_triangles_use_wall_vertices_and_close_the_loop(stadium_track):
    tris = road_triangles(stadium_track)
    left, right = stadium_track.left_wall(), stadium_track.right_wall()
    m = left.shape[0]
    np.testing.assert_array_equal(tris[0], [left[0], right[0], left[1]])
    np.testing.assert_array_equal(tris[1], [right[0], right[1], left[1]])
    # Closing pair: wall vertex M-1 back to 0.
    np.testing.assert_array_equal(tris[-2], [left[m - 1], right[m - 1], left[0]])
    np.testing.assert_array_equal(tris[-1], [right[m - 1], right[0], left[0]])


def test_road_triangles_stay_inside_road(stadium_track):
    tris = road_triangles(stadium_track)
    outer = shapely.Polygon(stadium_track.right_wall())  # CCW loop: right wall is outside
    inner = shapely.Polygon(stadium_track.left_wall())
    road = outer.difference(inner).buffer(1e-6)
    centroids = tris.mean(axis=1)
    assert shapely.contains_xy(road, centroids[:, 0], centroids[:, 1]).all()


# ------------------------------------------------------------------------ boundary_lines


def _check_line(track: Track, line: np.ndarray, s: float) -> None:
    right_pt, left_pt = line
    w_left, w_right = (float(v) for v in track.widths_at(s))
    # Endpoints on the walls: projecting gives n = -w_right / +w_left at the same s ...
    ps, pn, _ = track.project(line[:, 0], line[:, 1])
    np.testing.assert_allclose(pn, [-w_right, w_left], atol=0.01)
    ds = (ps - s + track.length / 2) % track.length - track.length / 2
    np.testing.assert_allclose(ds, 0.0, atol=0.01)
    # ... and lie on the wall polylines.
    rw = shapely.LinearRing(track.right_wall())
    lw = shapely.LinearRing(track.left_wall())
    assert rw.distance(shapely.Point(right_pt)) < 0.01
    assert lw.distance(shapely.Point(left_pt)) < 0.01
    # Spans the road width and is perpendicular to the track.
    d = left_pt - right_pt
    assert np.hypot(*d) == pytest.approx(w_left + w_right, abs=1e-6)
    h = float(track.heading_at(s))
    tangent = np.array([np.cos(h), np.sin(h)])
    assert abs(np.dot(d / np.hypot(*d), tangent)) < 1e-9
    # Right wall first: the line points to the left of the travel direction.
    assert tangent[0] * d[1] - tangent[1] * d[0] > 0.0


def test_boundary_lines_start_finish(stadium_track):
    lines = boundary_lines(stadium_track)
    assert lines["start_finish"].shape == (2, 2)
    _check_line(stadium_track, lines["start_finish"], 0.0)
    # At s = 0 the endpoints are exactly the first wall vertices.
    np.testing.assert_allclose(lines["start_finish"][0], stadium_track.right_wall()[0])
    np.testing.assert_allclose(lines["start_finish"][1], stadium_track.left_wall()[0])


def test_boundary_lines_sectors(stadium_track):
    lines = boundary_lines(stadium_track)
    sectors_s = stadium_track.data.sectors_s[1:]
    assert lines["sectors"].shape == (len(sectors_s), 2, 2)
    for line, s in zip(lines["sectors"], sectors_s, strict=True):
        _check_line(stadium_track, line, s)


def test_boundary_lines_without_sectors(circle_track):
    assert circle_track.data.sectors_s == ()
    lines = boundary_lines(circle_track)
    assert lines["sectors"].shape == (0, 2, 2)
    _check_line(circle_track, lines["start_finish"], 0.0)


# ----------------------------------------------------------------- drive (no window)


def _teleport(session, s: float, n: float) -> None:
    """Put the car at (s, n) and point the projection hint there (a teleport is no drive)."""
    track = session.track
    p = track.point_at(s)
    h = float(track.heading_at(s))
    session.state.x[:] = p[0] - np.sin(h) * n
    session.state.y[:] = p[1] + np.cos(h) * n
    session.idx = int(track.project(session.state.x, session.state.y)[2][0])


def test_importing_drive_opens_no_window():
    sys.modules.pop("f1sim.app.drive", None)
    drive = importlib.import_module("f1sim.app.drive")
    from arcade import window_commands

    assert window_commands._window is None
    assert callable(drive.run)


def test_drive_session_ramps_inputs_and_accelerates(stadium_track):
    from f1sim.app.drive import DT, PEDAL_RAMP_S, Controls, DriveSession

    session = DriveSession(stadium_track)
    controls = Controls(throttle=True)
    session.fixed_step(controls)
    assert 0.0 < session.throttle < 1.0
    for _ in range(int(np.ceil(PEDAL_RAMP_S / DT))):
        session.fixed_step(controls)
    assert session.throttle == pytest.approx(1.0)
    assert session.speed_kmh > 0.0
    # The accumulator runs whole fixed steps only.
    assert session.update(controls, 2.5 * DT) == 2
    assert session.update(controls, 0.5 * DT) == 1


def test_drive_session_lap_timing_is_interpolated(circle_track):
    from f1sim.app.drive import DriveSession

    session = DriveSession(circle_track)
    session.clock_running = True
    length = circle_track.length
    v, dt = 50.0, 0.1  # teleport along the centre line at a constant speed
    for k in range(1, int(2.5 * length / (v * dt))):
        p = circle_track.point_at(k * v * dt)
        session.state.x[:] = p[0]
        session.state.y[:] = p[1]
        session.update_progress(dt)
        assert not session.off_track
    assert session.laps == 2
    assert session.last_lap == pytest.approx(length / v, abs=1e-6)
    assert session.best_lap == pytest.approx(length / v, abs=1e-6)


def test_drive_session_zigzag_over_line_is_no_lap(circle_track):
    from f1sim.app.drive import DriveSession

    session = DriveSession(circle_track)
    session.clock_running = True
    for s in [10.0, -10.0, 20.0, -20.0, 30.0, -5.0, 15.0] * 5:
        p = circle_track.point_at(s)
        session.state.x[:] = p[0]
        session.state.y[:] = p[1]
        session.update_progress(0.1)
    assert session.laps == 0
    assert session.last_lap is None


def test_drive_session_off_track_resets_after_one_second(circle_track):
    from f1sim.app.drive import DT, OFF_TRACK_RESET_S, Controls, DriveSession

    session = DriveSession(circle_track)
    start = (session.x, session.y)
    w_left = float(circle_track.widths_at(100.0)[0])
    _teleport(session, 100.0, w_left - 0.9)  # inside the walls, past the half-width margin
    session.state.v[:] = 40.0
    session.update_progress(DT)
    assert session.off_track
    assert session.speed_kmh == 0.0
    frames = round(OFF_TRACK_RESET_S / DT)
    for _ in range(frames - 1):
        session.fixed_step(Controls(throttle=True))
    assert session.off_track
    session.fixed_step(Controls(throttle=True))
    assert not session.off_track
    assert (session.x, session.y) == pytest.approx(start)


def test_drive_session_off_track_uses_track_limits(circle_track):
    from f1sim.app.drive import DriveSession

    data = replace(
        circle_track.data,
        limit_left=circle_track.data.w_left + 3.0,
        limit_right=circle_track.data.w_right + 3.0,
    )
    track = Track(data)
    session = DriveSession(track)
    w_left = float(track.widths_at(100.0)[0])
    _teleport(session, 100.0, w_left + 1.0)  # over the white line, inside the limits
    session.update_progress(1 / 60)
    assert not session.off_track
    _teleport(session, 100.0, w_left + 2.5)  # within the half width of the limit
    session.update_progress(1 / 60)
    assert session.off_track


def test_drive_session_stays_on_track_with_margin(circle_track):
    from f1sim.app.drive import DriveSession

    session = DriveSession(circle_track)
    n = -(float(circle_track.widths_at(100.0)[1]) - 1.1)  # right side, inside the margin
    _teleport(session, 100.0, n)
    session.update_progress(1 / 60)
    assert not session.off_track
    assert session.n == pytest.approx(n, abs=1e-3)


def test_car_polygon_is_rotated_rectangle():
    from f1sim.app.drive import car_polygon

    corners = car_polygon(10.0, -5.0, np.pi / 2, 5.6, 2.0)
    # Facing +y: front corners have y = -5 + 2.8, the car is 2.0 m wide in x.
    np.testing.assert_allclose(corners[1], [11.0, -2.2], atol=1e-12)
    np.testing.assert_allclose(corners[2], [9.0, -2.2], atol=1e-12)
    assert abs(_shoelace(corners)) == pytest.approx(5.6 * 2.0)
    assert _shoelace(corners) > 0  # counter-clockwise
