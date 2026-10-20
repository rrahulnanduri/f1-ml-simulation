import numpy as np
import pytest

from _geometry_helpers import circle, stadium
from f1sim.data.track_builder import build_track
from f1sim.sim import sensors
from f1sim.sim.sensors import (
    CURVATURE_SCALE,
    LOOKAHEAD_M,
    OBS_SIZE,
    RAY_ANGLES_DEG,
    RAY_RANGE_M,
    V_NORM,
    Sensors,
)
from f1sim.track.grid import SegmentGrid
from f1sim.track.track import Track

W_LEFT, W_RIGHT = 4.0, 6.0


def make_circle(radius: float, ccw: bool = True) -> Track:
    x, y = circle(radius, n=int(2 * np.pi * radius / 2.0), ccw=ccw)
    m = x.size
    data = build_track(
        x, y, np.full(m, W_RIGHT), np.full(m, W_LEFT), track_id="circle", name="Circle"
    )
    return Track(data)


@pytest.fixture(scope="module")
def ccw_circle() -> Track:
    return make_circle(150.0)


def at(track: Track, s, n, yaw_offset=0.0):
    """Cars at arc length ``s`` and lateral offset ``n``, heading along the track + offset."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    n = np.broadcast_to(np.asarray(n, dtype=np.float64), s.shape)
    h = track.heading_at(s)
    p = track.point_at(s)
    x = p[:, 0] - np.sin(h) * n
    y = p[:, 1] + np.cos(h) * n
    return x, y, h + yaw_offset


def observe(sens: Sensors, x, y, yaw, v=None, action=None):
    s, n, _ = sens.track.project(x, y)
    v = np.zeros_like(x) if v is None else v
    action = np.zeros((x.size, 2)) if action is None else action
    return sens.observe(x, y, yaw, v, s, n, action)


def test_spec_constants():
    assert sensors.OBS_SPEC_VERSION == 1
    assert len(RAY_ANGLES_DEG) == 9 and len(LOOKAHEAD_M) == 5
    assert OBS_SIZE == len(RAY_ANGLES_DEG) + 3 + len(LOOKAHEAD_M) + 2 == 19
    assert (RAY_RANGE_M, V_NORM, CURVATURE_SCALE) == (200.0, 92.0, 50.0)


def test_dtype_and_shape(ccw_circle):
    sens = Sensors(ccw_circle)
    x, y, yaw = at(ccw_circle, [0.0, 100.0, 500.0], 0.0)
    obs = observe(sens, x, y, yaw)
    assert obs.shape == (3, OBS_SIZE) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    empty = sens.observe(*(np.zeros(0),) * 6, np.zeros((0, 2)))
    assert empty.shape == (0, OBS_SIZE) and empty.dtype == np.float32


def test_centre_line_facing_along_track(ccw_circle):
    sens = Sensors(ccw_circle)
    x, y, yaw = at(ccw_circle, [10.0, 300.0, 700.0], 0.0)
    obs = observe(sens, x, y, yaw)
    np.testing.assert_allclose(obs[:, 10], 0.0, atol=1e-6)  # heading error
    np.testing.assert_allclose(obs[:, 11], 0.0, atol=1e-6)  # lateral offset


def test_lateral_offset_feature(ccw_circle):
    sens = Sensors(ccw_circle)
    n = np.array([W_LEFT, 0.5 * W_LEFT, -0.5 * W_RIGHT, -W_RIGHT])
    x, y, yaw = at(ccw_circle, np.full(4, 200.0), n)
    obs = observe(sens, x, y, yaw)
    np.testing.assert_allclose(obs[:, 11], [1.0, 0.5, -0.5, -1.0], atol=1e-5)


def test_side_rays_measure_the_walls(ccw_circle):
    sens = Sensors(ccw_circle)
    x, y, yaw = at(ccw_circle, [50.0, 400.0], 0.0)
    obs = observe(sens, x, y, yaw)
    left = RAY_ANGLES_DEG.index(90)
    right = RAY_ANGLES_DEG.index(-90)
    # Walls are 2 m chords of circles: within 1 cm of the nominal widths.
    np.testing.assert_allclose(obs[:, left] * RAY_RANGE_M, W_LEFT, atol=0.01)
    np.testing.assert_allclose(obs[:, right] * RAY_RANGE_M, W_RIGHT, atol=0.01)
    # Forward ray on a CCW circle of radius 150: chord to the outer wall, then curving away.
    r_out = 150.0 + W_RIGHT
    fwd = obs[:, RAY_ANGLES_DEG.index(0)] * RAY_RANGE_M
    np.testing.assert_allclose(fwd, np.sqrt(r_out**2 - 150.0**2), atol=0.05)
    assert np.all(obs[:, :9] > 0.0) and np.all(obs[:, :9] <= 1.0)


def test_rays_use_the_given_grid_and_match_raycast(ccw_circle):
    grid = SegmentGrid.from_track(ccw_circle, cell_size_m=7.0)
    sens = Sensors(ccw_circle, grid)
    assert sens.grid is grid
    x, y, yaw = at(ccw_circle, [0.0, 123.0], [1.0, -2.0], 0.3)
    obs = observe(sens, x, y, yaw)
    want = grid.raycast(x, y, yaw[:, None] + np.deg2rad(RAY_ANGLES_DEG), RAY_RANGE_M)
    np.testing.assert_allclose(obs[:, :9], want / RAY_RANGE_M, rtol=1e-6)


def test_heading_error_sign_and_wrap(ccw_circle):
    sens = Sensors(ccw_circle)
    offsets = np.array([0.3, -0.5, 2 * np.pi + 0.2, np.pi - 1e-3, -np.pi + 1e-3])
    x, y, yaw = at(ccw_circle, np.full(5, 250.0), 0.0, offsets)
    obs = observe(sens, x, y, yaw)
    want = np.array([0.3, -0.5, 0.2, np.pi - 1e-3, -np.pi + 1e-3]) / np.pi
    np.testing.assert_allclose(obs[:, 10], want, atol=1e-6)


def test_speed_and_previous_action_copied(ccw_circle):
    sens = Sensors(ccw_circle)
    x, y, yaw = at(ccw_circle, [5.0, 6.0], 0.0)
    action = np.array([[0.25, -1.0], [-0.75, 0.5]])
    obs = observe(sens, x, y, yaw, v=np.array([46.0, 92.0]), action=action)
    np.testing.assert_allclose(obs[:, 9], [0.5, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(obs[:, 17:19], action.astype(np.float32))


def test_curvature_lookahead_sign_and_clip():
    ccw = make_circle(150.0, ccw=True)
    cw = make_circle(150.0, ccw=False)
    tight = make_circle(30.0, ccw=True)
    for track, want in ((ccw, 50.0 / 150.0), (cw, -50.0 / 150.0), (tight, 1.0)):
        sens = Sensors(track)
        x, y, yaw = at(track, [0.0, 0.4 * track.length], 0.0)
        obs = observe(sens, x, y, yaw)
        np.testing.assert_allclose(obs[:, 12:17], want, atol=2e-3)


def test_curvature_looks_ahead_along_s():
    # Stadium: straights (curvature 0) joined by CCW semicircles of radius 40.
    x, y = stadium(300.0, 40.0, 2.0)
    m = x.size
    track = Track(build_track(x, y, np.full(m, 6.0), np.full(m, 6.0), track_id="st", name="St"))
    sens = Sensors(track)
    s = np.array([0.0, 77.0, 1000.0])
    s_out, n_out, _ = track.project(*track.point_at(s).T)
    yaw = track.heading_at(s_out)
    pts = track.point_at(s_out)
    obs = sens.observe(pts[:, 0], pts[:, 1], yaw, np.zeros(3), s_out, n_out, np.zeros((3, 2)))
    want = np.clip(
        track.curvature_at(s_out[:, None] + np.array(LOOKAHEAD_M)) * CURVATURE_SCALE, -1, 1
    )
    np.testing.assert_allclose(obs[:, 12:17], want, atol=1e-6)
    assert np.ptp(obs[:, 12:17]) > 0.5  # mixes straight and bend values


def test_bad_shapes_rejected(ccw_circle):
    sens = Sensors(ccw_circle)
    z = np.zeros(3)
    with pytest.raises(ValueError):
        sens.observe(z, z, z, z, z, np.zeros(2), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        sens.observe(z, z, z, z, z, z, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        sens.observe(z, z, z, z, z, z, np.zeros(6))
    with pytest.raises(ValueError):
        sens.observe(np.zeros((3, 1)), z, z, z, z, z, np.zeros((3, 2)))
