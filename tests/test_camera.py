import numpy as np
import pytest

from f1sim.app.camera import MAX_PIXELS_PER_METRE, MIN_PIXELS_PER_METRE, Camera


@pytest.fixture
def cam() -> Camera:
    c = Camera(1280, 800)
    c.fit(-500.0, -300.0, 900.0, 400.0)
    return c


def test_round_trip_is_exact(cam):
    rng = np.random.default_rng(0)
    x = rng.uniform(-2000, 2000, 1000)
    y = rng.uniform(-2000, 2000, 1000)
    sx, sy = cam.world_to_screen(x, y)
    x2, y2 = cam.screen_to_world(sx, sy)
    np.testing.assert_allclose(x2, x, atol=1e-9)
    np.testing.assert_allclose(y2, y, atol=1e-9)

    px = rng.uniform(0, 1280, (10, 10))
    py = rng.uniform(0, 800, (10, 10))
    wx, wy = cam.screen_to_world(px, py)
    assert wx.shape == (10, 10)
    px2, py2 = cam.world_to_screen(wx, wy)
    np.testing.assert_allclose(px2, px, atol=1e-9)
    np.testing.assert_allclose(py2, py, atol=1e-9)


def test_y_is_up(cam):
    _, sy0 = cam.world_to_screen(0.0, 0.0)
    _, sy1 = cam.world_to_screen(0.0, 10.0)
    assert sy1 > sy0


def test_fit_contains_bbox_and_centres_it():
    cam = Camera(1280, 800)
    box = (-500.0, -300.0, 900.0, 400.0)
    cam.fit(*box, padding_px=20)
    sx, sy = cam.world_to_screen([box[0], box[2]], [box[1], box[3]])
    assert sx[0] >= 20 - 1e-9 and sx[1] <= 1280 - 20 + 1e-9
    assert sy[0] >= 20 - 1e-9 and sy[1] <= 800 - 20 + 1e-9
    # The limiting axis touches the padding (x here: 1400 m wide vs 700 m high).
    assert sx[0] == pytest.approx(20.0) and sx[1] == pytest.approx(1260.0)
    # Centred.
    cx, cy = cam.world_to_screen(200.0, 50.0)
    assert (cx, cy) == (pytest.approx(640.0), pytest.approx(400.0))
    assert cam.pixels_per_metre == pytest.approx(1240.0 / 1400.0)


def test_fit_tall_box_is_limited_by_height():
    cam = Camera(1000, 500)
    cam.fit(0.0, 0.0, 10.0, 1000.0, padding_px=10)
    sx, sy = cam.world_to_screen([0.0, 10.0], [0.0, 1000.0])
    assert sy[0] == pytest.approx(10.0) and sy[1] == pytest.approx(490.0)
    assert 0.0 < sx[0] < sx[1] < 1000.0


def test_zoom_at_keeps_cursor_point_fixed(cam):
    for factor, (px, py) in [(1.15, (100.0, 700.0)), (0.5, (1200.0, 30.0)), (3.0, (640, 400))]:
        before = cam.screen_to_world(px, py)
        ppm = cam.pixels_per_metre
        cam.zoom_at(factor, px, py)
        assert cam.pixels_per_metre == pytest.approx(ppm * factor)
        after = cam.screen_to_world(px, py)
        np.testing.assert_allclose(after, before, atol=1e-9)


def test_zoom_is_clamped_and_still_keeps_cursor_point(cam):
    before = cam.screen_to_world(300.0, 200.0)
    cam.zoom_at(1e9, 300.0, 200.0)
    assert cam.pixels_per_metre == MAX_PIXELS_PER_METRE
    np.testing.assert_allclose(cam.screen_to_world(300.0, 200.0), before, atol=1e-9)
    cam.zoom_at(1e-12, 300.0, 200.0)
    assert cam.pixels_per_metre == MIN_PIXELS_PER_METRE
    np.testing.assert_allclose(cam.screen_to_world(300.0, 200.0), before, atol=1e-6)
    with pytest.raises(ValueError):
        cam.zoom_at(0.0, 0.0, 0.0)


def test_pan_moves_scene_with_cursor(cam):
    sx, sy = cam.world_to_screen(123.0, -45.0)
    cam.pan(30.0, -12.0)
    sx2, sy2 = cam.world_to_screen(123.0, -45.0)
    assert sx2 == pytest.approx(sx + 30.0)
    assert sy2 == pytest.approx(sy - 12.0)


def test_follow_centres_point(cam):
    cam.follow(321.0, -654.0)
    sx, sy = cam.world_to_screen(321.0, -654.0)
    assert (sx, sy) == (pytest.approx(640.0), pytest.approx(400.0))


def test_resize_keeps_centre_and_scale(cam):
    centre = cam.screen_to_world(640.0, 400.0)
    ppm = cam.pixels_per_metre
    cam.resize(1920, 1080)
    assert cam.pixels_per_metre == ppm
    np.testing.assert_allclose(cam.screen_to_world(960.0, 540.0), centre, atol=1e-9)


def test_follow_at_puts_the_point_where_asked(cam):
    cam.follow_at(123.0, -45.0, 640.0, 428.0)  # e.g. the HUD's stage centre
    sx, sy = cam.world_to_screen(123.0, -45.0)
    assert (float(sx), float(sy)) == pytest.approx((640.0, 428.0))
    cam.follow_at(10.0, 20.0, 0.5 * cam.viewport_w, 0.5 * cam.viewport_h)
    assert cam.centre == pytest.approx((10.0, 20.0))  # the window centre: same as follow
