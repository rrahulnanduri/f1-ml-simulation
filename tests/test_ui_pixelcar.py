"""Pixel-art F1 car generation: pure numpy, no window."""

import math

import numpy as np
import pytest

from f1sim.app.ui import pixelcar as pc
from f1sim.app.ui import theme


def _rgba(colour):
    return tuple(int(c) for c in colour)


def test_art_is_a_12_by_30_symmetric_f1_silhouette():
    assert (pc.ART_W, pc.ART_H) == (12, 30)
    assert all(len(row) == pc.ART_W for row in pc.CAR_ART)
    for row in pc.CAR_ART:
        assert row == row[::-1], row  # mirror-symmetric about the car's axis
    assert set("".join(pc.CAR_ART)) <= set(".TtDKHPhpWsSEA")
    art = np.array([list(r) for r in pc.CAR_ART])
    solid = art != "."
    # Full-width front wing at the nose, a narrower rear wing at the back.
    assert solid[0].all()
    assert solid[-1].sum() == 10
    # Tyres: the rear ones are wider than the front ones and stick out of the body.
    front = [c for c in range(12) if (art[3:8, c] == "T").any()]
    rear = [c for c in range(12) if (art[21:27, c] == "T").any()]
    assert len(rear) > len(front) and 0 in rear and 0 not in front
    # A dark cockpit with a halo on the centre line, ahead of the middle of the car.
    k_rows = np.flatnonzero((art == "K").any(axis=1))
    assert 8 < k_rows.min() and k_rows.max() < 15
    assert ((art == "H").any(axis=1)).sum() >= 3


def test_pixels_take_the_livery():
    lv = theme.LIVERIES["mclaren"]
    px = pc.car_pixels(lv)
    assert px.shape == (30, 12, 4) and px.dtype == np.uint8
    alpha = px[..., 3]
    assert set(np.unique(alpha)) == {0, 255}  # crisp pixel art: no partial alpha
    np.testing.assert_array_equal(px, px[:, ::-1])  # symmetric colours too
    colours = {tuple(c) for c in px.reshape(-1, 4) if c[3]}
    assert _rgba(lv.primary) in colours
    assert _rgba(lv.secondary) in colours
    assert _rgba(lv.accent) in colours
    assert pc.TYRE in colours and pc.COCKPIT in colours
    # One-pixel highlight and shadow lines: lighter and darker versions of the primary.
    assert pc.shade(lv.primary, 0.32) in colours
    assert pc.shade(lv.primary, -0.30) in colours
    assert sum(pc.shade(lv.primary, 0.32)[:3]) > sum(_rgba(lv.primary)[:3])


def test_self_car_is_titanium_black_and_red():
    px = pc.car_pixels(theme.SELF_LIVERY)
    colours = {tuple(c) for c in px.reshape(-1, 4) if c[3]}
    assert _rgba(theme.hex_colour("#A9ADB2")) in colours
    assert _rgba(theme.ACCENT) in colours


def test_ghost_outline_is_one_light_pixel_round_the_envelope():
    lv = theme.LIVERIES["red bull"]
    plain = pc.car_pixels(lv)
    px = pc.car_pixels(lv, theme.CAR_OUTLINE)
    assert px.shape == (32, 14, 4)
    np.testing.assert_array_equal(px[1:-1, 1:-1][plain[..., 3] > 0], plain[plain[..., 3] > 0])
    ring = np.all(px == np.array(theme.CAR_OUTLINE, dtype=np.uint8), axis=-1)
    assert ring.any() and not (ring & (np.pad(plain[..., 3], 1) > 0)).any()
    # The outline is exactly the 4-neighbour ring round the car's envelope (each row filled
    # from its first to its last solid pixel).
    car = np.pad(plain[..., 3] > 0, 1)
    env = np.zeros_like(car)
    for r in range(car.shape[0]):
        c = np.flatnonzero(car[r])
        if c.size:
            env[r, c[0] : c[-1] + 1] = True
    grown = env.copy()
    grown[1:] |= env[:-1]
    grown[:-1] |= env[1:]
    grown[:, 1:] |= env[:, :-1]
    grown[:, :-1] |= env[:, 1:]
    np.testing.assert_array_equal(ring, grown & ~env)
    np.testing.assert_array_equal(px, px[:, ::-1])  # still symmetric
    # The gaps between tyres and body stay see-through.
    assert px[1 + 4, 1 + 3, 3] == 0


@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_sprite_scale_is_a_whole_multiple_with_a_minimum(ratio):
    # Zoomed out: the minimum on-screen length, in whole physical pixels per art pixel.
    k = pc.sprite_scale(0.1, ratio)
    assert k == math.ceil(theme.CAR_MIN_LENGTH_PX * ratio / pc.ART_H)
    assert pc.ART_H * k / ratio >= theme.CAR_MIN_LENGTH_PX
    # Zoomed in: the real 5.6 m, rounded to a whole multiple.
    for ppm in (10.0, 20.0, 37.0):
        k = pc.sprite_scale(ppm, ratio)
        assert isinstance(k, int)
        world = theme.CAR_LENGTH_M * ppm * ratio
        assert abs(pc.ART_H * k - world) <= 0.5 * pc.ART_H + 1e-9
    assert pc.sprite_scale(20.0, ratio) > pc.sprite_scale(10.0, ratio)


def test_label_offset_clears_the_rotated_car():
    assert pc.label_offset(0.0, 30, 12) == pytest.approx(15)
    assert pc.label_offset(math.pi / 2, 30, 12) == pytest.approx(6)
    assert pc.label_offset(math.pi / 4, 30, 12) == pytest.approx(21 / math.sqrt(2))


def test_shade():
    assert pc.shade((100, 100, 100, 255), 1.0) == (255, 255, 255, 255)
    assert pc.shade((100, 100, 100, 255), -1.0) == (0, 0, 0, 255)
    assert pc.shade((100, 50, 0, 255), 0.0) == (100, 50, 0, 255)
