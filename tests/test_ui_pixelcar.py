"""Pixel-art F1 car generation: pure numpy, no window."""

import math

import numpy as np
import pytest

from f1sim.app.ui import pixelcar as pc
from f1sim.app.ui import theme


def _rgba(colour):
    return tuple(int(c) for c in colour)


def test_art_is_a_16_by_40_symmetric_2026_silhouette():
    assert (pc.ART_W, pc.ART_H) == (16, 40)
    assert all(len(row) == pc.ART_W for row in pc.CAR_ART)
    for row in pc.CAR_ART:
        assert row == row[::-1], row  # mirror-symmetric about the car's axis
    assert set("".join(pc.CAR_ART)) <= set(".TtYDFKHPhpWsSEAL")
    art = np.array([list(r) for r in pc.CAR_ART])
    solid = art != "."
    # Full-width front wing at the nose, a narrower rear wing at the back.
    assert solid[0].all()
    assert 0 < solid[-3].sum() < pc.ART_W and not solid[-3, 0]
    # Tyres in the front and back thirds: the rear ones wider, reaching the car's edge, the
    # front ones inset; Pirelli-yellow sidewalls on both.
    tyre = np.isin(art, ["T", "t", "Y"])
    front = np.flatnonzero(tyre[: pc.ART_H // 3].any(axis=0))
    rear = np.flatnonzero(tyre[2 * pc.ART_H // 3 :].any(axis=0))
    assert len(rear) > len(front) and 0 in rear and 0 not in front
    assert (art[: pc.ART_H // 3] == "Y").any() and (art[2 * pc.ART_H // 3 :] == "Y").any()
    # A small dark cockpit on the centre line in the front half, ringed by the halo.
    k_rows, k_cols = np.nonzero(art == "K")
    assert k_rows.max() < pc.ART_H // 2 + 2
    cockpit = k_rows < 19
    assert set(k_cols[cockpit]) == {7, 8}
    assert ((art == "H").any(axis=1)).sum() >= 4
    # A red rain light at the very back.
    assert (art[-1] == "L").any()


def test_pixels_take_the_livery():
    lv = theme.LIVERIES["mclaren"]
    px = pc.car_pixels(lv)
    assert px.shape == (40, 16, 4) and px.dtype == np.uint8
    alpha = px[..., 3]
    assert set(np.unique(alpha)) == {0, 255}  # crisp pixel art: no partial alpha
    np.testing.assert_array_equal(px, px[:, ::-1])  # symmetric colours too
    colours = {tuple(c) for c in px.reshape(-1, 4) if c[3]}
    assert _rgba(lv.primary) in colours
    assert _rgba(lv.secondary) in colours
    assert _rgba(lv.accent) in colours
    assert pc.TYRE in colours and pc.COCKPIT in colours and pc.SIDEWALL in colours
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
    assert px.shape == (42, 18, 4)
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
    assert px[1 + 5, 1 + 5, 3] == 0


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


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0])
def test_fixed_scale_ignores_the_zoom(ratio):
    k = pc.fixed_scale(40.0, ratio)
    assert k == max(round(40.0 * ratio / pc.ART_H), 1)
    assert abs(pc.ART_H * k / ratio - 40.0) <= 0.5 * pc.ART_H / ratio + 1e-9
    assert pc.fixed_scale(1.0, ratio) == 1


def test_field_liveries_are_distinct():
    lvs = theme.field_liveries(64)
    assert len(lvs) == 64
    prim = np.array([lv.primary[:3] for lv in lvs], dtype=float)
    assert len({tuple(p) for p in prim.astype(int)}) == 64
    d = np.linalg.norm(prim[:, None] - prim[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 12  # no two cars the same colour
    # Neighbours in the field (consecutive cars) are far apart in colour.
    assert min(np.linalg.norm(prim[i] - prim[i + 1]) for i in range(63)) > 80
    assert all(lv.primary[3] == 255 for lv in lvs)


def test_label_offset_clears_the_rotated_car():
    assert pc.label_offset(0.0, 30, 12) == pytest.approx(15)
    assert pc.label_offset(math.pi / 2, 30, 12) == pytest.approx(6)
    assert pc.label_offset(math.pi / 4, 30, 12) == pytest.approx(21 / math.sqrt(2))


def test_shade():
    assert pc.shade((100, 100, 100, 255), 1.0) == (255, 255, 255, 255)
    assert pc.shade((100, 100, 100, 255), -1.0) == (0, 0, 0, 255)
    assert pc.shade((100, 50, 0, 255), 0.0) == (100, 50, 0, 255)
