"""Main menu logic (f1sim.app.menu) and the top-down car maps: offline, no window."""

import json
import os
import sys
import time

import numpy as np
import pytest

from f1sim.app import menu
from f1sim.app.ui import car_top, hero_maps, light_stage

# -------------------------------------------------------------------------- discovery


def _run(root, name, track, mtime):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.toml").write_text(f'track_id = "{track}"\n', encoding="utf-8")
    lap = d / "best_lap.npz"
    lap.write_bytes(b"x")
    os.utime(lap, (mtime, mtime))
    return d


def test_latest_run_is_the_newest_of_the_track(tmp_path):
    now = time.time()
    _run(tmp_path, "old", "silverstone", now - 100)
    new = _run(tmp_path, "new", "silverstone", now - 10)
    _run(tmp_path, "other", "monza", now)
    (tmp_path / "no-lap").mkdir()
    assert menu.latest_run("silverstone", tmp_path) == new
    assert menu.latest_run("spa", tmp_path) is None
    assert menu.latest_run("silverstone", tmp_path / "missing") is None


def test_latest_ghost_matches_the_track_exactly(tmp_path):
    now = time.time()
    for i, name in enumerate(
        [
            "2024_las_vegas_Q_VER",
            "2025_las_vegas_Q_LEC",
            "2025_las_vegas_2_Q_X",
            "notes_las_vegas_x",
        ]
    ):
        p = tmp_path / f"{name}.npz"
        p.write_bytes(b"x")
        os.utime(p, (now + i, now + i))
    assert menu.latest_ghost("las_vegas", tmp_path).stem == "2025_las_vegas_Q_LEC"
    assert menu.latest_ghost("monza", tmp_path) is None


def test_commands(tmp_path):
    run, ghost = tmp_path / "run", tmp_path / "g.npz"
    base = [sys.executable, "-m", "f1sim.cli"]
    assert menu.command("Drive", "spa", run=None, ghost=None) == [*base, "drive", "spa"]
    assert menu.command("Drive", "spa", run=None, ghost=ghost)[-2:] == ["--ghost", str(ghost)]
    assert menu.command("Replay", "spa", run=None, ghost=ghost) is None
    assert menu.command("Replay", "spa", run=run, ghost=ghost) == [
        *base,
        "replay",
        "spa",
        str(run),
        str(ghost),
    ]
    assert menu.command("Train", "spa", run=None, ghost=None)[-3:] == ["train", "--track", "spa"]
    assert menu.command("Quit", "spa", run=run, ghost=ghost) is None


# ------------------------------------------------------------------------------ hero


SIZE_M = (2 * car_top.HALF_WIDTH_M, car_top.FRONT_LENGTH_M)


@pytest.mark.parametrize(("w", "h"), [(1280, 760), (1920, 1080), (2400, 900), (1000, 1000)])
def test_hero_sits_in_the_right_third_nose_up(w, h):
    lay = menu.hero_layout(w, h, SIZE_M)
    left, bottom, rw, rh = lay.rect
    assert left + 0.5 * rw == pytest.approx(w * 5 / 6)  # centred on the right third
    assert lay.top == pytest.approx(h * (1 - menu.HERO_TOP_MARGIN))  # nose near the top
    car_w = 2 * car_top.HALF_WIDTH_M * lay.px_per_m
    assert car_w <= w / 3 + 1e-6  # no wider than a third
    visible_m = lay.top / lay.px_per_m  # nose to the window's bottom edge
    assert visible_m >= menu.HERO_VISIBLE_M - 1e-6
    if car_w < w / 3 - 1e-6:  # height-limited: exactly the nose to just behind the driver
        assert visible_m == pytest.approx(menu.HERO_VISIBLE_M)


def test_hero_window_and_car_coordinates_round_trip():
    lay = menu.hero_layout(1280, 760, SIZE_M)
    x, y = lay.to_window(0.0, 0.0)  # front wing leading edge, on the centre line
    assert x == pytest.approx(1280 * 5 / 6) and y == pytest.approx(lay.top)
    for cx, cy in [(0.3, 1.2), (-0.8, 3.0), (2.0, -1.0)]:
        assert lay.to_car(*lay.to_window(cx, cy)) == pytest.approx((cx, cy))
    _, y_back = lay.to_window(0.0, 1.0)
    assert y_back < y  # further back along the car is lower on screen


def test_light_eases_and_fades():
    light = menu.Light()
    light.update(1.0, ready=False)
    assert light.power == 0.0  # held off during the intro
    light.aim(1.0, 2.0)
    for _ in range(60):
        light.update(1 / 60)
    assert light.power > 0.9 and (light.x, light.y) == pytest.approx((1.0, 2.0), abs=1e-3)
    light.off()
    for _ in range(120):
        light.update(1 / 60)
    assert light.power < 0.01
    # Frame-rate independent: one 0.5 s step lands where 30 small ones do
    a, b = menu.Light(), menu.Light()
    for li in (a, b):
        li.aim(0.0, 0.0)
    a.update(0.5)
    for _ in range(30):
        b.update(0.5 / 30)
    assert (a.x, a.y, a.power) == pytest.approx((b.x, b.y, b.power))


# ------------------------------------------------------------------------- car maps


@pytest.fixture(scope="module")
def maps():
    return car_top.build_maps(160.0)


def test_maps_shapes_and_ranges(maps):
    h, w = maps.height.shape
    assert maps.albedo.shape == (h, w, 4) and maps.normal.shape == (h, w, 3)
    assert maps.material.shape == (h, w, 2)
    assert maps.size_m == pytest.approx(SIZE_M, abs=2 / maps.px_per_m)
    a = maps.albedo[..., 3]
    assert a.min() == 0.0 and a.max() == 1.0
    assert np.allclose(np.linalg.norm(maps.normal, axis=-1), 1.0, atol=1e-4)
    assert maps.normal[..., 2].min() > 0.0  # every normal faces up
    assert 0.6 < maps.height.max() < 1.1  # the airbox / halo, about a metre up


def test_car_is_symmetric_and_fades_at_the_back(maps):
    a = maps.albedo[..., 3]
    assert np.abs(a - a[:, ::-1]).mean() < 0.01  # mirror-symmetric about the centre line
    rows = a.shape[0]
    fade_rows = int(car_top.FADE_M * maps.px_per_m)
    assert a[-1].max() < 0.05 and a[rows - 2 * fade_rows].max() == 1.0


def test_front_tyres_are_where_and_how_high_they_should_be(maps):
    ppm = maps.px_per_m
    row = int(car_top.FRONT_AXLE_Y * ppm)
    col = int((car_top.HALF_WIDTH_M + car_top.WHEEL_X) * ppm)
    assert maps.height[row, col] == pytest.approx(2 * car_top.TYRE_RADIUS, abs=0.02)
    assert maps.albedo[row, col, :3] == pytest.approx(car_top.TYRE, abs=0.02)


def test_shading_is_brightest_under_the_light(maps):
    ppm = maps.px_per_m
    near = car_top.shade(maps, (0.0, 2.0))
    far = car_top.shade(maps, (0.0, 0.2))
    r0, r1 = int(1.9 * ppm), int(2.1 * ppm)  # the cockpit region
    c0 = int((car_top.HALF_WIDTH_M - 0.5) * ppm)
    c1 = int((car_top.HALF_WIDTH_M + 0.5) * ppm)
    assert near[r0:r1, c0:c1].mean() > 3 * far[r0:r1, c0:c1].mean()
    assert near.max() <= 1.0 and near.min() >= 0.0


def test_texture_bytes_are_rgba8_and_flipped(maps):
    hm = hero_maps.from_drawn(maps)
    data = menu._texture_bytes(hm)
    assert set(data) == {"albedo", "normal", "surface", "coat"}
    h, w = maps.height.shape
    for b in data.values():
        assert len(b) == h * w * 4
    first_row_alpha = np.frombuffer(data["albedo"], np.uint8).reshape(h, w, 4)[0, :, 3]
    assert first_row_alpha.max() < 20  # GL's first row is the faded back of the car


# ------------------------------------------------------------------- rendered maps


def _write_renders(d, w=40, h=200, ppm=20.0, margin=0.1):
    """A fake render folder: a car-shaped block, its passes and meta.json."""
    from PIL import Image

    d.mkdir()
    alpha = np.zeros((h, w), np.uint16)
    alpha[5:, 8:32] = 65535  # the car, nose 5 rows (0.25 m) below the top

    def save(name, rgb):
        a = np.dstack([rgb, alpha]).astype(np.uint16)
        # Pillow can't write 16-bit RGBA: store 8-bit, which load_rendered also reads
        Image.fromarray((a >> 8).astype(np.uint8)).save(d / f"{name}.png")

    full = np.full((h, w), 65535, np.uint16)
    half = np.full((h, w), 32768, np.uint16)
    zero = np.zeros((h, w), np.uint16)
    save("albedo", np.dstack([full, zero, zero]))  # red
    save("normal", np.dstack([half, half, full]))  # facing up
    save("surface", np.dstack([half, zero, half]))  # roughness 0.5, height 0.75 m
    save("ao", np.dstack([full, full, zero]))  # no occlusion, full clear coat
    meta = {"px_per_m": ppm, "margin_m": margin, "height_scale_m": 1.5}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_rendered_crops_fades_and_caches(tmp_path):
    d = tmp_path / "render"
    _write_renders(d)
    m = hero_maps.load_rendered(d)
    assert m is not None and m.source == "render"
    rows = round((0.1 + hero_maps.FRONT_LENGTH_M) * 20.0)
    assert m.albedo.shape == (rows, 40, 4)  # cropped behind the driver
    assert m.size_m == pytest.approx((2.0, rows / 20.0))
    assert m.albedo[rows // 3, 20].tolist() == [255, 0, 0, 255]
    assert m.albedo[-1, 20, 3] < 20  # faded at the cut
    assert m.albedo[0, 20, 3] == 0  # the margin above the nose is empty
    assert m.surface[rows // 3, 20, 3] == 255 and m.coat[rows // 3, 20] == 255
    assert (d / hero_maps.CACHE_NAME).is_file()
    again = hero_maps.load_rendered(d)  # from the cache
    assert np.array_equal(again.albedo, m.albedo) and again.front_m == m.front_m
    (d / "ao.png").unlink()
    assert hero_maps.load_rendered(d) is None  # a pass missing: no renders


def test_load_falls_back_to_the_drawn_car(tmp_path, monkeypatch):
    monkeypatch.setenv(hero_maps.RENDER_DIR_ENV, str(tmp_path / "nothing"))
    m = hero_maps.load()
    assert m.source == "drawn" and m.front_m == 0.0
    assert m.albedo.dtype == np.uint8 and m.albedo.shape[2] == 4


# ------------------------------------------------------------------------ light stage


def test_stage_blend_weights():
    xs, ys = (-1.0, 0.0, 1.0), (0.0, 1.0, 2.0)
    cells, outside = light_stage.blend_weights(xs, ys, 0.0, 1.0)  # exactly on a light
    assert outside == 0.0
    assert {(r, c): w for r, c, w in cells if w > 0} == {(1, 1): 1.0}
    cells, _ = light_stage.blend_weights(xs, ys, 0.5, 0.25)  # between four lights
    w = {(r, c): wt for r, c, wt in cells}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w == pytest.approx({(0, 1): 0.375, (0, 2): 0.375, (1, 1): 0.125, (1, 2): 0.125})
    cells, outside = light_stage.blend_weights(xs, ys, -4.0, 1.0)  # over the menu
    assert outside == pytest.approx(3.0)
    assert {(r, c) for r, c, wt in cells if wt > 0} == {(1, 0)}  # the nearest column


def _write_stage(d, complete=True, w=12, h=40):
    from PIL import Image

    d.mkdir(parents=True)
    xs, ys = (-0.5, 0.5), (0.5, 1.5, 2.5)
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            a = np.zeros((h, w, 4), np.uint8)
            a[..., 0] = 10 * (iy * len(xs) + ix)
            a[..., 3] = 255
            Image.fromarray(a).save(d / f"light_{iy}_{ix}.png")
    amb = np.full((h, w, 4), 30, np.uint8)
    amb[..., 3] = 255
    Image.fromarray(amb).save(d / "ambient.png")
    info = {"px_per_m": 10.0, "margin_m": 0.1, "xs": xs, "ys": ys, "complete": complete}
    (d / "stage.json").write_text(json.dumps(info), encoding="utf-8")


def test_light_stage_loads_fades_and_caches(tmp_path):
    d = tmp_path / "stage"
    _write_stage(d)
    st = light_stage.load(d)
    assert st.images.shape == (6, 40, 12, 3) and st.ambient.shape == (40, 12, 4)
    assert st.images[5, 0, 0, 0] == 50  # photo order: row * cols + col
    assert st.size_m == pytest.approx((1.2, 4.0))
    assert st.ambient[0, 0, 3] == 255 and st.ambient[-1, 0, 3] < 20  # faded at the back
    assert (d / light_stage.CACHE_NAME).is_file()
    assert np.array_equal(light_stage.load(d).images, st.images)  # from the cache
    (d / "light_2_1.png").unlink()
    assert light_stage.load(d) is None  # a photo missing


def test_light_stage_ignores_test_renders(tmp_path):
    _write_stage(tmp_path / "stage", complete=False)
    assert light_stage.load(tmp_path / "stage") is None
    assert light_stage.load(tmp_path / "missing") is None
