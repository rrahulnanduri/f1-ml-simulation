"""The baked loader asset (``f1sim.geo`` v1) and the build-time readers in ``scripts/geo``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src" / "f1sim" / "app" / "assets" / "geo" / "silverstone.npz"
sys.path.insert(0, str(ROOT / "scripts" / "geo"))

import geodata  # noqa: E402


@pytest.fixture(scope="module")
def asset():
    with np.load(ASSET, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_asset_is_small_and_versioned(asset):
    assert ASSET.stat().st_size < 64 * 1024
    assert str(asset["format"]) == "f1sim.geo" and int(asset["version"]) == 1
    assert "OpenStreetMap" in str(asset["attribution"])


def test_fit_and_elevation_are_sane(asset):
    rms, p95, _, scale, _, loop_rms = asset["fit"]
    assert rms < 3.0 and p95 < 5.0 and loop_rms < 3.0
    assert abs(scale - 1.0) < 0.01  # both are metres
    length = float(asset["length_m"])
    elev = asset["elev_rel_m"].astype(float)
    assert len(elev) == int(np.ceil(length))  # one per metre of centre line
    assert elev.min() == 0.0 and 2.0 < elev.max() < 40.0  # Silverstone is gently rolling
    assert np.isfinite(asset["grid_rel_m"].astype(float)).all()


def test_osm_loop_matches_the_circuit(asset):
    loop = asset["loop_xy"].astype(float)
    closed = np.vstack([loop, loop[:1]])
    perimeter = np.hypot(*np.diff(closed, axis=0).T).sum()
    assert perimeter == pytest.approx(float(asset["length_m"]), rel=0.02)
    centre = asset["centre_xy"].astype(float)
    assert np.hypot(*(loop[0] - centre[0])) < 15.0  # starts at the start line


def test_mercator_round_trip():
    lat, lon = np.array([52.07, -33.9]), np.array([-1.02, 151.2])
    back = geodata.inverse_mercator(*geodata.mercator(lat, lon))
    np.testing.assert_allclose(back, (lat, lon), atol=1e-9)
    frame = geodata.LocalFrame(52.07, -1.02)
    # 0.001 deg of latitude is about 111 m on the ground
    assert frame.from_latlon(52.071, -1.02)[1] == pytest.approx(111.3, abs=0.5)


def test_floating_point_predictor_decodes():
    rng = np.random.default_rng(0)
    z = (100 + rng.normal(size=(4, 6)) * 5).astype(np.float32)
    planes = z.astype(">f4").view(np.uint8).reshape(4, 6, 4).transpose(0, 2, 1).reshape(4, 24)
    coded = np.diff(planes, axis=1, prepend=np.zeros((4, 1), np.uint8)).astype(np.uint8)
    out = geodata._unpredict(coded.tobytes(), 6, 4, np.dtype("<f4"), 3)
    np.testing.assert_array_equal(out, z)
