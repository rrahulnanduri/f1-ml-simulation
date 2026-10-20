import json
from dataclasses import replace

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data.io import (
    BUILDER_VERSION,
    read_track_builder_version,
    read_track_json,
    write_track_json,
)
from f1sim.data.track_builder import build_track
from f1sim.track.track import TrackData

DECIMALS = {"x": 4, "y": 4, "s": 4, "w_left": 4, "w_right": 4, "heading": 6, "curvature": 8}


@pytest.fixture(scope="module")
def track() -> TrackData:
    x, y = stadium(150.0, 35.0, 4.0)
    n = x.size
    d = build_track(
        x,
        y,
        np.linspace(5.0, 7.0, n),
        np.full(n, 6.0),
        track_id="oval",
        name="Oval Raceway",
        source={"geometry": "synthetic", "widths": "test"},
    )
    return TrackData(
        **{
            **d.__dict__,
            "sectors_s": (0.0, 123.456789, 456.0),
            "fastf1_location": "Nowhere",
            "fastf1_alignment": {"years_validated": [2024], "rotation_rad": 0.1, "scale": 1.0},
        }
    )


def rounded(d: TrackData) -> TrackData:
    arrays = {k: np.round(getattr(d, k), v) for k, v in DECIMALS.items()}
    return TrackData(
        **{
            **d.__dict__,
            **arrays,
            "length_m": round(d.length_m, 4),
            "sectors_s": tuple(round(v, 4) for v in d.sectors_s),
        }
    )


def test_round_trip(tmp_path, track):
    path = tmp_path / "sub" / "oval.track.json"
    write_track_json(track, path)
    back = read_track_json(path)
    assert back == rounded(track)
    np.testing.assert_allclose(back.x, track.x, atol=5e-5)
    np.testing.assert_allclose(back.curvature, track.curvature, atol=5e-9)
    # Writing what was read gives identical bytes.
    path2 = tmp_path / "again.track.json"
    write_track_json(back, path2)
    assert path2.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_file_layout(tmp_path, track):
    path = tmp_path / "oval.track.json"
    write_track_json(track, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "f1sim.track"
    assert doc["version"] == 2
    assert doc["built_with"]["builder_version"] == BUILDER_VERSION
    assert "f1sim" in doc["built_with"]
    assert set(doc["points"]) == set(DECIMALS)
    assert all(len(v) == track.n_points for v in doc["points"].values())
    for key in ("id", "name", "fastf1_location", "source", "resolution_m", "length_m"):
        assert key in doc
    assert doc["direction"] == "counterclockwise"
    assert doc["sectors_s"] == [0.0, 123.4568, 456.0]
    assert read_track_builder_version(path) == BUILDER_VERSION


def test_optional_fields_default(tmp_path, track):
    path = tmp_path / "min.track.json"
    write_track_json(track, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sectors_s", "fastf1_location", "fastf1_alignment", "direction", "source"):
        del doc[key]
    path.write_text(json.dumps(doc), encoding="utf-8")
    d = read_track_json(path)
    assert d.sectors_s == () and d.fastf1_location is None and d.fastf1_alignment is None
    assert d.direction == "" and d.source == {}


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"format": "something.else"}, "format"),
        ({"format": None}, "format"),
        ({"version": 3}, "version"),
        ({"version": "1"}, "version"),
    ],
)
def test_rejects_wrong_format_or_version(tmp_path, track, change, match):
    path = tmp_path / "bad.track.json"
    write_track_json(track, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(change)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_track_json(path)


def test_rejects_malformed(tmp_path, track):
    path = tmp_path / "bad.track.json"
    write_track_json(track, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["points"]["heading"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_track_json(path)
    path.write_text("[1, 2", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        read_track_json(path)


def test_limits_round_trip_and_v1_files_still_load(tmp_path, track):
    data = track.data if hasattr(track, "data") else track
    lim_l = data.w_left + 1.5
    lim_r = data.w_right + 0.5
    limited = replace(data, limit_left=lim_l, limit_right=lim_r, limits={"source": "test"})
    path = tmp_path / "limited.track.json"
    write_track_json(limited, path)
    back = read_track_json(path)
    assert back.has_limits and back.limits == {"source": "test"}
    np.testing.assert_allclose(back.limit_left, lim_l, atol=1e-4)
    np.testing.assert_allclose(back.limit_right, lim_r, atol=1e-4)
    # A version-1 file (no limit columns) still reads, with no limits
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["version"] = 1
    for col in ("limit_left", "limit_right"):
        doc["points"].pop(col)
    doc.pop("limits")
    v1 = tmp_path / "v1.track.json"
    v1.write_text(json.dumps(doc), encoding="utf-8")
    assert not read_track_json(v1).has_limits
