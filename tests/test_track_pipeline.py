"""Offline tests for the catalogue -> TUM -> build -> align -> cache pipeline and the CLI.

TUM and FastF1 are replaced by a synthetic asymmetric circuit and a fake qualifying lap that runs
the other way round, starts mid-lap and lives in a rotated, shifted frame.
"""

import numpy as np
import pytest

from _geometry_helpers import circle, figure_eight, teardrop
from f1sim import cli
from f1sim.data import catalog, laps, track_pipeline, tum_source
from f1sim.data.catalog import CatalogEntry
from f1sim.data.laps import LapXY
from f1sim.data.track_builder import wall_self_crossings
from f1sim.track.track import Track

ENTRY = CatalogEntry(
    id="testring", name="Test Ring", tum_name="TestRing", fastf1_locations=("Testville",)
)
HALF_WIDTH = 6.0
SPEED = 50.0  # m/s, constant, so lap time maps directly to distance
ROT, TX, TY = 1.1, -1800.0, 950.0  # FastF1 frame = track frame rotated and shifted by this


def _raw_tum():
    x, y = teardrop(30.0, 120.0, 400.0, spacing=5.0)
    w = np.full(x.size, HALF_WIDTH)
    return tum_source.TumTrack(x=x, y=y, w_right=w, w_left=w.copy())


def _fake_lap(raw, start_frac=0.4, reverse=True, offset=2.0):
    """A lap in the FastF1 frame: driven against TUM's point order, starting `start_frac` round."""
    x, y = raw.x, raw.y
    if reverse:
        x, y = x[::-1], y[::-1]
    k = int(start_frac * x.size)
    x, y = np.roll(x, -k), np.roll(y, -k)
    # Shift sideways a little, like a racing line
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    x, y = x - np.sin(heading) * offset, y + np.cos(heading) * offset
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    lap_time = dist[-1] / SPEED
    # Track frame -> FastF1 frame: rotate by ROT, then shift
    c, s = np.cos(ROT), np.sin(ROT)
    fx, fy = c * x - s * y + TX, s * x + c * y + TY
    return LapXY(
        driver="TST",
        lap_time_s=lap_time,
        t=dist / SPEED,
        x_m=fx,
        y_m=fy,
        sector_end_t=(lap_time / 3, 2 * lap_time / 3),
    ), (x[0], y[0])


@pytest.fixture
def fake_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    raw = _raw_tum()
    monkeypatch.setattr(catalog, "get", lambda track_id: ENTRY)
    monkeypatch.setattr(tum_source, "fetch_tum_csv", lambda name: tmp_path / "unused.csv")
    monkeypatch.setattr(tum_source, "parse_tum_csv", lambda path: raw)
    lap, start = _fake_lap(raw)
    monkeypatch.setattr(laps, "reference_lap", lambda track_id, year, **kw: (lap, "openf1"))
    return raw, lap, start


def test_build_orients_track_like_the_lap_and_sets_sectors(fake_sources):
    raw, lap, lap_start = fake_sources
    report = track_pipeline.build_and_cache("testring", year=2024)
    t = report.track

    # TUM order is counterclockwise; the lap runs the other way, so the track must too
    assert t.direction == "clockwise"
    # s = 0 sits where the lap starts (lap start is 2 m off the centre line)
    assert np.hypot(t.x[0] - lap_start[0], t.y[0] - lap_start[1]) < 6.0
    # Constant speed: sector ends at 1/3 and 2/3 of the lap time are 1/3 and 2/3 of the distance
    assert t.sectors_s[0] == 0.0
    assert t.sectors_s[1] == pytest.approx(t.length_m / 3, rel=0.02)
    assert t.sectors_s[2] == pytest.approx(2 * t.length_m / 3, rel=0.02)
    assert wall_self_crossings(t) == (0, 0)

    a = report.alignment
    assert a.inside_ratio >= 0.98 and a.median_dist_m < 3.0
    assert t.fastf1_alignment["reference_driver"] == "TST"
    assert t.fastf1_alignment["lap_source"] == "openf1"
    assert t.fastf1_location == "Testville"

    # The cached file loads back as the same track
    assert report.path.exists()
    loaded = track_pipeline.load_track("testring")
    assert loaded.length == pytest.approx(t.length_m)
    assert loaded.data.sectors_s == t.sectors_s


def test_lap_from_a_different_layout_is_rejected(fake_sources, monkeypatch):
    cx, cy = circle(300.0, n=400)
    wrong = LapXY("TST", 40.0, np.linspace(0, 40, cx.size), cx, cy, (13.0, 26.0))
    monkeypatch.setattr(laps, "reference_lap", lambda track_id, year, **kw: (wrong, "openf1"))
    with pytest.raises(track_pipeline.AlignmentRejected, match="layout may have changed"):
        track_pipeline.build_and_cache("testring", year=2024)
    assert not track_pipeline.track_path("testring").exists()


def test_build_without_fastf1_keeps_tum_order(fake_sources):
    report = track_pipeline.build_and_cache("testring", year=None)
    assert report.alignment is None
    assert report.track.direction == "counterclockwise"
    assert report.track.sectors_s == ()


def test_track_without_tum_data_needs_the_fallback(monkeypatch):
    no_tum = CatalogEntry(id="monaco", name="Monaco", tum_name=None, fastf1_locations=("Monaco",))
    monkeypatch.setattr(catalog, "get", lambda track_id: no_tum)
    with pytest.raises(NotImplementedError, match="fallback arrives in M5"):
        track_pipeline.build_and_cache("monaco")


def test_cli_build_and_list(fake_sources, capsys):
    assert cli.main(["build-track", "testring"]) == 0
    out = capsys.readouterr().out
    assert "0 left / 0 right self-crossings" in out
    assert "reference    TST" in out
    assert Track(track_pipeline.load_track("testring").data).length > 0


def test_cli_reports_a_figure_eight_as_an_error(fake_sources, monkeypatch, capsys):
    x, y = figure_eight()
    w = np.full(x.size, 5.0)
    crossing = tum_source.TumTrack(x=x, y=y, w_right=w, w_left=w.copy())
    monkeypatch.setattr(tum_source, "parse_tum_csv", lambda path: crossing)
    assert cli.main(["build-track", "testring", "--no-fastf1"]) == 1
    assert "crosses itself" in capsys.readouterr().err


def test_cli_plot_track(fake_sources, tmp_path, capsys):
    assert cli.main(["build-track", "testring"]) == 0
    out = tmp_path / "ring.png"
    assert cli.main(["plot-track", "testring", "--out", str(out), "--dpi", "40"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_plot_unbuilt_track_is_an_error(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    assert cli.main(["plot-track", "silverstone"]) == 1
    assert "isn't built" in capsys.readouterr().err


def test_cli_reports_unknown_track(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    assert cli.main(["build-track", "nowhere", "--no-fastf1"]) == 1
    assert "error:" in capsys.readouterr().err
