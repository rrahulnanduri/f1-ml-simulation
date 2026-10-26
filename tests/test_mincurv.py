import numpy as np
import pytest

from _geometry_helpers import circle, stadium
from f1sim.data.track_builder import build_track
from f1sim.sim.car_model import CarParams
from f1sim.sim.mincurv import EDGE_MARGIN_M, min_curvature_line
from f1sim.sim.raceline import raceline_for
from f1sim.sim.scripted import CAR_HALF_WIDTH_M, INSIDE_MARGIN_M, drive_reference_lap
from f1sim.track.track import Track

KEEP = CAR_HALF_WIDTH_M + INSIDE_MARGIN_M + EDGE_MARGIN_M


@pytest.fixture(scope="module")
def stadium_track() -> Track:
    x, y = stadium(300.0, 60.0, 3.0)
    n = x.size
    return Track(build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="stad", name="S"))


def test_line_stays_inside_the_margin(stadium_track):
    x, y = min_curvature_line(stadium_track)
    s, n, _ = stadium_track.project(x, y)
    w_left, w_right = stadium_track.widths_at(s)
    assert np.all(n <= w_left - KEEP + 1e-3)
    assert np.all(n >= -(w_right - KEEP) - 1e-3)


def test_line_uses_the_width_on_a_stadium(stadium_track):
    """Through the bends the line swings out and in: it uses most of the width."""
    x, y = min_curvature_line(stadium_track)
    _, n, _ = stadium_track.project(x, y)
    assert n.max() - n.min() > 2 * (6.0 - KEEP) - 0.5


def test_circle_line_is_the_outer_circle():
    """On a constant-width circle the least curved line is the biggest circle that fits."""
    x, y = circle(150.0, 300)
    n = x.size
    track = Track(build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="c", name="C"))
    lx, ly = min_curvature_line(track)
    _, off, _ = track.project(lx, ly)
    outer = -(6.0 - KEEP) if np.mean(off) < 0 else 6.0 - KEEP
    np.testing.assert_allclose(off, outer, atol=0.05)


def test_faster_than_the_centre_line(stadium_track):
    car = CarParams()
    centre = drive_reference_lap(stadium_track, car)
    line = drive_reference_lap(stadium_track, car, raceline_xy=min_curvature_line(stadium_track))
    assert line.completed and np.isfinite(line.lap_time_s)
    assert line.lap_time_s < centre.lap_time_s - 0.3


def test_raceline_for_keeps_the_centre_line_off_catalogue(stadium_track):
    xy, source = raceline_for(stadium_track)
    assert xy is None and source == "centre-line"


def test_raceline_for_computes_one_for_a_catalogue_circuit_without_tum():
    """Marina Bay is in the catalogue but not in TUM's database (no download needed)."""
    x, y = stadium(300.0, 60.0, 3.0)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="marina_bay", name="M")
    xy, source = raceline_for(Track(data))
    assert source == "min-curvature"
    assert xy is not None and xy[0].shape == xy[1].shape


def test_raceline_for_skips_tum_on_a_track_built_from_laps(monkeypatch):
    """A lap-built track may be a newer layout than TUM's raceline (Albert Park, Barcelona)."""
    from f1sim.sim import raceline

    called = []
    monkeypatch.setattr(raceline, "tum_raceline", lambda track_id: called.append(track_id))
    x, y = stadium(300.0, 60.0, 3.0)
    n = x.size
    data = build_track(
        x,
        y,
        np.full(n, 6.0),
        np.full(n, 6.0),
        track_id="melbourne",
        name="M",
        source={"geometry": "openf1-laps"},
    )
    xy, source = raceline_for(Track(data))
    assert source == "min-curvature" and not called


def test_raceline_for_drops_a_tum_line_the_driver_cannot_lap(monkeypatch):
    """Shanghai's TUM line runs off the track, so the scripted lap fails: compute one."""
    from types import SimpleNamespace

    from f1sim.sim import raceline

    x, y = stadium(300.0, 60.0, 3.0)
    n = x.size
    data = build_track(x, y, np.full(n, 6.0), np.full(n, 6.0), track_id="shanghai", name="S")
    monkeypatch.setattr(raceline, "tum_raceline", lambda track_id: (x, y))
    failed = SimpleNamespace(completed=True, lap_time_s=float("nan"))
    monkeypatch.setattr(raceline.scripted, "drive_reference_lap", lambda *a, **kw: failed)
    xy, source = raceline_for(Track(data))
    assert source == "min-curvature"
