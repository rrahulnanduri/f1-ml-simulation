"""The track picker's flags (f1sim.app.ui.flags): every circuit has one, and the fit maths."""

import pytest

from f1sim.app.ui import flags
from f1sim.data import catalog


def test_every_circuit_has_a_flag():
    assert {e.id for e in catalog.CATALOG.values()} <= set(flags.FLAGS)
    assert flags.flag_for("nowhere") is None


def test_stripes_cover_the_flag_in_order():
    for f in flags.FLAGS.values():
        bands = f.bands()
        assert bands[0][0] == 0.0
        assert bands[-1][1] == pytest.approx(1.0)
        assert all(a[1] == pytest.approx(b[0]) for a, b in zip(bands, bands[1:], strict=False))
    spain = flags.flag_for("madrid").bands()
    assert [round(b - a, 2) for a, b, _ in spain] == [0.25, 0.5, 0.25]


def test_shapes_stay_inside_the_flag():
    for f in flags.FLAGS.values():
        for sh in f.shapes:
            if isinstance(sh, flags.Rect):
                assert 0.0 <= sh.x0 < sh.x1 <= 1.0 and 0.0 <= sh.y0 < sh.y1 <= 1.0
            elif isinstance(sh, flags.Line):
                assert all(0.0 <= v <= 1.0 for v in (sh.x0, sh.y0, sh.x1, sh.y1))
            else:
                assert 0.0 <= sh.x <= 1.0 and 0.0 <= sh.y <= 1.0
                assert sh.r <= 0.5  # of the height: fits the box


def test_fit_scale_shrinks_text_to_the_room_down_to_a_floor():
    assert flags.fit_scale(100.0, 200.0) == 1.0
    assert flags.fit_scale(300.0, 150.0) == pytest.approx(0.6)  # floor (0.5 would be too small)
    assert flags.fit_scale(200.0, 160.0) == pytest.approx(0.8)
