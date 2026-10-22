"""Sharp text on HiDPI displays (f1sim.app.ui.theme.SharpText): offline, no window."""

import pytest

from f1sim.app.ui import theme


class _Label:
    dpi = 96


class _Text:
    """Stands in for arcade.Text: lays out in framebuffer pixels, its size grows with dpi."""

    def __init__(self):
        self.label = _Label()
        self.position = (0, 0)
        self.x = self.y = 0
        self.tracking = 0.0
        self.width = None

    @property
    def content_width(self):
        return 50 * self.label.dpi / 96  # 50 logical pixels wide

    @property
    def content_height(self):
        return 12 * self.label.dpi / 96


def test_logical_layout_framebuffer_glyphs(monkeypatch):
    monkeypatch.setattr(theme, "pixel_ratio", lambda: 2.0)  # a 2x display
    t = theme.SharpText(_Text(), tracking_px=1.5)
    assert t._t.label.dpi == 192  # glyphs rastered at the framebuffer's resolution
    assert t._t.tracking == 3.0
    t.position = (100.3, 40.6)
    assert t.position == (100.3, 40.6)  # the layouts keep working in logical pixels
    assert t._t.position == (201, 81)  # drawn on whole framebuffer pixels
    assert t.content_width == pytest.approx(50.0)  # measured back in logical pixels
    t.width = 440
    assert t.width == 440 and t._t.width == 880
    t.x = 10
    assert t._t.position == (20, 81)


def test_it_follows_the_pixel_ratio(monkeypatch):
    assert theme.pixel_ratio() == 1.0  # no window: 1
    t = theme.SharpText(_Text())
    t.position = (11, 7)
    assert t._t.position == (11, 7) and t._t.label.dpi == 96
    monkeypatch.setattr(theme, "pixel_ratio", lambda: 1.5)  # moved to a 150 % display
    assert t.content_width == pytest.approx(50.0)  # any use re-syncs
    assert t._t.label.dpi == 144
    assert t._t.position == (16, 10)  # round(16.5) -> 16 (banker's), round(10.5) -> 10
