"""The paging card's state machine and easing, pure Python (times in seconds, passed in).

Pages slide vertically: :meth:`Pager.position` is a float page index that eases from where the
card was to the target page over ``ANIM_S``, so a page change in the middle of an animation
starts from the current on-screen position instead of jumping.
"""

from __future__ import annotations

ANIM_S = 0.38
"""Slide duration."""
WHEEL_DEBOUNCE_S = 0.40
"""After a wheel page change, further wheel notches are ignored for this long."""


def ease_out_cubic(u: float) -> float:
    """``1 - (1 - u)^3`` on ``u`` clamped to ``[0, 1]``: fast start, gentle stop."""
    u = min(max(float(u), 0.0), 1.0)
    return 1.0 - (1.0 - u) ** 3


class Pager:
    """Which of ``n_pages`` the card shows, with the slide animation between them."""

    def __init__(self, n_pages: int, page: int = 0):
        if n_pages < 1:
            raise ValueError("need at least one page")
        self.n_pages = int(n_pages)
        self.page = min(max(int(page), 0), self.n_pages - 1)
        self._from = float(self.page)
        self._t0 = float("-inf")
        self._last_wheel = float("-inf")

    def position(self, now: float) -> float:
        """Eased page position at time ``now``: ``page`` once the slide has finished."""
        u = (now - self._t0) / ANIM_S
        if u >= 1.0:
            return float(self.page)
        return self._from + (self.page - self._from) * ease_out_cubic(u)

    def animating(self, now: float) -> bool:
        return (now - self._t0) < ANIM_S

    def go_to(self, page: int, now: float) -> None:
        page = min(max(int(page), 0), self.n_pages - 1)
        if page == self.page:
            return
        self._from = self.position(now)
        self._t0 = float(now)
        self.page = page

    def next(self, now: float) -> None:
        """Next page, wrapping from the last back to the first (click and ``Tab``)."""
        self.go_to((self.page + 1) % self.n_pages, now)

    def wheel(self, notches: float, now: float) -> bool:
        """One page per wheel event (down = next, up = previous, no wrap), ignoring events
        within ``WHEEL_DEBOUNCE_S`` of the last page change by the wheel. Returns ``True``
        if the page changed."""
        if notches == 0 or (now - self._last_wheel) < WHEEL_DEBOUNCE_S:
            return False
        step = 1 if notches < 0 else -1  # arcade: scroll_y < 0 is the wheel turned down
        before = self.page
        self.go_to(self.page + step, now)
        if self.page != before:
            self._last_wheel = float(now)
            return True
        return False
