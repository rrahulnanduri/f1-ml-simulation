"""The text "scramble/decode" effect of the ticker line, pure Python (times in seconds).

When a value changes, its letters and digits cycle through random characters of the same kind
(letters stay letters, digits stay digits, so tabular digits don't change width) and settle
left to right over ``DURATION_S``. Spaces and punctuation never scramble. A value that doesn't
change costs nothing: :meth:`Scramble.text` returns the stored string.

The random characters change every ``STEP_S`` (not every frame) and are a pure function of the
value, the step number and a seed, so the effect is deterministic and testable.
"""

from __future__ import annotations

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
DURATION_S = 0.25
STEP_S = 1.0 / 30.0
"""New random characters this often while scrambling."""
HOLD = 0.35
"""Share of the duration during which every character scrambles before the first settles."""


def _rand(seed: int, step: int, i: int) -> int:
    """A small deterministic hash -> non-negative int (no global RNG state)."""
    x = (seed * 0x9E3779B1 + step * 0x85EBCA77 + i * 0xC2B2AE3D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x


def scramble_char(ch: str, r: int) -> str:
    """``ch`` replaced by a random character of its kind (``r`` picks it); other characters
    are returned unchanged."""
    if ch.isdigit():
        return DIGITS[r % len(DIGITS)]
    if ch.isalpha():
        out = LETTERS[r % len(LETTERS)]
        return out if ch.isupper() else out.lower()
    return ch


def settle_time(i: int, n: int, duration: float = DURATION_S) -> float:
    """Seconds after a change at which character ``i`` of ``n`` shows its final value."""
    if n <= 0:
        return 0.0
    return duration * (HOLD + (1.0 - HOLD) * (i + 1) / n)


class Scramble:
    """One value with the decode animation."""

    def __init__(self, text: str = "", seed: int = 0, duration: float = DURATION_S):
        self.target = text
        self.seed = int(seed)
        self.duration = float(duration)
        self._t0 = float("-inf")

    def set(self, text: str, now: float) -> bool:
        """New value at time ``now``; starts the animation if it differs. Returns whether it
        changed."""
        if text == self.target:
            return False
        self.target = text
        self._t0 = float(now)
        return True

    def animating(self, now: float) -> bool:
        return (now - self._t0) < self.duration

    def text(self, now: float) -> str:
        """What to show at time ``now``."""
        dt = now - self._t0
        if dt >= self.duration:
            return self.target
        step = int(max(dt, 0.0) / STEP_S)
        n = len(self.target)
        out = []
        for i, ch in enumerate(self.target):
            if dt >= settle_time(i, n, self.duration):
                out.append(ch)
            else:
                out.append(scramble_char(ch, _rand(self.seed, step, i)))
        return "".join(out)


class ScrambleLine:
    """Several values joined by ``sep``; each scrambles on its own when it changes."""

    def __init__(self, n: int, sep: str = " · "):
        self.parts = [Scramble("", seed=k + 1) for k in range(n)]
        self.sep = sep

    def set(self, values: list[str], now: float) -> bool:
        if len(values) != len(self.parts):
            raise ValueError(f"expected {len(self.parts)} values, got {len(values)}")
        changed = False
        for part, value in zip(self.parts, values, strict=True):
            changed |= part.set(value, now)
        return changed

    def animating(self, now: float) -> bool:
        return any(p.animating(now) for p in self.parts)

    def text(self, now: float) -> str:
        return self.sep.join(p.text(now) for p in self.parts if p.target)
