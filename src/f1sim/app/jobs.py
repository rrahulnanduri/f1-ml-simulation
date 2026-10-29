"""Downloads the menu runs in the background, so the window never needs a terminal.

What ``f1sim build-track`` and ``f1sim ghost`` did on the command line, as :class:`Job` s a
button can start: :func:`get_track_data` builds a circuit and :func:`get_real_lap` makes the
ghost of a real driver's fastest lap. A :class:`Job` runs one of them in a thread and keeps
what the window shows: its state, how long it has run, and a plain-English message (also when
it fails: :func:`friendly_error`). No window or GL here, so it is tested without one.

A job cannot stop the library call it runs. :meth:`Job.cancel` only makes the window forget
it: the call finishes in the background and its result is dropped.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

NO_CONNECTION = "No connection to the data server."

FIRST_REAL_LAP_YEAR = 2023
"""OpenF1 (the lap source that needs no sign-in, D6) starts in this season."""

LAP_ERRORS = (KeyError, ValueError, FileNotFoundError, OSError, RuntimeError)


def friendly_error(exc: BaseException) -> str:
    """One plain sentence for what went wrong, with what to do about it."""
    name = type(exc).__name__.lower()
    text = str(exc).strip() or type(exc).__name__
    network = ("connection", "timeout", "urlerror", "httperror", "ssl", "resolve", "socket")
    if isinstance(exc, ConnectionError | TimeoutError) or any(w in name for w in network):
        return NO_CONNECTION
    if "alignment" in name or "trackbuild" in name:
        return f"This circuit could not be built from the data: {text}"
    if len(text) > 110:
        text = text[:107].rstrip() + "..."
    return text


@dataclass(frozen=True)
class JobSpec:
    """What a job does, for the window: a title (``Getting track data``) and a work function
    that returns the message to show when it is done."""

    title: str
    work: Callable[[], str]


class Job:
    """One background task. ``state`` is ``running``, ``done``, ``failed`` or ``cancelled``."""

    def __init__(self, spec: JobSpec, clock: Callable[[], float] = time.monotonic):
        self.spec = spec
        self.state = "running"
        self.message = ""
        self._clock = clock
        self._t0 = clock()
        self._t_end: float | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=spec.title, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            message, state = self.spec.work(), "done"
        except Exception as exc:  # noqa: BLE001 - whatever the library raises is shown
            message, state = friendly_error(exc), "failed"
        with self._lock:
            if self.state == "running":  # not cancelled meanwhile
                self.state, self.message, self._t_end = state, message, self._clock()

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def elapsed(self) -> float:
        """Seconds the job has run (frozen once it ends)."""
        end = self._t_end if self._t_end is not None else self._clock()
        return end - self._t0

    def cancel(self) -> None:
        with self._lock:
            if self.state == "running":
                self.state, self.message, self._t_end = "cancelled", "Cancelled.", self._clock()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the work function returns (tests; the window never waits)."""
        self._thread.join(timeout)


def format_elapsed(seconds: float) -> str:
    s = int(max(seconds, 0.0))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------------------- the work


def get_track_data(track_id: str, build: Callable | None = None) -> str:
    """Build ``track_id`` into the cache (TUM data aligned to a real lap, or real laps)."""
    if build is None:
        from f1sim.data.track_pipeline import build_and_cache as build
    report = build(track_id)
    return f"{report.track.name} is ready."


def real_lap_years(now: int | None = None) -> list[int]:
    """Seasons to try for a real lap, newest first."""
    year = datetime.date.today().year if now is None else now
    return list(range(year, FIRST_REAL_LAP_YEAR - 1, -1))


def get_real_lap(
    track_id: str,
    years: list[int] | None = None,
    build: Callable | None = None,
    save: Callable | None = None,
    path_for: Callable | None = None,
) -> str:
    """Make the ghost of the fastest qualifying lap at ``track_id`` in the newest season that
    has one, and save it where the menu looks for ghosts."""
    if build is None or save is None or path_for is None:
        from f1sim.data import ghost_builder

        build = build or ghost_builder.build_ghost
        save = save or ghost_builder.save_ghost
        path_for = path_for or ghost_builder.ghost_path
    errors: list[BaseException] = []
    for year in years or real_lap_years():
        try:
            ghost = build(track_id, year)
        except LAP_ERRORS as exc:
            errors.append(exc)
            continue
        driver = ghost.meta["driver"]
        save(ghost, path_for(year, track_id, "Q", driver))
        return f"Got {driver}'s {year} qualifying lap."
    for exc in errors:  # a network problem is the thing to tell the user about
        if friendly_error(exc) == NO_CONNECTION:
            raise exc
    seasons = f"{min(years or real_lap_years())}-{max(years or real_lap_years())}"
    raise LookupError(f"No qualifying lap was found for this circuit in {seasons}.")
