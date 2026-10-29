"""The menu's background downloads (f1sim.app.jobs): no window, no network."""

from types import SimpleNamespace

import pytest

from f1sim.app import jobs


def _job(work):
    job = jobs.Job(jobs.JobSpec("test", work))
    job.wait(5)
    return job


def test_a_job_that_returns_is_done_with_its_message():
    job = _job(lambda: "ready")
    assert (job.state, job.message) == ("done", "ready")
    assert not job.running


def test_a_job_that_raises_fails_with_a_plain_message():
    def boom():
        raise ConnectionError("getaddrinfo failed")

    job = _job(boom)
    assert job.state == "failed"
    assert job.message == jobs.NO_CONNECTION


def test_a_cancelled_job_drops_its_result():
    import threading

    gate = threading.Event()
    job = jobs.Job(jobs.JobSpec("slow", lambda: (gate.wait(5), "late")[1]))
    assert job.running
    job.cancel()
    gate.set()
    job.wait(5)
    assert (job.state, job.message) == ("cancelled", "Cancelled.")


def test_elapsed_time_is_shown_as_minutes_and_seconds():
    assert jobs.format_elapsed(0) == "0:00"
    assert jobs.format_elapsed(75.9) == "1:15"


def test_long_unknown_errors_are_cut():
    msg = jobs.friendly_error(ValueError("x" * 300))
    assert len(msg) <= 110 and msg.endswith("...")


def test_track_data_uses_the_builder_and_names_the_circuit():
    seen = []

    def build(track_id):
        seen.append(track_id)
        return SimpleNamespace(track=SimpleNamespace(name="Silverstone Circuit"))

    assert jobs.get_track_data("silverstone", build=build) == "Silverstone Circuit is ready."
    assert seen == ["silverstone"]


def test_real_lap_years_go_from_the_newest_season_back_to_2023():
    assert jobs.real_lap_years(2026) == [2026, 2025, 2024, 2023]


def test_real_lap_takes_the_newest_season_that_has_one_and_saves_it():
    saved = {}

    def build(track_id, year):
        if year > 2024:
            raise ValueError("no session")
        return SimpleNamespace(meta={"driver": "RUS"})

    msg = jobs.get_real_lap(
        "silverstone",
        years=[2026, 2025, 2024, 2023],
        build=build,
        save=lambda g, p: saved.update(path=p),
        path_for=lambda y, t, k, d: f"{y}_{t}_{k}_{d}.npz",
    )
    assert msg == "Got RUS's 2024 qualifying lap."
    assert saved["path"] == "2024_silverstone_Q_RUS.npz"


def test_real_lap_reports_a_network_problem_not_just_no_lap():
    def build(track_id, year):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        jobs.get_real_lap("x", years=[2025], build=build, save=print, path_for=print)


def test_real_lap_with_no_data_says_so():
    def build(track_id, year):
        raise ValueError("no session")

    with pytest.raises(LookupError, match="2024-2025"):
        jobs.get_real_lap("x", years=[2025, 2024], build=build, save=print, path_for=print)
