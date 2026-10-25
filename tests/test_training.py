"""Training view logic (f1sim.app.training) and the trainer's generation.npz, offline.

The window itself isn't opened; these tests drive the pure parts on the small oval of
tests/test_trainer.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from f1sim.app import training as tr
from f1sim.app.ui.layout import Rect
from f1sim.app.ui.train_hud import chart_points
from f1sim.train.metrics import GENERATION_KEYS, RunWriter, read_generation
from f1sim.train.trainer import Trainer
from test_trainer import T_REF, agents, controller_genome, small_config, track  # noqa: F401

COLOURS = [(i,) for i in range(5)]


def _generation(p: int = 3, n: int = 5, **kw) -> dict:
    g = {
        "gen": 4,
        "stage": 2,
        "params": np.zeros((p, n), dtype=np.float32),
        "fitness": np.arange(p, dtype=np.float64),
        "start_s": np.zeros(p),
        "start_v": np.full(p, 30.0),
        "episode_s": 10.0,
        "lap_time_s": np.full(p, np.nan),
        "progress_m": np.zeros(p),
    }
    g.update(kw)
    return g


# ------------------------------------------------------------------------ generation.npz


def test_write_and_read_generation(tmp_path):
    w = RunWriter(tmp_path / "run")
    path = w.write_generation(**_generation())
    g = read_generation(path)
    assert set(g) == set(GENERATION_KEYS)
    assert g["params"].dtype == np.float32 and int(g["gen"]) == 4
    with pytest.raises(KeyError):
        w.write_generation(**{k: v for k, v in _generation().items() if k != "gen"})
    with pytest.raises(ValueError):
        w.write_generation(**_generation(fitness=np.zeros(2)))
    assert read_generation(tmp_path / "missing.npz") is None


def test_trainer_writes_each_generation(track, agents, tmp_path):  # noqa: F811
    cfg = small_config(max_generations=2)
    Trainer(track, cfg, tmp_path / "run", reference_lap_s=T_REF).run()
    g = read_generation(tmp_path / "run" / "generation.npz")
    assert int(g["gen"]) == 1 and int(g["stage"]) == 1
    assert g["params"].shape[0] == cfg.population
    assert g["start_s"].shape == (cfg.population,)
    assert float(g["episode_s"]) == cfg.stage1_episode_s

    off = small_config(max_generations=1, write_generations=False)
    Trainer(track, off, tmp_path / "off", reference_lap_s=T_REF).run()
    assert not (tmp_path / "off" / "generation.npz").exists()


# --------------------------------------------------------------------------- playback


def _stage2_trainer(oval, policy, run_dir: Path) -> tuple[Trainer, np.ndarray]:
    cfg = small_config(sigma0=0.01)
    rng = np.random.default_rng(3)
    base = controller_genome(policy)
    params = (base + rng.normal(0.0, 0.02, (cfg.population, base.size))).astype(np.float32)
    params[1] = 0.0  # a genome that never moves (it stalls)
    trainer = Trainer(oval, cfg, run_dir, reference_lap_s=T_REF)
    trainer.writer.write_config(trainer.resolved_config())
    return trainer, params


def test_playback_drives_what_the_trainer_scored(track, agents, tmp_path):  # noqa: F811
    trainer, params = _stage2_trainer(track, agents, tmp_path / "run")
    fitness, stats = trainer.evaluate(params, 2)
    gen = _generation(
        params=params,
        fitness=fitness,
        start_s=stats["start_s"],
        start_v=stats["start_v"],
        episode_s=trainer.stage2_cap_s,
        lap_time_s=stats["lap_time_s"],
        progress_m=stats["progress_m"],
    )
    setup = tr.RunSetup.from_config(tmp_path / "run" / "config.toml")
    world = tr.default_make_world(track)(setup, params.shape[0], trainer.stage2_cap_s)
    pb = tr.GenerationPlayback(world, tr.default_make_act(setup), gen)
    assert pb.best == int(np.argmax(fitness))
    while pb.running.any():
        pb.step()
    assert np.isfinite(stats["lap_time_s"]).any()
    np.testing.assert_allclose(pb.lap_time, stats["lap_time_s"], equal_nan=True)
    np.testing.assert_allclose(pb.progress(), stats["progress_m"])
    states = pb.states()
    assert states[1] == "out"
    order = pb.order()
    assert order[0].state == "lap"
    laps = [r.lap_time for r in order if r.state == "lap"]
    assert laps == sorted(laps)


def test_playback_update_steps_in_real_time(track, agents, tmp_path):  # noqa: F811
    trainer, params = _stage2_trainer(track, agents, tmp_path / "run")
    setup = tr.RunSetup.from_config(tmp_path / "run" / "config.toml")
    world = tr.default_make_world(track)(setup, params.shape[0], trainer.stage2_cap_s)
    gen = _generation(
        params=params,
        fitness=np.zeros(params.shape[0]),
        start_s=np.full(params.shape[0], trainer.stage2_start_s),
        start_v=np.full(params.shape[0], trainer.stage2_start_v),
        episode_s=trainer.stage2_cap_s,
        lap_time_s=np.full(params.shape[0], np.nan),
        progress_m=np.zeros(params.shape[0]),
    )
    pb = tr.GenerationPlayback(world, tr.default_make_act(setup), gen, speed=2.0)
    pb.update(0.1)  # 0.2 s simulated = 6 decision steps of 1/30 s
    assert pb.steps == 6
    pb.update(0.01)  # 0.02 s: no new step, the pose is drawn between the last two
    assert pb.steps == 6 and 0.0 < pb.alpha < 1.0
    x, _, _ = pb.poses()
    x_prev, x_now = pb._prev[0], world.car.x
    lo, hi = np.minimum(x_prev, x_now), np.maximum(x_prev, x_now)
    assert np.all((x >= lo - 1e-9) & (x <= hi + 1e-9))
    pb.paused = True
    pb.update(1.0)
    assert pb.steps == 6
    pb.paused = False
    pb.update(10.0)  # capped at MAX_FRAME_S
    assert pb.steps == 6 + round(tr.MAX_FRAME_S * 2.0 / pb.decision_dt)


def test_watch_loads_generations_as_they_come(track, agents, tmp_path):  # noqa: F811
    run = tmp_path / "run"
    watch = tr.TrainingWatch(run, tr.default_make_world(track), tr.default_make_act)
    assert watch.update(1.0) is False and watch.playback is None
    assert watch.status() == ("Waiting for a generation", False)

    Trainer(track, small_config(max_generations=2), run, reference_lap_s=T_REF).run()
    assert watch.update(1.0) is True
    assert watch.playback.gen == 1 and watch.newest_gen == 1
    assert watch.status() == ("Watching", False)
    first = watch.playback
    for _ in range(400):  # stage-1 episodes are 3 s; the next one starts when it's over
        if watch.update(0.05):
            break
    assert watch.playback is not first and watch.playback.gen == 1
    watch.set_speed(8.0)
    assert watch.playback.speed == 8.0


# ---------------------------------------------------------------------------- helpers


def test_read_metrics_stops_at_a_half_written_row(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("gen,best_lap_s\n0,\n1,90.5\n2,8x\n", encoding="utf-8")
    rows = tr.read_metrics(path)
    assert len(rows) == 2 and math.isnan(rows[0]["best_lap_s"])
    gens, laps = tr.best_lap_history(rows)
    assert gens.tolist() == [1.0] and laps.tolist() == [90.5]
    assert tr.run_best_lap(rows) == 90.5
    assert tr.read_metrics(tmp_path / "none.csv") == []


def test_speed_steps():
    assert tr.faster(2.0) == 4.0 and tr.faster(16.0) == 16.0
    assert tr.slower(2.0) == 1.0 and tr.slower(0.5) == 0.5


def test_tower_rows():
    order = [
        tr.CarRow(2, "lap", 80.0, 1000.0),
        tr.CarRow(0, "lap", 80.5, 1000.0),
        tr.CarRow(1, "run", math.nan, 734.4),
        tr.CarRow(3, "out", math.nan, 12.0),
        tr.CarRow(4, "time", math.nan, 5.0),
    ]
    rows = tr.tower_rows(order, best=0, colours=COLOURS)
    assert [r.value for r in rows] == ["1:20.000", "+0.500", "734 m", "OUT", "TIME"]
    assert [r.name for r in rows][:2] == ["Car 3", "Best"]
    assert rows[1].is_self and rows[0].colour == (2,) and rows[2].colour == (1,)
    no_lap = tr.tower_rows([tr.CarRow(0, "run", math.nan, 1234.0)], best=0, colours=COLOURS)
    assert no_lap[0].value == "1,234 m"


def test_status_text():
    class P:
        def __init__(self, code, stopped=False):
            self.returncode, self.stopped, self.log_path = code, stopped, Path("train.log")

    setup = object()
    assert tr.status_text(P(None), False, None)[0] == "Copying the scripted driver"
    assert tr.status_text(P(None), False, setup)[0] == "Evaluating the first generation"
    assert tr.status_text(P(None), True, setup) == ("Training", False)
    assert tr.status_text(P(0), True, setup) == ("Finished", False)
    assert tr.status_text(P(1, stopped=True), True, setup) == ("Stopped", False)
    assert tr.status_text(P(1), True, setup) == ("Training failed: see train.log", True)


def test_new_run_dir_and_command(tmp_path):
    a = tr.new_run_dir("spa", tmp_path)
    a.mkdir()
    b = tr.new_run_dir("spa", tmp_path)
    assert a.name.startswith("spa-") and b != a and b.name.startswith(a.name)
    cmd = tr.train_command("spa", b)
    assert cmd[-4:] == ["--track", "spa", "--run-dir", str(b)]


def test_chart_points():
    r = Rect(0, 0, 100, 50)
    assert chart_points(np.zeros(0), np.zeros(0), r) == ([], [])
    pts, line = chart_points(np.array([0.0, 5.0, 10.0]), np.array([90.0, 85.0, 88.0]), r)
    assert pts[0] == (0.0, 0.0) and pts[1] == (50.0, 50.0) and pts[2][0] == 100.0
    assert [y for _, y in line] == [0.0, 50.0, 50.0]  # best so far never gets worse
    (single,), _ = chart_points(np.array([3.0]), np.array([90.0]), r)
    assert single == (0.0, 25.0)
