"""Trainer and run-folder tests on a small synthetic oval (offline, a few seconds each).

The trainer imports ``f1sim.agents.policy``, ``.optimisers`` and ``.checkpoint`` lazily. The
``agents`` fixture uses the real modules when they exist and otherwise installs the minimal
stand-ins below in ``sys.modules`` (set ``F1SIM_TRAINER_STANDINS=1`` to force them). The tests
rely only on what docs/m3-interfaces.md sections I and J guarantee: the flat parameter layout,
``make_optimiser(name, dim, population=, sigma0=, seed=, init=)`` starting from the ``init``
rows, and the checkpoint meta keys.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import math
import os
import sys
import tomllib
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from _geometry_helpers import stadium
from f1sim.data.track_builder import build_track
from f1sim.sim.car_model import CarParams
from f1sim.sim.sensors import OBS_SIZE, OBS_SPEC_VERSION
from f1sim.track.track import Track
from f1sim.train.metrics import METRICS_COLUMNS, TRACE_COLUMNS, to_toml
from f1sim.train.trainer import TrainConfig, Trainer, load_train_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_default.toml"
STRAIGHT, RADIUS, HALF_WIDTH = 200.0, 60.0, 7.0
HIDDEN = (8, 8)
T_REF = 30.0
META_KEYS = {
    "obs_spec_version",
    "policy",
    "car_params_hash",
    "track_id",
    "seed",
    "generation",
    "fitness",
    "best_lap_s",
    "git_commit",
    "created",
}


# --------------------------------------------------------------------------- stand-ins


def _standin_policy() -> types.ModuleType:
    mod = types.ModuleType("f1sim.agents.policy")

    @dataclass(frozen=True)
    class PolicySpec:
        obs_size: int = OBS_SIZE
        hidden: tuple[int, ...] = (64, 64)
        action_size: int = 2
        activation: str = "tanh"

        @property
        def sizes(self) -> tuple[int, ...]:
            return (self.obs_size, *self.hidden, self.action_size)

        @property
        def n_params(self) -> int:
            s = self.sizes
            return sum(a * b + b for a, b in zip(s[:-1], s[1:], strict=True))

        def to_dict(self) -> dict:
            d = asdict(self)
            d["hidden"] = list(self.hidden)
            return d

        @classmethod
        def from_dict(cls, d: dict) -> PolicySpec:
            return cls(**{**d, "hidden": tuple(d["hidden"])})

    def unflatten(spec, params):
        params = np.asarray(params)
        lead = params.shape[:-1]
        out, off = [], 0
        for a, b in zip(spec.sizes[:-1], spec.sizes[1:], strict=True):
            w = params[..., off : off + a * b].reshape(*lead, a, b)
            off += a * b
            out.append((w, params[..., off : off + b]))
            off += b
        return out

    def init_params(spec, n, rng):
        rows = []
        for a, b in zip(spec.sizes[:-1], spec.sizes[1:], strict=True):
            rows += [rng.normal(0.0, 1.0 / math.sqrt(a), (n, a * b)), np.zeros((n, b))]
        return np.concatenate(rows, axis=1).astype(np.float32)

    def act(spec, params, obs):
        obs = np.asarray(obs, dtype=np.float64)
        params = np.broadcast_to(
            np.asarray(params, dtype=np.float64), (obs.shape[0], spec.n_params)
        )
        h = obs
        for w, b in unflatten(spec, params):
            h = np.tanh(np.einsum("ni,nio->no", h, w) + b)
        return h

    mod.ACTION_SIZE = 2
    mod.PolicySpec = PolicySpec
    mod.unflatten = unflatten
    mod.init_params = init_params
    mod.act = act
    return mod


def _standin_optimisers() -> types.ModuleType:
    mod = types.ModuleType("f1sim.agents.optimisers")

    class GA:
        """Truncation selection + Gaussian mutation, 2 elites; deterministic per seed."""

        def __init__(self, dim, *, population=64, sigma0=0.1, seed=0, init=None, elites=2):
            self.dim, self.population, self.sigma = dim, population, float(sigma0)
            self.elites = elites
            self.generation = 0
            self.best_params = np.zeros(dim, dtype=np.float32)
            self.best_fitness = -math.inf
            self._rng = np.random.default_rng(seed)
            if init is None:
                pop = self._rng.normal(0.0, sigma0, (population, dim))
            else:
                pop = np.broadcast_to(np.asarray(init), (population, dim))
            self._pop = np.array(pop, dtype=np.float32)

        def ask(self):
            return self._pop.copy()

        def tell(self, fitness):
            fitness = np.asarray(fitness, dtype=np.float64)
            order = np.argsort(-fitness, kind="stable")
            if fitness[order[0]] > self.best_fitness:
                self.best_fitness = float(fitness[order[0]])
                self.best_params = self._pop[order[0]].copy()
            parents = self._pop[order[: max(2, self.population // 2)]]
            n_child = self.population - self.elites
            pick = self._rng.integers(0, len(parents), n_child)
            noise = self._rng.normal(0.0, self.sigma, (n_child, self.dim))
            children = parents[pick] + noise.astype(np.float32)
            self._pop = np.vstack([self._pop[order[: self.elites]], children])
            self.generation += 1

        def state_dict(self):
            return {"pop": self._pop.copy(), "sigma": np.float64(self.sigma)}

        def load_state_dict(self, d):
            self._pop = np.array(d["pop"], dtype=np.float32)
            self.sigma = float(d["sigma"])

    def make_optimiser(name, dim, **kwargs):
        if name != "ga":
            raise ValueError(f"stand-in only has 'ga', got {name!r}")
        return GA(dim, **kwargs)

    mod.GA = GA
    mod.make_optimiser = make_optimiser
    return mod


def _standin_checkpoint() -> types.ModuleType:
    mod = types.ModuleType("f1sim.agents.checkpoint")

    def car_params_hash(params):
        blob = json.dumps(asdict(params), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def current_git_commit():
        return "standin"

    def save_checkpoint(path, params, meta, optimiser_state=None):
        missing = META_KEYS - set(meta)
        if missing:
            raise ValueError(f"missing meta keys: {sorted(missing)}")
        path = Path(path)
        arrays = {"params": np.asarray(params, dtype=np.float32)}
        arrays.update({f"opt__{k}": np.asarray(v) for k, v in (optimiser_state or {}).items()})
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        path.write_bytes(buf.getvalue())
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    mod.car_params_hash = car_params_hash
    mod.current_git_commit = current_git_commit
    mod.save_checkpoint = save_checkpoint
    return mod


_STANDINS = {
    "f1sim.agents.policy": _standin_policy,
    "f1sim.agents.optimisers": _standin_optimisers,
    "f1sim.agents.checkpoint": _standin_checkpoint,
}


@pytest.fixture
def agents(monkeypatch):
    """The policy module the trainer will use (real if present, else the stand-in)."""
    force = os.environ.get("F1SIM_TRAINER_STANDINS", "").strip() not in ("", "0")
    for name, make in _STANDINS.items():
        if not force:
            try:
                importlib.import_module(name)
                continue
            except ModuleNotFoundError as exc:
                if exc.name != name:
                    raise
        monkeypatch.setitem(sys.modules, name, make())
    return importlib.import_module("f1sim.agents.policy")


# ------------------------------------------------------------------------------ helpers


@pytest.fixture(scope="module")
def track():
    x, y = stadium(STRAIGHT, RADIUS)
    w = np.full(x.size, HALF_WIDTH)
    return Track(build_track(x, y, w, w.copy(), track_id="oval", name="Oval"))


def controller_genome(policy, hidden=HIDDEN, target_v=30.0) -> np.ndarray:
    """A hand-made line holder in the contract's flat layout: W1, b1, W2, b2, Wout, bout.

    Hidden unit 0 carries the steering law ``-3 heading - 0.6 lateral + 0.24 curvature`` and
    unit 1 the speed law ``(target_v - v) / 5`` (see tests/test_world.py); small-signal gain 1.
    """
    spec = policy.PolicySpec(hidden=hidden)
    h1, h2 = hidden
    a = 0.5
    w1 = np.zeros((OBS_SIZE, h1))
    b1 = np.zeros(h1)
    w1[10, 0], w1[11, 0], w1[12, 0] = -3.0 * a, -0.6 * a, 0.24 * a
    w1[9, 1], b1[1] = -92.0 / 5.0 * a, target_v / 5.0 * a
    w2 = np.zeros((h1, h2))
    w2[0, 0] = w2[1, 1] = 1.0
    wo = np.zeros((h2, 2))
    wo[0, 0] = wo[1, 1] = 2.0
    flat = np.concatenate([w1.ravel(), b1, w2.ravel(), np.zeros(h2), wo.ravel(), np.zeros(2)])
    assert flat.size == spec.n_params
    return flat.astype(np.float32)


def small_config(**kw) -> TrainConfig:
    base = {
        "track_id": "oval",
        "seed": 7,
        "population": 8,
        "episodes_per_genome": 2,
        "hidden": HIDDEN,
        "max_generations": 3,
        "checkpoint_every": 2,
        "stage1_episode_s": 3.0,
        "stage1_promote_progress_m": 1e6,  # stay in stage 1 unless a test lowers it
        "stage2_lead_in_m": 100.0,  # on the bottom straight, so the car can brake to 30 m/s
        "init": "random",  # the curriculum tests; test_clone_init covers "clone"
        "sigma0": 0.05,  # the train_scratch.toml steps
        "sigma_max": 0.15,
        "sigma_min": 0.003,
    }
    base.update(kw)
    return TrainConfig(**base)


def read_metrics(run_dir: Path) -> tuple[list[str], list[dict]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def promotion_run(track, policy, run_dir: Path):
    cfg = small_config(stage1_promote_progress_m=20.0, sigma0=0.01)
    init = np.tile(controller_genome(policy), (cfg.population, 1))
    trainer = Trainer(track, cfg, run_dir, reference_lap_s=T_REF, init=init)
    return trainer, trainer.run()


# -------------------------------------------------------------------------------- tests


def test_run_folder_files_and_columns(track, agents, tmp_path):
    rows_seen = []
    cfg = small_config()
    trainer = Trainer(track, cfg, tmp_path / "run", reference_lap_s=T_REF)
    result = trainer.run(on_generation=rows_seen.append)

    run = tmp_path / "run"
    assert result.run_dir == run and result.generations == 3 and result.stage == 1
    assert result.best_lap_s is None and result.first_lap_wall_s is None
    header, rows = read_metrics(run)
    assert tuple(header) == METRICS_COLUMNS
    assert [int(r["gen"]) for r in rows] == [0, 1, 2]
    assert all(r["stage"] == "1" for r in rows)
    assert all(r["best_lap_s"] == "" for r in rows)
    assert [r["gen"] for r in rows_seen] == [0, 1, 2]
    assert all(tuple(r) == METRICS_COLUMNS for r in rows_seen)
    walls = [float(r["wall_s"]) for r in rows]
    assert walls == sorted(walls)
    for r in rows:
        assert float(r["best_fitness"]) >= float(r["mean_fitness"])
        assert 0.0 <= float(r["completion_rate"]) <= 1.0
        assert float(r["sigma"]) > 0.0

    conf = tomllib.loads((run / "config.toml").read_text(encoding="utf-8"))
    assert conf["seed"] == 7 and conf["hidden"] == list(HIDDEN)
    assert conf["resolved"]["t_ref_s"] == T_REF and conf["resolved"]["t_ref_source"] == "override"
    assert conf["car"] == asdict(CarParams())
    assert conf["run"]["obs_spec_version"] == OBS_SPEC_VERSION
    assert load_train_config(run / "config.toml") == cfg

    # gen 0 (first best, and gen % 2 == 0) and gen 2; the last generation is always saved
    ckpts = sorted(p.name for p in (run / "checkpoints").glob("*.npz"))
    assert "00000.npz" in ckpts and "00002.npz" in ckpts
    meta = json.loads((run / "checkpoints" / "00002.json").read_text(encoding="utf-8"))
    assert META_KEYS <= set(meta)
    assert meta["generation"] == 2 and meta["track_id"] == "oval" and meta["seed"] == 7
    with np.load(run / "checkpoints" / "00002.npz", allow_pickle=False) as z:
        assert z["params"].shape == (trainer.spec.n_params,)
    assert not (run / "best_lap.npz").exists()


def test_same_seed_reproduces_metrics(track, agents, tmp_path):
    def run(name, seed):
        Trainer(track, small_config(seed=seed), tmp_path / name, reference_lap_s=T_REF).run()
        _, rows = read_metrics(tmp_path / name)
        return [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]

    a, b, c = run("a", 3), run("b", 3), run("c", 4)
    assert a == b
    assert a != c


def test_promotion_and_best_lap(track, agents, tmp_path):
    trainer, result = promotion_run(track, agents, tmp_path / "run")
    run = tmp_path / "run"
    _, rows = read_metrics(run)
    stages = [int(r["stage"]) for r in rows]
    assert stages[0] == 1 and stages[1:] == [2] * (len(rows) - 1) and result.stage == 2

    # Stage 2: the line holder laps the oval at ~30 m/s
    assert float(rows[-1]["completion_rate"]) > 0.0
    assert result.best_lap_s is not None
    assert result.best_lap_s == pytest.approx(track.length / 30.0, rel=0.1)
    assert result.first_lap_wall_s is not None and result.first_lap_wall_s > 0.0
    cap = trainer.config.stage2_time_factor * T_REF
    assert float(rows[-1]["best_fitness"]) > track.length + 1000.0
    assert float(rows[-1]["best_fitness"]) <= track.length + 1000.0 + 50.0 * cap

    with np.load(run / "best_lap.npz", allow_pickle=False) as z:
        assert set(z.files) == set(TRACE_COLUMNS)
        trace = {k: z[k] for k in z.files}
    n = trace["t"].size
    assert n > 100 and all(trace[c].shape == (n,) for c in TRACE_COLUMNS)
    assert trace["t"][0] == 0.0 and np.all(np.diff(trace["t"]) > 0.0)
    assert trace["t"][-1] == pytest.approx(result.best_lap_s, abs=1e-3)
    assert np.all(np.diff(trace["t"])[1:-1] == pytest.approx(1.0 / 60.0))  # every physics step
    assert trace["s"][0] == 0.0 and np.all(np.diff(trace["s"]) >= 0.0)
    assert trace["s"][-1] == pytest.approx(track.length, rel=0.03)
    assert np.all((trace["throttle"] >= 0.0) & (trace["throttle"] <= 1.0))
    assert set(np.unique(trace["brake"])) <= {0.0, 1.0}
    assert np.all(trace["gear"] == 0)
    assert np.all((trace["speed"] > 20.0) & (trace["speed"] < 70.0))
    _, nn, _ = track.project(trace["x"], trace["y"])
    assert np.all(np.abs(nn) < HALF_WIDTH)


def test_clone_init_starts_in_stage_2_with_laps(track, agents, tmp_path):
    """init = "clone": the scripted driver is copied before generation 0, which already
    laps the oval; the same seed gives the same metrics."""

    def run(name):
        cfg = small_config(
            init="clone", clone_rounds=3, sigma0=0.005, sigma_max=0.05, sigma_min=0.0005
        )
        trainer = Trainer(track, cfg, tmp_path / name, reference_lap_s=T_REF)
        result = trainer.run()
        _, rows = read_metrics(tmp_path / name)
        return trainer, result, rows

    trainer, result, rows = run("a")
    assert [int(r["stage"]) for r in rows] == [2] * len(rows) and result.stage == 2
    assert float(rows[0]["completion_rate"]) > 0.5
    assert result.best_lap_s is not None and result.first_lap_wall_s is not None
    assert trainer.clone_wall_s is not None and trainer.clone_wall_s > 0.0
    conf = tomllib.loads((tmp_path / "a" / "config.toml").read_text(encoding="utf-8"))
    assert conf["init"] == "clone" and conf["resolved"]["clone_wall_s"] > 0.0
    strip = [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    _, _, rows_b = run("b")
    assert strip == [{k: v for k, v in r.items() if k != "wall_s"} for r in rows_b]


def test_promotion_run_is_reproducible(track, agents, tmp_path):
    promotion_run(track, agents, tmp_path / "a")
    promotion_run(track, agents, tmp_path / "b")

    def strip(run):
        _, rows = read_metrics(run)
        return [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]

    assert strip(tmp_path / "a") == strip(tmp_path / "b")
    a = np.load(tmp_path / "a" / "best_lap.npz", allow_pickle=False)
    b = np.load(tmp_path / "b" / "best_lap.npz", allow_pickle=False)
    for c in TRACE_COLUMNS:
        np.testing.assert_array_equal(a[c], b[c])
    a.close()
    b.close()


def test_stage1_crash_penalty(track, agents, tmp_path):
    cfg = small_config()
    trainer = Trainer(track, cfg, tmp_path / "run", reference_lap_s=T_REF)
    # Zero weights, output bias (the last two parameters) = full left steer, half throttle.
    genome = np.zeros(trainer.spec.n_params, dtype=np.float32)
    genome[-2:] = [5.0, 0.55]
    fit, stats = trainer.evaluate(np.tile(genome, (cfg.population, 1)), stage=1)
    assert fit.shape == (cfg.population,)
    assert stats["crash_rate"] == 1.0
    assert np.all(stats["progress_m"] > 0.0)
    np.testing.assert_allclose(fit, 0.9 * stats["progress_m"], rtol=1e-12)
    assert stats["sim_s"] < cfg.stage1_episode_s  # stopped early: every car crashed


def test_stage2_lap_fitness(track, agents, tmp_path):
    cfg = small_config()
    trainer = Trainer(track, cfg, tmp_path / "run", reference_lap_s=T_REF)
    params = np.tile(controller_genome(agents), (cfg.population, 1))
    fit, stats = trainer.evaluate(params, stage=2)
    assert stats["completion_rate"] == 1.0
    cap = cfg.stage2_time_factor * T_REF
    expected = track.length + 1000.0 + 50.0 * (cap - stats["lap_time_s"])
    assert np.all(stats["steer_change"] > 0.0)
    expected -= cfg.steer_change_weight * stats["steer_change"]
    np.testing.assert_allclose(fit, expected, rtol=1e-12)
    assert stats["sim_s"] < cap  # stopped once every car finished its lap
    assert trainer.stage2_start_s == pytest.approx(track.length - 100.0)
    assert 20.0 < trainer.stage2_start_v <= 60.0


def test_steer_change_weight_only_shifts_fitness(track, agents, tmp_path):
    params = np.tile(controller_genome(agents), (small_config().population, 1))
    fits = {}
    for w in (0.0, 50.0):
        cfg = small_config(steer_change_weight=w)
        trainer = Trainer(track, cfg, tmp_path / f"run{w:g}", reference_lap_s=T_REF)
        fits[w] = trainer.evaluate(params, stage=2)
    (f0, s0), (f50, s50) = fits[0.0], fits[50.0]
    np.testing.assert_array_equal(s0["lap_time_s"], s50["lap_time_s"])
    np.testing.assert_allclose(f0 - f50, 50.0 * s0["steer_change"], rtol=1e-12)


def test_reference_lap_falls_back_to_the_centre_line(track, agents, tmp_path):
    trainer = Trainer(track, small_config(), tmp_path / "run")  # "oval" isn't in the catalogue
    assert trainer.t_ref_source == "centre-line"
    assert 0.5 * track.length / 92.0 < trainer.t_ref_s < track.length / 20.0


def test_wall_time_limit_stops_after_one_generation(track, agents, tmp_path):
    cfg = small_config(max_wall_s=0.0, max_generations=50)
    result = Trainer(track, cfg, tmp_path / "run", reference_lap_s=T_REF).run()
    assert result.generations == 1
    _, rows = read_metrics(tmp_path / "run")
    assert len(rows) == 1


def test_default_config_file_matches_the_dataclass():
    assert load_train_config(DEFAULT_CONFIG) == TrainConfig()
    assert TrainConfig().promote_progress_m == pytest.approx(0.8 * 30.0 * 40.0)


def test_load_train_config_rejects_bad_input(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('track_id = "monza"\nhidden = [32, 16]\nsigma0 = 1\n', encoding="utf-8")
    cfg = load_train_config(p)
    assert cfg.track_id == "monza" and cfg.hidden == (32, 16) and cfg.sigma0 == 1.0
    assert isinstance(cfg.sigma0, float)

    for text in (
        "populaton = 10\n",  # typo
        "population = 1.5\n",
        "population = true\n",
        'hidden = ["64"]\n',
        "seed = '0'\n",
        "[extra]\nx = 1\n",
    ):
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_train_config(p)


def test_toml_writer_round_trips():
    data = {
        "a": 1,
        "b": 0.1,
        "c": 'x "quoted" \\ é',
        "d": [1, 2],
        "e": True,
        "f": 1e-7,
        "skip": None,
        "t": {"g": 2.5, "h": [0.5, 1.5], "nested": {"i": -3}},
    }
    back = tomllib.loads(to_toml(data))
    expected = {k: v for k, v in data.items() if k != "skip"}
    assert back == expected
    assert math.isnan(tomllib.loads(to_toml({"n": float("nan")}))["n"])
