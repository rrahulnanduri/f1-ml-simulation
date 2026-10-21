import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from f1sim.agents import checkpoint as ckpt_mod
from f1sim.agents.checkpoint import (
    REQUIRED_META_KEYS,
    CheckpointMismatch,
    car_params_hash,
    current_git_commit,
    load_checkpoint,
    save_checkpoint,
)
from f1sim.agents.policy import PolicySpec, init_params
from f1sim.sim.car_model import CarParams
from f1sim.sim.sensors import OBS_SPEC_VERSION

SRC = Path(__file__).resolve().parents[1] / "src"


def _meta(**overrides) -> dict:
    meta = {
        "car_params_hash": car_params_hash(CarParams()),
        "track_id": "silverstone",
        "seed": 0,
        "generation": 10,
        "fitness": 1234.5,
        "best_lap_s": None,
        "git_commit": "",
        "created": "2026-10-20T12:00:00+00:00",
    }
    meta.update(overrides)
    return meta


def _params(spec: PolicySpec | None = None) -> np.ndarray:
    return init_params(spec or PolicySpec(), 1, np.random.default_rng(0))[0]


def test_round_trip_with_optimiser_state(tmp_path):
    spec = PolicySpec(hidden=(16, 8))
    params = _params(spec)
    state = {
        "mean": np.linspace(0, 1, spec.n_params),
        "population": np.arange(12, dtype=np.float32).reshape(3, 4),
        "sigma": 0.25,
        "generation": 10,
        "rng_state": np.array([1, 2, 3], dtype=np.uint64),
        "flag": True,
        "best_fitness": np.float64(-3.5),
    }
    path = tmp_path / "checkpoints" / "00010.npz"
    save_checkpoint(path, params, _meta(policy=spec, best_lap_s=91.25), state)

    assert path.exists() and path.with_suffix(".json").exists()
    ck = load_checkpoint(path, spec=spec)
    assert ck.params.dtype == np.float32
    np.testing.assert_array_equal(ck.params, params)
    assert ck.meta["policy"] == spec.to_dict()
    assert ck.meta["obs_spec_version"] == OBS_SPEC_VERSION
    assert ck.meta["best_lap_s"] == 91.25
    assert ck.meta["generation"] == 10
    assert set(ck.optimiser_state) == set(state)
    np.testing.assert_array_equal(ck.optimiser_state["mean"], state["mean"])
    assert ck.optimiser_state["population"].dtype == np.float32
    np.testing.assert_array_equal(ck.optimiser_state["population"], state["population"])
    assert ck.optimiser_state["rng_state"].dtype == np.uint64
    assert ck.optimiser_state["sigma"] == 0.25 and isinstance(ck.optimiser_state["sigma"], float)
    assert ck.optimiser_state["generation"] == 10
    assert isinstance(ck.optimiser_state["generation"], int)
    assert ck.optimiser_state["flag"] is True
    assert ck.optimiser_state["best_fitness"] == -3.5

    with np.load(path, allow_pickle=False) as data:
        assert sorted(data.files) == sorted(["params"] + [f"opt__{k}" for k in state])
        assert data["opt__sigma"].ndim == 0


def test_auto_filled_meta_and_numpy_values(tmp_path):
    path = tmp_path / "00001.npz"
    meta = _meta(seed=np.int64(3), fitness=np.float32(2.5))
    save_checkpoint(path, _params(), meta)
    doc = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert doc["obs_spec_version"] == OBS_SPEC_VERSION
    assert doc["policy"] == PolicySpec().to_dict()
    assert doc["seed"] == 3 and doc["fitness"] == 2.5
    assert set(REQUIRED_META_KEYS) <= set(doc)
    ck = load_checkpoint(path)
    assert ck.optimiser_state == {}
    assert "obs_spec_version" not in meta  # caller's dict is untouched


def test_overwrite_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "00002.npz"
    save_checkpoint(path, _params(), _meta(generation=1), {"sigma": 0.1})
    new = _params() + 1.0
    save_checkpoint(path, new, _meta(generation=2), {"sigma": 0.2})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00002.json", "00002.npz"]
    ck = load_checkpoint(path)
    np.testing.assert_array_equal(ck.params, new)
    assert ck.meta["generation"] == 2 and ck.optimiser_state["sigma"] == 0.2


def test_failed_write_keeps_old_file_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "00003.npz"
    save_checkpoint(path, _params(), _meta(generation=1))
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt_mod.np, "savez", boom)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, _params() + 1.0, _meta(generation=2))
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00003.json", "00003.npz"]


@pytest.mark.parametrize(
    "key", [k for k in REQUIRED_META_KEYS if k not in ("obs_spec_version", "policy")]
)
def test_missing_meta_key_raises(tmp_path, key):
    meta = _meta()
    del meta[key]
    path = tmp_path / "00004.npz"
    with pytest.raises(ValueError, match=key):
        save_checkpoint(path, _params(), meta)
    assert list(tmp_path.iterdir()) == []


def test_save_validation_errors(tmp_path):
    with pytest.raises(ValueError, match=r"needs"):
        save_checkpoint(tmp_path / "a.npz", np.zeros(10, np.float32), _meta())
    with pytest.raises(ValueError, match=r"flat"):
        save_checkpoint(tmp_path / "a.npz", np.zeros((2, PolicySpec().n_params)), _meta())
    with pytest.raises(ValueError, match=r"\.npz"):
        save_checkpoint(tmp_path / "a.bin", _params(), _meta())
    with pytest.raises(ValueError, match=r"optimiser_state\['bad'\]"):
        save_checkpoint(tmp_path / "a.npz", _params(), _meta(), {"bad": None})
    assert list(tmp_path.iterdir()) == []


def test_obs_spec_version_mismatch(tmp_path):
    path = tmp_path / "00005.npz"
    save_checkpoint(path, _params(), _meta(obs_spec_version=OBS_SPEC_VERSION + 1, git_commit="abc"))
    with pytest.raises(
        CheckpointMismatch,
        match=rf"observation spec v{OBS_SPEC_VERSION + 1}.*uses v{OBS_SPEC_VERSION}",
    ) as exc:
        load_checkpoint(path)
    assert "abc" in str(exc.value)
    assert isinstance(exc.value, ValueError)


def test_policy_spec_mismatch(tmp_path):
    path = tmp_path / "00006.npz"
    saved = PolicySpec(hidden=(32, 32))
    save_checkpoint(path, _params(saved), _meta(policy=saved.to_dict()))
    assert load_checkpoint(path).meta["policy"]["hidden"] == [32, 32]  # no spec: no check
    with pytest.raises(
        CheckpointMismatch, match=r"hidden: checkpoint \[32, 32\], expected \[64, 64\]"
    ):
        load_checkpoint(path, spec=PolicySpec())
    with pytest.raises(CheckpointMismatch, match=r"activation: checkpoint 'tanh', expected 'relu'"):
        load_checkpoint(path, spec=PolicySpec(hidden=(32, 32), activation="relu"))


def test_corrupt_policy_in_meta(tmp_path):
    path = tmp_path / "00007.npz"
    save_checkpoint(path, _params(), _meta())
    meta_path = path.with_suffix(".json")
    doc = json.loads(meta_path.read_text(encoding="utf-8"))
    doc["policy"]["layers"] = "lstm"
    meta_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CheckpointMismatch, match="policy"):
        load_checkpoint(path)


def test_car_params_hash_definition_and_sensitivity():
    p = CarParams()
    text = json.dumps(asdict(p), sort_keys=True)
    assert car_params_hash(p) == hashlib.sha1(text.encode()).hexdigest()[:12]
    assert len(car_params_hash(p)) == 12
    assert car_params_hash(p) == car_params_hash(CarParams())
    assert car_params_hash(replace(p, max_brake_g=p.max_brake_g + 0.1)) != car_params_hash(p)


def test_car_params_hash_stable_across_processes():
    code = (
        "from f1sim.agents.checkpoint import car_params_hash;"
        "from f1sim.sim.car_model import CarParams;"
        "print(car_params_hash(CarParams()))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONHASHSEED": "123"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == car_params_hash(CarParams())


def test_current_git_commit_never_raises(monkeypatch):
    sha = current_git_commit()
    assert sha == "" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ckpt_mod.subprocess, "run", missing)
    assert current_git_commit() == ""
