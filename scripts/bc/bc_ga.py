"""GA fine-tuning of the raceline-aware clone (experiment D16).

Usage: python bc_ga.py <clone.npz> <run_dir> <minutes> <sigma0>

Patches World observations (+ raceline features) and the policy spec for this process only.
"""

import sys
from pathlib import Path

import bc_common as bcc
import numpy as np

from f1sim.agents import policy as pol
from f1sim.data.track_pipeline import load_track
from f1sim.sim import world as world_mod
from f1sim.train.trainer import TrainConfig, Trainer

clone_path, run_dir, minutes, sigma0 = (
    sys.argv[1],
    sys.argv[2],
    float(sys.argv[3]),
    float(sys.argv[4]),
)
clone = np.load(clone_path)["params"].astype(np.float32)
tch = bcc.teacher()
feat = bcc.raceline_features(tch)

orig_observe = world_mod.World._observe
world_mod.World._observe = lambda self: feat(self, orig_observe(self))
_Spec = pol.PolicySpec


def _spec(*a, **k):
    k.setdefault("obs_size", 19 + bcc.N_EXTRA)
    return _Spec(*a, **k)


class _Mod:  # the trainer reads PolicySpec from the module it imported
    pass


import importlib  # noqa: E402
import types  # noqa: E402

import f1sim.train.trainer as trainer_mod  # noqa: E402


def _import(name, package=None):
    m = importlib.import_module(name, package)
    if name == "f1sim.agents.policy":
        shim = _Mod()
        shim.__dict__.update(vars(m))
        shim.PolicySpec = _spec
        return shim
    return m


trainer_mod.importlib = types.SimpleNamespace(import_module=_import)

P = 64
rng = np.random.default_rng(0)
init = clone[None, :] + rng.normal(0.0, sigma0, (P, clone.size)).astype(np.float32)
init[0] = clone
cfg = TrainConfig(
    population=P,
    hidden=(16, 16),
    sigma0=sigma0,
    sigma_max=0.05,
    sigma_min=0.0005,
    stage1_promote_progress_m=1.0,  # straight to timed laps
    max_wall_s=minutes * 60.0,
    checkpoint_every=0,
)
tr = Trainer(load_track("silverstone"), cfg, Path(run_dir), init=init)


def report(row):
    if row["gen"] % 10 == 0:
        print(
            f"gen {row['gen']:4d} stage {row['stage']} {row['wall_s']:6.0f}s "
            f"best lap {row['best_lap_s']:.3f} laps {row['completion_rate']:.0%} "
            f"sigma {row['sigma']:.4f}",
            flush=True,
        )


res = tr.run(on_generation=report)
print("RESULT best lap", res.best_lap_s, "T_ref", tr.t_ref_s)
