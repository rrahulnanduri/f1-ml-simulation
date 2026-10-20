"""Agent checkpoints (docs/data-formats.md section 4, docs/m3-interfaces.md section I).

A checkpoint is two files side by side:

- ``<gen>.npz``: ``params`` (flat float32 policy vector) plus one ``opt__<key>`` array per entry
  of the optimiser's ``state_dict()``; scalars are stored as 0-d arrays and come back as Python
  scalars. Loaded with ``allow_pickle=False``.
- ``<gen>.json``: the metadata (:data:`REQUIRED_META_KEYS`).

Both files are written atomically (temp file in the same folder, then ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

import numpy as np

from f1sim.agents.policy import PolicySpec
from f1sim.sim.car_model import CarParams
from f1sim.sim.sensors import OBS_SPEC_VERSION

REQUIRED_META_KEYS = (
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
)
"""Keys every checkpoint's meta must hold; ``obs_spec_version`` and ``policy`` are auto-filled."""

OPT_PREFIX = "opt__"
PARAMS_KEY = "params"


class CheckpointMismatch(ValueError):
    """The checkpoint doesn't fit the current code (observation spec or policy shape)."""


@dataclass(frozen=True)
class Checkpoint:
    params: np.ndarray  # (n_params,) float32
    meta: dict
    optimiser_state: dict[str, np.ndarray]


def car_params_hash(params: CarParams) -> str:
    """First 12 hex chars of sha1 over the sorted ``asdict`` JSON; stable across processes."""
    text = json.dumps(asdict(params), sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def current_git_commit() -> str:
    """SHA of the f1sim source checkout's ``HEAD``, or ``""`` if it can't be determined."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # any failure (no git, not a checkout, timeout) means "unknown"
        return ""
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        return ""
    return sha


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _atomic_write(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Write ``path`` via a temp file in the same folder and ``os.replace``."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _resolve_meta(meta: dict, n_params: int) -> dict:
    meta = dict(meta)
    meta.setdefault("obs_spec_version", OBS_SPEC_VERSION)
    policy = meta.get("policy")
    if policy is None:
        policy = PolicySpec()
    if isinstance(policy, PolicySpec):
        policy = policy.to_dict()
    try:
        spec = PolicySpec.from_dict(policy)
    except (TypeError, ValueError) as e:
        raise ValueError(f"meta['policy'] is not a valid PolicySpec dict: {e}") from e
    meta["policy"] = spec.to_dict()
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"checkpoint meta is missing required key(s): {', '.join(missing)}")
    if spec.n_params != n_params:
        raise ValueError(
            f"params has {n_params} entries but meta['policy'] {spec.layer_sizes} needs "
            f"{spec.n_params}; set meta['policy'] = spec.to_dict() for the policy you trained"
        )
    return meta


def _opt_arrays(optimiser_state: dict | None) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in (optimiser_state or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"optimiser_state keys must be non-empty strings, got {key!r}")
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError(
                f"optimiser_state[{key!r}] must be a numpy array or scalar, "
                f"got {type(value).__name__}"
            )
        arrays[OPT_PREFIX + key] = arr
    return arrays


def save_checkpoint(path: Path, params, meta: dict, optimiser_state: dict | None = None) -> None:
    """Write ``path`` (``<gen>.npz``) and its sidecar ``path.with_suffix(".json")`` atomically.

    ``params`` is the flat ``(n_params,)`` policy vector (stored as float32). ``meta`` must hold
    :data:`REQUIRED_META_KEYS` (``ValueError`` otherwise); ``obs_spec_version`` defaults to the
    current :data:`~f1sim.sim.sensors.OBS_SPEC_VERSION` and ``policy`` to ``PolicySpec()``.
    ``policy`` may be a :class:`PolicySpec` or its ``to_dict()``. ``optimiser_state`` maps names
    to numpy arrays or scalars (``Optimiser.state_dict()``); ``git_commit`` may be ``""``.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"checkpoint path must end in .npz, got {path}")
    flat = np.asarray(params, dtype=np.float32)
    if flat.ndim != 1:
        raise ValueError(f"params must be a flat (n_params,) vector, got shape {flat.shape}")
    meta = _resolve_meta(meta, flat.shape[0])
    arrays = {PARAMS_KEY: flat, **_opt_arrays(optimiser_state)}
    text = json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda fh: np.savez(fh, **arrays))
    _atomic_write(path.with_suffix(".json"), lambda fh: fh.write(text.encode("utf-8")))


def _restore(arr: np.ndarray):
    return arr.item() if arr.ndim == 0 else arr


def load_checkpoint(path: Path, *, spec: PolicySpec | None = None) -> Checkpoint:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Raises :class:`CheckpointMismatch` when the checkpoint's ``obs_spec_version`` differs from
    the current one, or when ``spec`` is given and differs from ``meta["policy"]``; the message
    says what differs. Optimiser scalars come back as Python scalars, arrays as arrays.
    """
    path = Path(path)
    meta_path = path.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{meta_path}: not valid JSON ({e})") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: expected a JSON object, got {type(meta).__name__}")

    version = meta.get("obs_spec_version")
    if version != OBS_SPEC_VERSION:
        raise CheckpointMismatch(
            f"{path}: saved with observation spec v{version}, but this code uses "
            f"v{OBS_SPEC_VERSION} (sim/sensors.py); the policy's inputs would be wrong. "
            "Retrain, or use the f1sim version that wrote it"
            + (f" (git {meta['git_commit']})" if meta.get("git_commit") else "")
        )
    if not isinstance(meta.get("policy"), dict):
        raise CheckpointMismatch(f"{meta_path}: no 'policy' description in the metadata")
    try:
        saved = PolicySpec.from_dict(meta["policy"])
    except (TypeError, ValueError) as e:
        raise CheckpointMismatch(
            f"{path}: meta['policy'] is not understood by this code: {e}"
        ) from e
    if spec is not None and saved.to_dict() != spec.to_dict():
        diffs = [
            f"{key}: checkpoint {saved_v!r}, expected {want_v!r}"
            for key, want_v in spec.to_dict().items()
            if (saved_v := saved.to_dict().get(key)) != want_v
        ]
        raise CheckpointMismatch(
            f"{path}: policy doesn't match the requested spec ({'; '.join(diffs)})"
        )

    with np.load(path, allow_pickle=False) as data:
        if PARAMS_KEY not in data.files:
            raise ValueError(f"{path}: no 'params' array in the checkpoint")
        params = np.asarray(data[PARAMS_KEY], dtype=np.float32)
        opt = {
            name[len(OPT_PREFIX) :]: _restore(data[name])
            for name in data.files
            if name.startswith(OPT_PREFIX)
        }
    if params.shape != (saved.n_params,):
        raise ValueError(
            f"{path}: params has shape {params.shape}, but meta['policy'] "
            f"{saved.layer_sizes} needs ({saved.n_params},)"
        )
    meta["policy"] = saved.to_dict()
    return Checkpoint(params=params, meta=meta, optimiser_state=opt)
