"""Run-folder writer for training (docs/data-formats.md section 5, docs/m3-interfaces.md K).

A run folder holds:

- ``config.toml``: the fully resolved config, written once with :func:`to_toml` (``tomllib``
  only reads TOML, so this module has a tiny writer for the value types a config uses).
- ``metrics.csv``: one row per generation, columns :data:`METRICS_COLUMNS`. Floats are written
  with ``repr`` (shortest round-trip form), so identical runs give identical files apart from
  ``wall_s``; a missing value (no lap this generation) is an empty field.
- ``checkpoints/<gen:05d>.npz`` (+ ``.json``), through ``f1sim.agents.checkpoint``.
- ``best_lap.npz``: the best lap's trace, columns :data:`TRACE_COLUMNS` (the ghost columns,
  data-formats section 3), written atomically.
- ``generation.npz``: the latest generation, :data:`GENERATION_KEYS`, replaced atomically
  every generation for the app's training view (:func:`read_generation`).
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METRICS_COLUMNS = (
    "gen",
    "stage",
    "wall_s",
    "best_fitness",
    "mean_fitness",
    "completion_rate",
    "best_lap_s",
    "best_progress_m",
    "sigma",
)
TRACE_COLUMNS = ("t", "x", "y", "s", "speed", "throttle", "brake", "gear")
CHECKPOINT_DIR = "checkpoints"
GENERATION_FILE = "generation.npz"
GENERATION_KEYS = (
    "gen",  # () int
    "stage",  # () int
    "params",  # (P, n_params) float32, one genome per row
    "fitness",  # (P,)
    "start_s",  # (P,) where each genome's (first) episode started, m along the centre line
    "start_v",  # (P,) its start speed, m/s
    "episode_s",  # () the episode's time cap, s
    "lap_time_s",  # (P,) nan where no lap
    "progress_m",  # (P,)
)


# ----------------------------------------------------------------------------- TOML writer


def _toml_key(key: str) -> str:
    if key and all(c.isalnum() or c in "_-" for c in key):
        return key
    return json.dumps(key)


def _toml_value(value) -> str:
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple | np.ndarray):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if value is None:
        raise TypeError("TOML has no null; leave the key out instead")
    raise TypeError(f"can't write {type(value).__name__} to TOML")


def to_toml(data: Mapping) -> str:
    """Serialise a dict of scalars, lists and (one level or more of) sub-dicts to TOML.

    Plain keys come first, then each sub-dict as a ``[table]``. Keys whose value is ``None``
    are skipped (TOML has no null).
    """
    out = io.StringIO()

    def emit(table: Mapping, prefix: tuple[str, ...]) -> None:
        subs = []
        for key, value in table.items():
            if value is None:
                continue
            if isinstance(value, Mapping):
                subs.append((key, value))
            else:
                out.write(f"{_toml_key(str(key))} = {_toml_value(value)}\n")
        for key, value in subs:
            path = (*prefix, str(key))
            out.write(f"\n[{'.'.join(_toml_key(k) for k in path)}]\n")
            emit(value, path)

    emit(data, ())
    return out.getvalue()


# ------------------------------------------------------------------------- atomic writes


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return str(int(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    v = float(value)
    return "" if math.isnan(v) else repr(v)


# --------------------------------------------------------------------------------- writer


class RunWriter:
    """Writes one run folder. Creating it truncates any earlier ``metrics.csv`` there."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.config_path = self.run_dir / "config.toml"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.best_lap_path = self.run_dir / "best_lap.npz"
        self.generation_path = self.run_dir / GENERATION_FILE
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow(METRICS_COLUMNS)

    def write_config(self, config: Mapping) -> Path:
        """Write the resolved config dict as ``config.toml``."""
        _atomic_write_bytes(self.config_path, to_toml(config).encode("utf-8"))
        return self.config_path

    def append_metrics(self, row: Mapping) -> None:
        """Append one generation; ``row`` must have every key of :data:`METRICS_COLUMNS`."""
        missing = [c for c in METRICS_COLUMNS if c not in row]
        if missing:
            raise KeyError(f"metrics row is missing {', '.join(missing)}")
        values = [
            f"{float(row[c]):.3f}" if c == "wall_s" else _fmt(row[c]) for c in METRICS_COLUMNS
        ]
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow(values)

    def checkpoint_path(self, gen: int) -> Path:
        return self.checkpoint_dir / f"{int(gen):05d}.npz"

    def save_checkpoint(
        self, gen: int, params: np.ndarray, meta: dict, optimiser_state: dict | None = None
    ) -> Path:
        """``checkpoints/<gen:05d>.npz`` + ``.json`` via ``f1sim.agents.checkpoint``."""
        checkpoint = importlib.import_module("f1sim.agents.checkpoint")
        path = self.checkpoint_path(gen)
        checkpoint.save_checkpoint(
            path, np.asarray(params, dtype=np.float32), meta, optimiser_state
        )
        return path

    def write_best_lap(self, trace: Mapping[str, np.ndarray]) -> Path:
        """Write ``best_lap.npz`` with exactly the :data:`TRACE_COLUMNS` arrays."""
        missing = [c for c in TRACE_COLUMNS if c not in trace]
        if missing:
            raise KeyError(f"best-lap trace is missing {', '.join(missing)}")
        n = {np.asarray(trace[c]).shape for c in TRACE_COLUMNS}
        if len(n) != 1:
            raise ValueError(f"best-lap trace columns differ in shape: {n}")
        buf = io.BytesIO()
        np.savez(buf, **{c: np.asarray(trace[c]) for c in TRACE_COLUMNS})
        _atomic_write_bytes(self.best_lap_path, buf.getvalue())
        return self.best_lap_path

    def write_generation(self, **values) -> Path:
        """Replace ``generation.npz`` with one generation: every key of
        :data:`GENERATION_KEYS`, nothing else."""
        keys = set(values)
        if keys != set(GENERATION_KEYS):
            raise KeyError(
                f"generation needs exactly {', '.join(GENERATION_KEYS)}; "
                f"got {', '.join(sorted(keys))}"
            )
        arrays = {k: np.asarray(values[k]) for k in GENERATION_KEYS}
        arrays["params"] = arrays["params"].astype(np.float32)
        p = arrays["params"].shape[0]
        for k in ("fitness", "start_s", "start_v", "lap_time_s", "progress_m"):
            if arrays[k].shape != (p,):
                raise ValueError(f"generation {k} must be ({p},), got {arrays[k].shape}")
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        _atomic_write_bytes(self.generation_path, buf.getvalue())
        return self.generation_path


def read_generation(path: Path) -> dict[str, np.ndarray] | None:
    """A ``generation.npz`` as a dict of arrays, or None if it's missing, unreadable or
    lacks a key (the writer replaces it atomically, but a reader may still lose a race with
    the rename on Windows)."""
    try:
        with np.load(Path(path)) as z:
            if not all(k in z.files for k in GENERATION_KEYS):
                return None
            return {k: z[k] for k in GENERATION_KEYS}
    except (OSError, ValueError, EOFError):
        return None
