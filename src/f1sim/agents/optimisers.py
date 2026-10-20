"""Black-box optimisers for neuroevolution: a GA (default) and diagonal CMA-ES.

Both follow the ask/tell :class:`Optimiser` protocol (docs/m3-interfaces.md, section J):

    opt = make_optimiser("ga", dim, population=64, seed=0, init=init_rows)
    while training:
        pop = opt.ask()              # (population, dim) float32
        fitness = evaluate(pop)      # (population,), higher is better
        opt.tell(fitness)

``ask`` is idempotent: it returns the same population until ``tell`` is called, and ``tell``
always refers to that population. ``state_dict`` holds only numpy arrays and numpy/Python
scalars, so the trainer can store it in the checkpoint NPZ as ``opt__<key>`` arrays and
read it back with ``allow_pickle=False``.

Fitness is noisy (random starts), so besides the best ever told (``best_params``,
``best_fitness``) both optimisers expose the last generation's best,
``generation_best_fitness``, and the per-generation bests, ``best_history``.
Non-finite fitness values (NaN, Â±inf) are treated as the worst possible result.
"""

from __future__ import annotations

import pickle
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["CMAES", "GA", "Optimiser", "make_optimiser"]

CMA_INSTALL_HINT = 'pip install -e ".[es]"'


@runtime_checkable
class Optimiser(Protocol):
    dim: int
    population: int
    generation: int  # tells so far
    sigma: float  # current step size, for logging
    best_params: np.ndarray  # (dim,) best ever told
    best_fitness: float

    def ask(self) -> np.ndarray: ...  # (population, dim) float32 to evaluate

    def tell(self, fitness: np.ndarray) -> None: ...  # (population,), higher = better

    def state_dict(self) -> dict: ...  # numpy arrays and scalars only, so it fits the NPZ

    def load_state_dict(self, d: dict) -> None: ...


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------

_U64 = (1 << 64) - 1


def _rng_state_to_array(rng: np.random.Generator) -> np.ndarray:
    """PCG64 state as a uint64 array (the 128-bit ints don't fit an NPZ scalar)."""
    st = rng.bit_generator.state
    if st["bit_generator"] != "PCG64":  # pragma: no cover - we always build PCG64
        raise ValueError(f"unsupported bit generator {st['bit_generator']}")
    s, inc = st["state"]["state"], st["state"]["inc"]
    return np.array(
        [s >> 64, s & _U64, inc >> 64, inc & _U64, st["has_uint32"], st["uinteger"]],
        dtype=np.uint64,
    )


def _rng_from_array(a: np.ndarray) -> np.random.Generator:
    v = [int(x) for x in np.asarray(a, dtype=np.uint64)]
    if len(v) != 6:
        raise ValueError(f"rng_state must have 6 entries, got {len(v)}")
    bg = np.random.PCG64()
    bg.state = {
        "bit_generator": "PCG64",
        "state": {"state": (v[0] << 64) | v[1], "inc": (v[2] << 64) | v[3]},
        "has_uint32": v[4],
        "uinteger": v[5],
    }
    return np.random.Generator(bg)


def _clean_fitness(fitness: Any, population: int) -> np.ndarray:
    f = np.asarray(fitness, dtype=np.float64)
    if f.shape != (population,):
        raise ValueError(f"fitness must have shape ({population},), got {f.shape}")
    return np.where(np.isfinite(f), f, -np.inf)


def _init_rows(init: Any, dim: int) -> np.ndarray:
    rows = np.asarray(init, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != dim or rows.shape[0] == 0:
        raise ValueError(
            f"init must have shape (dim,) or (k, dim) with dim={dim}, got {rows.shape}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("init contains non-finite values")
    return rows


def _check_kind(d: dict, kind: str, dim: int, population: int) -> None:
    got = str(np.asarray(d["kind"]))
    if got != kind:
        raise ValueError(f"optimiser state is for {got!r}, not {kind!r}")
    for key, want in (("dim", dim), ("population", population)):
        if int(d[key]) != want:
            raise ValueError(f"optimiser state has {key}={int(d[key])}, this one has {want}")


class _BestTracker:
    """Best-ever and per-generation-best bookkeeping shared by both optimisers."""

    best_params: np.ndarray
    best_fitness: float
    generation_best_fitness: float
    _history: list[float]

    def _reset_best(self, dim: int, first_row: np.ndarray) -> None:
        self.best_params = np.asarray(first_row, dtype=np.float32).reshape(dim).copy()
        self.best_fitness = float("-inf")
        self.generation_best_fitness = float("nan")
        self._history = []

    def _record(self, pop: np.ndarray, f: np.ndarray) -> float:
        i = int(np.argmax(f))
        gen_best = float(f[i])
        if gen_best > self.best_fitness:
            self.best_fitness = gen_best
            self.best_params = np.asarray(pop[i], dtype=np.float32).copy()
        self.generation_best_fitness = gen_best
        self._history.append(gen_best)
        return gen_best

    @property
    def best_history(self) -> list[float]:
        """Best fitness of each generation told so far (oldest first)."""
        return list(self._history)

    def _best_state(self) -> dict:
        return {
            "best_params": self.best_params.copy(),
            "best_fitness": np.float64(self.best_fitness),
            "generation_best_fitness": np.float64(self.generation_best_fitness),
            "best_history": np.asarray(self._history, dtype=np.float64),
        }

    def _load_best_state(self, d: dict, dim: int) -> None:
        bp = np.asarray(d["best_params"], dtype=np.float32)
        if bp.shape != (dim,):
            raise ValueError(f"best_params must have shape ({dim},), got {bp.shape}")
        self.best_params = bp.copy()
        self.best_fitness = float(d["best_fitness"])
        self.generation_best_fitness = float(d["generation_best_fitness"])
        self._history = [float(x) for x in np.asarray(d["best_history"], dtype=np.float64)]


# ----------------------------------------------------------------------------------------
# GA
# ----------------------------------------------------------------------------------------


class GA(_BestTracker):
    """Generational GA with elitism, tournament selection, BLX-0.5 crossover and an
    adaptive Gaussian mutation step.

    **Generation step** (in ``tell``):

    1. *Elites*: the ``elites`` best genomes of the told population are copied unchanged to
       the front of the next population, so they are **re-evaluated every generation**. With
       noisy fitness a lucky elite can't stay on top forever.
    2. *Children*: each of the other ``population - elites`` rows picks two parents by
       tournament (``tournament`` contestants drawn uniformly with replacement, the fittest
       wins), blends them with BLX-0.5 (each gene uniform in
       ``[min - 0.5 d, max + 0.5 d]``, ``d = |p1 - p2|``) and adds ``N(0, sigma)`` noise.

    **Sigma adaptation**, using the per-generation best ``g`` (not the best ever):

    - *Improvement*: ``g`` beats (strictly) the best of the previous ``patience``
      generations. Then ``sigma *= 0.9`` and the stale counter resets.
    - Otherwise the generation is *stale*. After ``patience`` consecutive stale generations
      ``sigma *= 1.5`` and the counter resets, so a flat run widens its search every
      ``patience`` generations.
    - ``sigma`` always stays within ``[sigma_min, sigma_max]`` (``sigma0`` is clamped too).
      The first generation only sets the baseline and leaves ``sigma`` unchanged.

    Comparing against a sliding window, rather than the best ever, keeps one lucky noisy
    evaluation from freezing ``sigma`` at its maximum forever.

    **Initial population.** With ``init`` (``(dim,)`` or ``(k, dim)``), the rows are used as
    they are (only the first ``population`` if ``k`` is larger), and any remaining rows are
    ``init[i % k] + N(0, sigma0)``. The trainer passes ``policy.init_params(spec, population,
    rng)``. Without ``init``, rows are drawn ``N(0, sigma0)``: that's only a fallback, and not
    the per-layer ``1/fan_in`` scaling of ``policy.init_params``.

    **Determinism.** Everything random comes from ``np.random.default_rng(seed)`` (PCG64), so
    the same seed and the same told fitnesses give the same asks. Asks are float32, and the
    population is stored as float32, so elites are bit-identical in the next ask.
    """

    def __init__(
        self,
        dim: int,
        *,
        population: int = 64,
        elites: int = 4,
        tournament: int = 4,
        sigma0: float = 0.1,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        patience: int = 10,
        seed: int = 0,
        init: np.ndarray | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if population < 2:
            raise ValueError(f"population must be >= 2, got {population}")
        if not 0 <= elites < population:
            raise ValueError(f"elites must be in [0, population), got {elites}")
        if tournament < 1:
            raise ValueError(f"tournament must be >= 1, got {tournament}")
        if not 0 < sigma_min <= sigma_max:
            raise ValueError(f"need 0 < sigma_min <= sigma_max, got {sigma_min}, {sigma_max}")
        if sigma0 <= 0:
            raise ValueError(f"sigma0 must be > 0, got {sigma0}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.dim = int(dim)
        self.population = int(population)
        self.elites = int(elites)
        self.tournament = int(tournament)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.patience = int(patience)
        self.seed = int(seed)

        self.sigma = float(np.clip(sigma0, self.sigma_min, self.sigma_max))
        self.generation = 0
        self._stale = 0
        self._rng = np.random.default_rng(self.seed)

        P, D = self.population, self.dim
        if init is None:
            pop = self._rng.normal(0.0, sigma0, size=(P, D))
        else:
            rows = _init_rows(init, D)[:P]
            k = rows.shape[0]
            pop = np.empty((P, D), dtype=np.float64)
            pop[:k] = rows
            if k < P:
                base = rows[np.arange(k, P) % k]
                pop[k:] = base + self._rng.normal(0.0, sigma0, size=(P - k, D))
        self._pop = pop.astype(np.float32)
        self._reset_best(D, self._pop[0])

    # -- protocol ------------------------------------------------------------------------

    def ask(self) -> np.ndarray:
        return self._pop.copy()

    def tell(self, fitness: np.ndarray) -> None:
        f = _clean_fitness(fitness, self.population)
        window = self._history[-self.patience :]
        gen_best = self._record(self._pop, f)
        self._adapt_sigma(gen_best, window)
        self._pop = self._breed(f)
        self.generation += 1

    # -- internals -----------------------------------------------------------------------

    def _adapt_sigma(self, gen_best: float, window: list[float]) -> None:
        if not window:
            return  # first generation: baseline only
        if gen_best > max(window):
            self.sigma *= 0.9
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.sigma *= 1.5
                self._stale = 0
        self.sigma = float(np.clip(self.sigma, self.sigma_min, self.sigma_max))

    def _breed(self, f: np.ndarray) -> np.ndarray:
        P, D, E = self.population, self.dim, self.elites
        rng = self._rng
        # stable sort on -f: ties keep population order, so it's deterministic
        order = np.argsort(-f, kind="stable")
        new = np.empty((P, D), dtype=np.float32)
        new[:E] = self._pop[order[:E]]

        n_child = P - E
        if n_child:
            contestants = rng.integers(0, P, size=(n_child, 2, self.tournament))
            winners = np.take_along_axis(
                contestants, np.argmax(f[contestants], axis=-1)[..., None], axis=-1
            )[..., 0]
            # float32 and in place: ~3x faster than float64 temporaries at dim 5.6k
            p1 = self._pop[winners[:, 0]]  # fancy indexing copies
            p2 = self._pop[winners[:, 1]]
            span = np.abs(p1 - p2)
            lo = np.minimum(p1, p2, out=p1)
            # BLX-0.5: lo + span * (2u - 0.5), u ~ U[0, 1)  ->  [min - span/2, max + span/2)
            child = rng.random((n_child, D), dtype=np.float32)
            child *= 2.0
            child -= 0.5
            child *= span
            child += lo
            noise = rng.standard_normal((n_child, D), dtype=np.float32)
            noise *= np.float32(self.sigma)
            child += noise
            new[E:] = child
        return new

    # -- state ---------------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "kind": np.str_("ga"),
            "dim": np.int64(self.dim),
            "population": np.int64(self.population),
            "generation": np.int64(self.generation),
            "sigma": np.float64(self.sigma),
            "stale": np.int64(self._stale),
            "pop": self._pop.copy(),
            "rng_state": _rng_state_to_array(self._rng),
            **self._best_state(),
        }

    def load_state_dict(self, d: dict) -> None:
        _check_kind(d, "ga", self.dim, self.population)
        pop = np.asarray(d["pop"], dtype=np.float32)
        if pop.shape != (self.population, self.dim):
            raise ValueError(f"pop must have shape {(self.population, self.dim)}, got {pop.shape}")
        self._pop = pop.copy()
        self.generation = int(d["generation"])
        self.sigma = float(d["sigma"])
        self._stale = int(d["stale"])
        self._rng = _rng_from_array(d["rng_state"])
        self._load_best_state(d, self.dim)


# ----------------------------------------------------------------------------------------
# CMA-ES
# ----------------------------------------------------------------------------------------


class _Randn:
    """``randn(*shape)`` for cma, drawing from our own Generator.

    cma's default sampler uses the global ``np.random`` state, which other code can disturb
    and which ``pickle.dumps(es)`` doesn't capture. With this object in ``opts['randn']`` the
    generator lives inside ``es``, so it's isolated, deterministic and pickled with it.
    Module level so it pickles.
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self.rng.standard_normal(shape)


def _import_cma():
    try:
        import cma
    except ImportError as e:
        raise ImportError(
            f"CMA-ES needs the optional 'cma' package (>=4.0): {CMA_INSTALL_HINT}"
        ) from e
    return cma


class CMAES(_BestTracker):
    """CMA-ES through the ``cma`` package, with diagonal covariance (sep-CMA), which scales
    to the ~5.6k policy parameters.

    - ``x0`` is the mean of the ``init`` rows (or the ``(dim,)`` vector itself), else zeros.
      Note that the mean of many random ``policy.init_params`` rows shrinks toward zero, so
      pass a single row (``n=1``) to start from one properly scaled network.
    - ``population=None`` uses cma's default ``4 + floor(3 ln dim)`` (29 at dim 5.6k).
    - Options: ``CMA_diagonal=True``, ``popsize``, ``verbose=-9`` (no output, no log files).
      Sampling uses ``randn`` from ``np.random.default_rng(seed + 1)`` stored inside ``es``
      instead of cma's ``seed`` option, which only seeds the global ``np.random``. So the
      run is isolated from global RNG use and the pickled state is complete.
    - ``tell`` passes ``-fitness`` because cma minimises. Non-finite fitness counts as worse
      than every finite one.
    - ``sigma`` is ``es.sigma``.

    **State.** ``state_dict()["es_pickle"]`` is ``pickle.dumps(es)`` as a uint8 array, so it
    fits the NPZ, which is loaded with ``allow_pickle=False``. ``load_state_dict`` unpickles
    it, so only load optimiser state from checkpoints this project wrote: unpickling
    untrusted data can run arbitrary code.
    """

    def __init__(
        self,
        dim: int,
        *,
        population: int | None = None,
        sigma0: float = 0.1,
        seed: int = 0,
        init: np.ndarray | None = None,
    ) -> None:
        cma = _import_cma()
        if dim < 2:
            raise ValueError(f"dim must be >= 2 for cma, got {dim}")
        if sigma0 <= 0:
            raise ValueError(f"sigma0 must be > 0, got {sigma0}")
        if population is not None and population < 2:
            raise ValueError(f"population must be >= 2, got {population}")
        self.dim = int(dim)
        self.seed = int(seed)
        x0 = np.zeros(self.dim) if init is None else _init_rows(init, self.dim).mean(axis=0)
        opts: dict[str, Any] = {
            "CMA_diagonal": True,
            "seed": np.nan,  # see class doc: randn below replaces the global-RNG seed
            "randn": _Randn(self.seed + 1),
            "verbose": -9,
        }
        if population is not None:
            opts["popsize"] = int(population)
        self._es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)
        self.population = int(self._es.popsize)
        self.generation = 0
        self._pending: np.ndarray | None = None  # float64 solutions of the current ask
        self._reset_best(self.dim, x0)

    @property
    def sigma(self) -> float:
        return float(self._es.sigma)

    def ask(self) -> np.ndarray:
        if self._pending is None:
            self._pending = np.asarray(self._es.ask(), dtype=np.float64)
        return self._pending.astype(np.float32)

    def tell(self, fitness: np.ndarray) -> None:
        if self._pending is None:
            raise RuntimeError("tell() called before ask()")
        f = _clean_fitness(fitness, self.population)
        self._record(self._pending, f)
        cost = -f
        bad = ~np.isfinite(cost)
        if bad.any():
            finite = cost[~bad]
            worst = float(finite.max()) if finite.size else 0.0
            cost[bad] = worst + 1.0 + abs(worst)
        self._es.tell(list(self._pending), cost.tolist())
        self._pending = None
        self.generation += 1

    def state_dict(self) -> dict:
        pending = np.zeros((0, self.dim)) if self._pending is None else self._pending.copy()
        return {
            "kind": np.str_("cmaes"),
            "dim": np.int64(self.dim),
            "population": np.int64(self.population),
            "generation": np.int64(self.generation),
            "es_pickle": np.frombuffer(pickle.dumps(self._es), dtype=np.uint8).copy(),
            "pending": pending,
            **self._best_state(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from :meth:`state_dict`. Unpickles ``es_pickle``: own checkpoints only."""
        _check_kind(d, "cmaes", self.dim, self.population)
        blob = np.asarray(d["es_pickle"], dtype=np.uint8).tobytes()
        self._es = pickle.loads(blob)  # our own checkpoints only (see class doc)
        pending = np.asarray(d["pending"], dtype=np.float64)
        self._pending = pending.copy() if pending.shape[0] else None
        self.generation = int(d["generation"])
        self._load_best_state(d, self.dim)


# ----------------------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------------------

_OPTIMISERS = {"ga": GA, "cmaes": CMAES}


def make_optimiser(name: str, dim: int, **kwargs: Any) -> Optimiser:
    """``"ga"`` -> :class:`GA`, ``"cmaes"`` -> :class:`CMAES` (case-insensitive).

    ``kwargs`` go to the constructor unchanged, so GA-only options (``elites``,
    ``tournament``, ``sigma_min``, ...) raise ``TypeError`` for CMA-ES. ``population``,
    ``sigma0``, ``seed`` and ``init`` work for both. Raises ``ImportError`` naming
    ``pip install -e ".[es]"`` when ``cma`` is missing.
    """
    key = name.strip().lower()
    if key not in _OPTIMISERS:
        raise ValueError(f"unknown optimiser {name!r}; expected one of {sorted(_OPTIMISERS)}")
    return _OPTIMISERS[key](dim, **kwargs)
