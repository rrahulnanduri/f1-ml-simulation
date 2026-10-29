"""Tests for f1sim.agents.optimisers (GA and diagonal CMA-ES)."""

from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from f1sim.agents.optimisers import CMAES, GA, Optimiser, make_optimiser

DIM = 50
CENTRE = np.linspace(-0.5, 0.5, DIM)


def sphere(x: np.ndarray) -> np.ndarray:
    return -np.sum((x.astype(np.float64) - CENTRE) ** 2, axis=1)


def rastrigin_lite(x: np.ndarray) -> np.ndarray:
    """Shifted Rastrigin with amplitude 1 instead of 10: still multimodal, but solvable."""
    z = x.astype(np.float64) - CENTRE
    return -np.sum(z**2 + 1.0 - np.cos(2 * np.pi * z), axis=1)


def run(opt: Optimiser, fn, generations: int) -> float:
    for _ in range(generations):
        opt.tell(fn(opt.ask()))
    return opt.best_fitness


def npz_roundtrip(d: dict) -> dict:
    """What the checkpoint does: savez, then load with allow_pickle=False."""
    buf = io.BytesIO()
    np.savez(buf, **d)
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _have_cma() -> bool:
    try:
        import cma  # noqa: F401
    except ImportError:
        return False
    return True


needs_cma = pytest.mark.skipif(not _have_cma(), reason='needs cma: pip install -e ".[es]"')

OPTIMISERS = [
    pytest.param("ga", {"population": 64}, id="ga"),
    pytest.param("cmaes", {"population": 32}, id="cmaes", marks=needs_cma),
]


# -- convergence -----------------------------------------------------------------------
# From the default N(0, 0.1) start the best of the first generation is about -4 on the
# sphere (|c|^2 = 4.25) and about -50 on Rastrigin-lite. In 100 generations both optimisers
# reach better than -0.02 on the sphere and better than -3 on Rastrigin-lite over seeds 0-4
# (measured), so the thresholds of -0.1 (a 40x reduction) and -6 (8x) leave a wide margin.


@pytest.mark.parametrize(("name", "kw"), OPTIMISERS)
@pytest.mark.parametrize("seed", [0, 1])
def test_improves_sphere(name, kw, seed):
    opt = make_optimiser(name, DIM, seed=seed, **kw)
    first = sphere(opt.ask()).max()
    best = run(opt, sphere, 100)
    assert first < -3.0
    assert best > -0.1
    assert opt.generation == 100
    assert len(opt.best_history) == 100
    np.testing.assert_allclose(sphere(opt.best_params[None])[0], best, rtol=1e-5)


@pytest.mark.parametrize(("name", "kw"), OPTIMISERS)
@pytest.mark.parametrize("seed", [0, 1])
def test_improves_rastrigin_lite(name, kw, seed):
    opt = make_optimiser(name, DIM, seed=seed, **kw)
    first = rastrigin_lite(opt.ask()).max()
    best = run(opt, rastrigin_lite, 100)
    assert first < -40.0
    assert best > -6.0


# -- GA mechanics ----------------------------------------------------------------------


def test_ga_ask_shape_dtype_and_idempotent():
    ga = GA(DIM, population=16, seed=3)
    a = ga.ask()
    assert a.shape == (16, DIM) and a.dtype == np.float32
    np.testing.assert_array_equal(a, ga.ask())
    a[:] = 0  # the caller's copy doesn't alias the population
    assert np.any(ga.ask() != 0)


def test_ga_elites_unchanged_in_next_ask():
    ga = GA(DIM, population=20, elites=4, seed=1)
    for _ in range(3):
        pop = ga.ask()
        f = sphere(pop)
        ga.tell(f)
        top = np.argsort(-f, kind="stable")[:4]
        nxt = ga.ask()
        np.testing.assert_array_equal(nxt[:4], pop[top])
        # the children are new rows
        assert not any(np.array_equal(nxt[i], row) for i in range(4, 20) for row in pop)


def test_ga_best_tracking_with_noise():
    ga = GA(DIM, population=8, elites=1, seed=0)
    pop = ga.ask()
    ga.tell(np.array([1.0, 5.0, 2.0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(ga.best_params, pop[1])
    assert ga.best_fitness == 5.0 and ga.generation_best_fitness == 5.0
    ga.tell(np.full(8, 3.0))  # a worse (noisy) generation doesn't replace the best ever
    assert ga.best_fitness == 5.0 and ga.generation_best_fitness == 3.0
    np.testing.assert_array_equal(ga.best_params, pop[1])
    assert ga.best_history == [5.0, 3.0]


def test_ga_nonfinite_fitness_is_worst():
    ga = GA(DIM, population=4, elites=1, seed=0)
    pop = ga.ask()
    ga.tell(np.array([np.nan, np.inf, -1.0, -2.0]))
    assert ga.best_fitness == -1.0
    np.testing.assert_array_equal(ga.ask()[0], pop[2])


def test_ga_sigma_rises_after_patience_stale_generations():
    ga = GA(DIM, population=8, sigma0=0.1, patience=3, seed=0)
    flat = np.zeros(8)
    ga.tell(flat)  # baseline
    assert ga.sigma == pytest.approx(0.1)
    ga.tell(flat)
    ga.tell(flat)
    assert ga.sigma == pytest.approx(0.1)  # 2 stale generations: not yet
    ga.tell(flat)
    assert ga.sigma == pytest.approx(0.15)  # 3rd stale generation: x1.5
    for _ in range(3):
        ga.tell(flat)
    assert ga.sigma == pytest.approx(0.225)
    for _ in range(30):
        ga.tell(flat)
    assert ga.sigma == pytest.approx(0.5)  # clamped at sigma_max


def test_ga_sigma_falls_on_improvement_and_clamps():
    ga = GA(DIM, population=8, sigma0=0.1, sigma_min=0.05, patience=3, seed=0)
    ga.tell(np.zeros(8))
    ga.tell(np.ones(8))
    assert ga.sigma == pytest.approx(0.09)
    ga.tell(np.full(8, 2.0))
    assert ga.sigma == pytest.approx(0.081)
    for k in range(3, 30):
        ga.tell(np.full(8, float(k)))
    assert ga.sigma == pytest.approx(0.05)  # clamped at sigma_min


def test_ga_sigma_window_forgets_a_lucky_outlier():
    """Improvement is judged against the last `patience` generation bests, so one lucky
    evaluation stops counting after `patience` generations."""
    ga = GA(DIM, population=8, sigma0=0.1, patience=3, seed=0)
    ga.tell(np.zeros(8))
    ga.tell(np.full(8, 100.0))  # lucky: improvement, sigma 0.09
    for _ in range(3):
        ga.tell(np.ones(8))  # stale x3 -> 0.135
    assert ga.sigma == pytest.approx(0.135)
    ga.tell(np.full(8, 1.5))  # 100 has left the window: improvement
    assert ga.sigma == pytest.approx(0.1215)


def test_ga_success_rule_shrinks_when_children_fail_and_grows_when_they_win():
    ga = GA(DIM, population=10, elites=2, sigma0=0.1, step_rule="success", seed=0)
    ga.tell(np.zeros(10))  # first population has no parents: sigma unchanged
    assert ga.sigma == pytest.approx(0.1)
    elite_good = np.r_[np.full(2, 5.0), np.zeros(8)]  # no child beats the elites
    ga.tell(elite_good)
    assert ga.sigma == pytest.approx(0.1 * np.exp(0.5 * (0.0 - 0.2)))
    sigma = ga.sigma
    all_win = np.r_[np.zeros(2), np.ones(8)]  # every child beats the worst elite
    ga.tell(all_win)
    assert ga.sigma == pytest.approx(sigma * np.exp(0.5 * (1.0 - 0.2)))


def test_ga_success_rule_is_clamped_and_validates():
    ga = GA(DIM, population=10, elites=2, sigma_min=0.05, sigma0=0.06, step_rule="success")
    ga.tell(np.zeros(10))
    for _ in range(20):
        ga.tell(np.r_[np.full(2, 5.0), np.zeros(8)])
    assert ga.sigma == pytest.approx(0.05)  # clamped at sigma_min
    with pytest.raises(ValueError, match="step_rule"):
        GA(DIM, population=10, step_rule="nope")
    with pytest.raises(ValueError, match="success_target"):
        GA(DIM, population=10, success_target=1.0)


def test_ga_init_rows_and_fill():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(3, DIM)).astype(np.float32)
    ga = GA(DIM, population=10, sigma0=0.01, init=rows, seed=0)
    pop = ga.ask()
    np.testing.assert_array_equal(pop[:3], rows)
    for i in range(3, 10):
        d = pop[i] - rows[i % 3]
        assert 0 < np.abs(d).max() < 0.1  # small Gaussian perturbation of its base row
    # a single vector and too many rows
    one = GA(DIM, population=4, elites=1, init=rows[0], seed=0).ask()
    np.testing.assert_array_equal(one[0], rows[0])
    many = rng.normal(size=(12, DIM))
    np.testing.assert_array_equal(
        GA(DIM, population=4, elites=1, init=many).ask(), many[:4].astype(np.float32)
    )
    with pytest.raises(ValueError, match="init"):
        GA(DIM, init=np.zeros((2, DIM + 1)))


def test_ga_default_init_is_normal_sigma0():
    pop = GA(2000, population=8, sigma0=0.2, seed=0).ask()
    assert abs(pop.mean()) < 0.01
    assert pop.std() == pytest.approx(0.2, rel=0.02)


def test_ga_validation():
    with pytest.raises(ValueError):
        GA(DIM, population=4, elites=4)
    with pytest.raises(ValueError):
        GA(DIM, sigma_min=0.5, sigma_max=0.1)
    ga = GA(DIM, population=4, elites=1)
    with pytest.raises(ValueError, match="shape"):
        ga.tell(np.zeros(5))


# -- determinism and state ---------------------------------------------------------------


def _noisy_fitness(x: np.ndarray, gen: int) -> np.ndarray:
    """Deterministic but noisy-looking fitness, so ties and orderings vary."""
    return sphere(x) + 0.1 * np.sin(np.arange(len(x)) * 1.7 + gen)


@pytest.mark.parametrize(("name", "kw"), OPTIMISERS)
def test_same_seed_same_asks(name, kw):
    a = make_optimiser(name, DIM, seed=7, **kw)
    b = make_optimiser(name, DIM, seed=7, **kw)
    c = make_optimiser(name, DIM, seed=8, **kw)
    assert not np.array_equal(a.ask(), c.ask())
    for g in range(15):
        xa, xb = a.ask(), b.ask()
        np.testing.assert_array_equal(xa, xb)
        a.tell(_noisy_fitness(xa, g))
        b.tell(_noisy_fitness(xb, g))
    assert a.sigma == b.sigma
    assert a.best_history == b.best_history


@pytest.mark.parametrize(("name", "kw"), OPTIMISERS)
@pytest.mark.parametrize("pending", [False, True], ids=["after_tell", "after_ask"])
def test_state_dict_roundtrip_gives_identical_asks(name, kw, pending):
    ref = make_optimiser(name, DIM, seed=11, **kw)
    for g in range(6):
        ref.tell(_noisy_fitness(ref.ask(), g))
    if pending:
        ref.ask()
    state = npz_roundtrip(ref.state_dict())
    for v in state.values():
        assert v.dtype != object

    # a different seed proves the RNG state comes from the dict
    new = make_optimiser(name, DIM, seed=999, **kw)
    new.load_state_dict(state)
    assert new.generation == ref.generation == 6
    assert new.sigma == ref.sigma
    assert new.best_fitness == ref.best_fitness
    assert new.best_history == ref.best_history
    np.testing.assert_array_equal(new.best_params, ref.best_params)
    for g in range(6, 12):
        xr, xn = ref.ask(), new.ask()
        np.testing.assert_array_equal(xr, xn)
        ref.tell(_noisy_fitness(xr, g))
        new.tell(_noisy_fitness(xn, g))
    assert new.sigma == ref.sigma


def test_load_state_dict_rejects_mismatch():
    ga = GA(DIM, population=8)
    with pytest.raises(ValueError, match="population"):
        GA(DIM, population=10).load_state_dict(ga.state_dict())
    with pytest.raises(ValueError, match="dim"):
        GA(DIM + 1, population=8).load_state_dict(ga.state_dict())


# -- CMA-ES specifics -------------------------------------------------------------------


@needs_cma
def test_cmaes_basics():
    es = CMAES(DIM, population=12, sigma0=0.3, seed=0, init=np.full((4, DIM), 2.0))
    x = es.ask()
    assert x.shape == (12, DIM) and x.dtype == np.float32
    np.testing.assert_array_equal(x, es.ask())  # idempotent until tell
    assert abs(x.mean() - 2.0) < 0.2  # x0 = mean of the init rows
    assert es.sigma == pytest.approx(0.3)
    with pytest.raises(ValueError, match="shape"):
        es.tell(np.zeros(5))
    f = sphere(x)
    f[0] = np.nan
    es.tell(f)
    assert es.generation == 1
    assert es.best_fitness == np.nanmax(f[1:])
    assert CMAES(5634, seed=0).population == 29  # cma default 4 + floor(3 ln n)
    assert isinstance(es, Optimiser)


@needs_cma
def test_cmaes_tell_before_ask_raises():
    with pytest.raises(RuntimeError):
        CMAES(DIM, population=8).tell(np.zeros(8))


@needs_cma
def test_cmaes_ignores_global_numpy_rng():
    """Sampling uses the optimiser's own generator, not np.random."""
    a = CMAES(DIM, population=8, seed=2)
    b = CMAES(DIM, population=8, seed=2)
    for g in range(4):
        np.random.seed(g)  # noqa: NPY002 - deliberately the legacy global RNG
        xa = a.ask()
        np.random.seed(1000 + g)  # noqa: NPY002
        np.random.standard_normal(100)  # noqa: NPY002
        xb = b.ask()
        np.testing.assert_array_equal(xa, xb)
        a.tell(sphere(xa))
        b.tell(sphere(xb))


def test_cmaes_missing_cma_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cma", None)  # makes `import cma` fail
    with pytest.raises(ImportError, match=r'pip install -e "\.\[es\]"'):
        make_optimiser("cmaes", DIM)


# -- factory ---------------------------------------------------------------------------


def test_make_optimiser_names():
    ga = make_optimiser("ga", DIM, population=8, elites=2, seed=0)
    assert isinstance(ga, GA) and ga.population == 8 and ga.elites == 2
    assert isinstance(make_optimiser(" GA ", DIM), GA)
    assert isinstance(ga, Optimiser)
    with pytest.raises(ValueError, match="unknown optimiser 'adam'"):
        make_optimiser("adam", DIM)


@needs_cma
def test_make_optimiser_cmaes():
    es = make_optimiser("cmaes", DIM, population=16, sigma0=0.2, seed=0)
    assert isinstance(es, CMAES) and es.population == 16
    assert isinstance(make_optimiser("CMAES", DIM), CMAES)


class _Boom:
    """Unpickling this calls ``os.system``: what a crafted checkpoint would carry."""

    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


@needs_cma
def test_cmaes_state_round_trips_through_safe_unpickler():
    a = make_optimiser("cmaes", DIM, population=8, seed=0)
    for _ in range(3):
        pop = a.ask()
        a.tell(-np.sum(pop.astype(np.float64) ** 2, axis=1))
    b = make_optimiser("cmaes", DIM, population=8, seed=0)
    b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(a.ask(), b.ask())


@needs_cma
def test_cmaes_refuses_crafted_es_pickle():
    import pickle

    es = make_optimiser("cmaes", DIM, population=8, seed=0)
    state = es.state_dict()
    state["es_pickle"] = np.frombuffer(pickle.dumps(_Boom()), dtype=np.uint8).copy()
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        es.load_state_dict(state)
