import numpy as np
import pytest

from f1sim.agents.policy import ACTION_SIZE, PolicySpec, act, init_params, unflatten
from f1sim.sim.sensors import OBS_SIZE


def _naive_act(spec: PolicySpec, row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One car, float64, explicit slicing of the documented flat layout."""
    sizes = spec.layer_sizes
    h = x.astype(np.float64)
    off = 0
    for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        W = row[off : off + i * o].astype(np.float64).reshape(i, o)
        off += i * o
        b = row[off : off + o].astype(np.float64)
        off += o
        h = h @ W + b
        if k == len(sizes) - 2 or spec.activation == "tanh":
            h = np.tanh(h)
        else:
            h = np.maximum(h, 0.0)
    assert off == row.shape[0]
    return h


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (PolicySpec(), 19 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2),
        (PolicySpec(hidden=()), 19 * 2 + 2),
        (PolicySpec(obs_size=3, hidden=(5,), action_size=2), 3 * 5 + 5 + 5 * 2 + 2),
        (PolicySpec(hidden=(32, 16, 8)), 19 * 32 + 32 + 32 * 16 + 16 + 16 * 8 + 8 + 8 * 2 + 2),
    ],
)
def test_n_params(spec, expected):
    assert spec.n_params == expected


def test_defaults_match_contract():
    spec = PolicySpec()
    assert spec.obs_size == OBS_SIZE
    assert spec.hidden == (64, 64)
    assert spec.action_size == ACTION_SIZE == 2
    assert spec.activation == "tanh"


def test_spec_dict_round_trip_and_validation():
    spec = PolicySpec(hidden=(8, 4), activation="relu")
    d = spec.to_dict()
    assert d["hidden"] == [8, 4]
    assert d["output_activation"] == "tanh"
    assert PolicySpec.from_dict(d) == spec
    assert PolicySpec(hidden=[8, 4]).hidden == (8, 4)
    with pytest.raises(ValueError, match="unknown"):
        PolicySpec.from_dict({**d, "dropout": 0.1})
    with pytest.raises(ValueError, match="activation"):
        PolicySpec(activation="sigmoid")
    with pytest.raises(ValueError, match="hidden"):
        PolicySpec(hidden=(64, 0))


def test_flat_layout_order():
    spec = PolicySpec(obs_size=3, hidden=(4,), action_size=2)
    flat = np.arange(spec.n_params, dtype=np.float32)
    (W1, b1), (W2, b2) = unflatten(spec, flat)
    np.testing.assert_array_equal(W1, np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(b1, np.arange(12, 16))
    np.testing.assert_array_equal(W2, np.arange(16, 24).reshape(4, 2))
    np.testing.assert_array_equal(b2, np.arange(24, 26))
    assert all(np.shares_memory(a, flat) for pair in unflatten(spec, flat) for a in pair)


def test_unflatten_batched_views():
    spec = PolicySpec(obs_size=3, hidden=(4,), action_size=2)
    params = np.arange(2 * spec.n_params, dtype=np.float32).reshape(2, spec.n_params)
    layers = unflatten(spec, params)
    assert [W.shape for W, _ in layers] == [(2, 3, 4), (2, 4, 2)]
    assert [b.shape for _, b in layers] == [(2, 4), (2, 2)]
    np.testing.assert_array_equal(layers[0][0][1], params[1, :12].reshape(3, 4))
    for W, b in layers:
        assert np.shares_memory(W, params) and np.shares_memory(b, params)
    layers[1][1][1, 0] = -1.0
    assert params[1, 24] == -1.0


@pytest.mark.parametrize("n", [1, 7, 64])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_act_matches_naive_loop(n, activation):
    spec = PolicySpec(hidden=(16, 8), activation=activation)
    rng = np.random.default_rng(n)
    params = (rng.standard_normal((n, spec.n_params)) * 0.5).astype(np.float32)
    obs = rng.uniform(-1, 1, (n, spec.obs_size)).astype(np.float32)
    out = act(spec, params, obs)
    assert out.shape == (n, 2)
    assert out.dtype == np.float64
    expected = np.stack([_naive_act(spec, params[i], obs[i]) for i in range(n)])
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_act_float64_params_exact():
    spec = PolicySpec(hidden=(64, 64))
    rng = np.random.default_rng(1)
    params = rng.standard_normal((7, spec.n_params)) * 0.2
    obs = rng.uniform(-1, 1, (7, spec.obs_size))
    expected = np.stack([_naive_act(spec, params[i], obs[i]) for i in range(7)])
    np.testing.assert_allclose(act(spec, params, obs), expected, atol=1e-12)


def test_act_broadcasts_single_row():
    spec = PolicySpec()
    rng = np.random.default_rng(2)
    row = init_params(spec, 1, rng)[0]
    obs = rng.uniform(-1, 1, (9, spec.obs_size)).astype(np.float32)
    shared = act(spec, row, obs)
    tiled = act(spec, np.tile(row, (9, 1)), obs)
    np.testing.assert_allclose(shared, tiled, atol=1e-6)
    expected = np.stack([_naive_act(spec, row, obs[i]) for i in range(9)])
    np.testing.assert_allclose(shared, expected, atol=1e-5)


def test_act_output_range_with_large_weights():
    spec = PolicySpec()
    rng = np.random.default_rng(3)
    params = (rng.standard_normal((50, spec.n_params)) * 100).astype(np.float32)
    obs = rng.uniform(-5, 5, (50, spec.obs_size)).astype(np.float32)
    out = act(spec, params, obs)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)
    assert np.any(np.abs(out) > 0.99)  # saturates, still in range


def test_act_non_contiguous_params():
    spec = PolicySpec(hidden=(8,))
    rng = np.random.default_rng(4)
    big = rng.standard_normal((6, 2 * spec.n_params)).astype(np.float32)
    params = big[:, ::2]  # column stride != itemsize
    obs = rng.uniform(-1, 1, (6, spec.obs_size))
    expected = np.stack([_naive_act(spec, params[i], obs[i]) for i in range(6)])
    np.testing.assert_allclose(act(spec, params, obs), expected, atol=1e-5)


def test_act_zero_cars():
    spec = PolicySpec()
    out = act(spec, np.zeros((0, spec.n_params), np.float32), np.zeros((0, spec.obs_size)))
    assert out.shape == (0, 2)


def test_act_shape_errors():
    spec = PolicySpec()
    obs = np.zeros((4, spec.obs_size), np.float32)
    with pytest.raises(ValueError, match=r"params must have shape"):
        act(spec, np.zeros((4, spec.n_params + 1), np.float32), obs)
    with pytest.raises(ValueError, match=r"params must have shape"):
        act(spec, np.zeros((4, 1, spec.n_params), np.float32), obs)
    with pytest.raises(ValueError, match=r"obs must have shape"):
        act(spec, np.zeros((4, spec.n_params), np.float32), np.zeros((4, 18)))
    with pytest.raises(ValueError, match=r"obs must have shape"):
        act(spec, np.zeros(spec.n_params, np.float32), np.zeros(spec.obs_size))
    with pytest.raises(ValueError, match=r"3 rows but obs has 4"):
        act(spec, np.zeros((3, spec.n_params), np.float32), obs)


def test_init_params_statistics():
    spec = PolicySpec()
    params = init_params(spec, 200, np.random.default_rng(5))
    assert params.shape == (200, spec.n_params)
    assert params.dtype == np.float32
    for W, b in unflatten(spec, params):
        fan_in = W.shape[1]
        assert np.all(b == 0.0)
        assert abs(W.mean()) < 0.01
        assert W.std() == pytest.approx(1.0 / np.sqrt(fan_in), rel=0.03)


def test_init_params_deterministic():
    spec = PolicySpec(hidden=(8,))
    a = init_params(spec, 3, np.random.default_rng(6))
    b = init_params(spec, 3, np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
