"""Feed-forward policy with one weight row per car (docs/m3-interfaces.md section I).

A policy maps an observation row (``sim.sensors``, spec v1) to an action ``(steer, pedal)`` in
``[-1, 1]``. Its weights live in a flat vector so that the optimisers can treat each genome as
one row of a ``(population, n_params)`` array.

Flat layout, row-major, in this order::

    W1 (obs_size, h1), b1 (h1,), W2 (h1, h2), b2 (h2,), ..., Wout (h_last, 2), bout (2,)

A layer computes ``h_next = f(h @ W + b)``, with ``f`` the hidden activation (``tanh`` or
``relu``) and always ``tanh`` on the output layer.

:func:`act` evaluates N cars at once, each with its own row of weights, using batched matmuls on
views into the flat array (no copies and no Python loop over cars). With float32 weights it
computes in float32, which is what the optimisers produce; any other dtype is computed in
float64. Measured on the dev machine with (64, 64) hidden layers and float32 weights: about
0.14 ms for N = 192 and 1.5-3.3 ms for N = 1024 (memory bound: the weights are ~22 MB at that
size). ``np.matmul`` beat ``np.einsum`` (with or without ``optimize``) by 15-30% and a
broadcast-multiply-sum by ~10x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1sim.sim.sensors import OBS_SIZE

ACTION_SIZE = 2
OUTPUT_ACTIVATION = "tanh"
ACTIVATIONS = ("tanh", "relu")


@dataclass(frozen=True)
class PolicySpec:
    """Shape of a policy network: input size, hidden layer sizes, output size, activation."""

    obs_size: int = OBS_SIZE
    hidden: tuple[int, ...] = (64, 64)
    action_size: int = ACTION_SIZE
    activation: str = "tanh"  # hidden activation; the output is always tanh

    def __post_init__(self) -> None:
        hidden = tuple(self.hidden)
        object.__setattr__(self, "hidden", hidden)
        for name, value in (("obs_size", self.obs_size), ("action_size", self.action_size)):
            if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
                raise ValueError(f"PolicySpec.{name} must be a positive int, got {value!r}")
        for h in hidden:
            if isinstance(h, bool) or not isinstance(h, int | np.integer) or h <= 0:
                raise ValueError(f"PolicySpec.hidden must hold positive ints, got {hidden!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"PolicySpec.activation must be one of {ACTIVATIONS}, got {self.activation!r}"
            )

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """``(obs_size, *hidden, action_size)``."""
        return (int(self.obs_size), *(int(h) for h in self.hidden), int(self.action_size))

    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector."""
        sizes = self.layer_sizes
        return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:], strict=True))

    def to_dict(self) -> dict:
        """JSON-ready description, stored as ``meta["policy"]`` in checkpoints."""
        return {
            "obs_size": int(self.obs_size),
            "hidden": [int(h) for h in self.hidden],
            "action_size": int(self.action_size),
            "activation": self.activation,
            "output_activation": OUTPUT_ACTIVATION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PolicySpec:
        """Inverse of :meth:`to_dict`; missing keys take their defaults, unknown keys raise."""
        d = dict(d)
        out_act = d.pop("output_activation", OUTPUT_ACTIVATION)
        if out_act != OUTPUT_ACTIVATION:
            raise ValueError(
                f"unsupported output_activation {out_act!r} (only {OUTPUT_ACTIVATION!r})"
            )
        known = {"obs_size", "hidden", "action_size", "activation"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown PolicySpec key(s): {', '.join(unknown)}")
        if "hidden" in d:
            d["hidden"] = tuple(d["hidden"])
        return cls(**d)


def _layout(spec: PolicySpec) -> list[tuple[int, int, int, int]]:
    """``(w_start, b_start, fan_in, fan_out)`` of each layer in the flat vector."""
    sizes = spec.layer_sizes
    out = []
    offset = 0
    for i, o in zip(sizes[:-1], sizes[1:], strict=True):
        out.append((offset, offset + i * o, i, o))
        offset += i * o + o
    return out


def _check_params(spec: PolicySpec, params: np.ndarray) -> None:
    if params.ndim not in (1, 2) or params.shape[-1] != spec.n_params:
        raise ValueError(
            f"params must have shape (N, {spec.n_params}) or ({spec.n_params},) for "
            f"layers {spec.layer_sizes}, got {params.shape}"
        )


def unflatten(spec: PolicySpec, params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(W, b), ...]`` views into ``params``, one pair per layer.

    For a ``(n_params,)`` vector, ``W`` is ``(fan_in, fan_out)`` and ``b`` ``(fan_out,)``; for an
    ``(N, n_params)`` array they gain a leading ``N`` axis. They are views (writing to them
    writes ``params``) whenever the last axis of ``params`` is contiguous.
    """
    params = np.asarray(params)
    _check_params(spec, params)
    lead = params.shape[:-1]
    return [
        (params[..., w0:b0].reshape(*lead, i, o), params[..., b0 : b0 + o])
        for w0, b0, i, o in _layout(spec)
    ]


def init_params(spec: PolicySpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, n_params)`` float32. Biases are 0; weights ~ N(0, 1/fan_in)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    params = np.zeros((n, spec.n_params), dtype=np.float32)
    for W, _b in unflatten(spec, params):
        fan_in = W.shape[1]
        W[...] = rng.standard_normal(W.shape, dtype=np.float32) * np.float32(1.0 / np.sqrt(fan_in))
    return params


def _hidden_activation(name: str):
    if name == "tanh":
        return lambda h: np.tanh(h, out=h)
    return lambda h: np.maximum(h, 0.0, out=h)


def act(spec: PolicySpec, params: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Actions ``(N, action_size)`` float64 in ``[-1, 1]`` for observations ``(N, obs_size)``.

    ``params`` is ``(N, n_params)``, one row per car, or ``(n_params,)``, shared by every car.
    Per-car rows are evaluated with batched matmuls ``(N, 1, in) @ (N, in, out)`` on views of
    ``params``, with no Python loop over cars. float32 params are computed in float32 (faster),
    anything else in float64. Shapes are validated with descriptive ``ValueError``.
    """
    params = np.asarray(params)
    obs = np.asarray(obs)
    _check_params(spec, params)
    if obs.ndim != 2 or obs.shape[1] != spec.obs_size:
        raise ValueError(f"obs must have shape (N, {spec.obs_size}), got {obs.shape}")
    n = obs.shape[0]
    if params.ndim == 2 and params.shape[0] != n:
        raise ValueError(
            f"params has {params.shape[0]} rows but obs has {n}; pass one row per car "
            "or a single (n_params,) vector to share it"
        )

    dtype = np.float32 if params.dtype == np.float32 else np.float64
    if params.dtype != dtype or params.strides[-1] != params.itemsize:
        params = np.ascontiguousarray(params, dtype=dtype)
    hidden_f = _hidden_activation(spec.activation)
    layers = unflatten(spec, params)
    last = len(layers) - 1

    if params.ndim == 1:
        # One weight set shared by every car: plain GEMMs.
        h = obs.astype(dtype, copy=False)
        for k, (W, b) in enumerate(layers):
            h = h @ W
            h += b
            if k == last:
                np.tanh(h, out=h)
            else:
                hidden_f(h)
        return h.astype(np.float64)

    # Per-car weights: (N, 1, in) @ (N, in, out), one small GEMM per car inside numpy.
    h = obs.astype(dtype, copy=False)[:, None, :]
    for k, (W, b) in enumerate(layers):
        h = np.matmul(h, W)
        h += b[:, None, :]
        if k == last:
            np.tanh(h, out=h)
        else:
            hidden_f(h)
    return h[:, 0, :].astype(np.float64)
