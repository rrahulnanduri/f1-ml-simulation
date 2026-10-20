"""Optional numba acceleration, shared by the hot loops (ray casts, track projection).

numba is an optional dependency (``pip install -e ".[fast]"``). Without it, or with the
``F1SIM_NO_NUMBA`` environment variable set, every caller uses its numpy implementation.
Both give the same results; tests compare them.
"""

from __future__ import annotations

import os

NO_NUMBA_ENV = "F1SIM_NO_NUMBA"

try:
    import numba
except ImportError:  # pragma: no cover - depends on the environment
    numba = None


def njit(fn):
    """``numba.njit(cache=True, nogil=True)`` if numba is installed, else None."""
    if numba is None:
        return None
    return numba.njit(cache=True, nogil=True)(fn)


def enabled(kernel) -> bool:
    """Whether to use a compiled ``kernel`` (it exists and isn't disabled by the env var)."""
    return kernel is not None and not os.environ.get(NO_NUMBA_ENV)
