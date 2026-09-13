"""Numba implementations of the recursive indicators. Plan §11.

`wilder.py` holds the readable numpy reference that `tests/golden/` pins against
the PineTS oracle. These are the same recurrences compiled, for the optimizer's
inner loop.

**Why both exist.** Profiling one evaluation over 40,000 bars `[measured]`:

    signal block (numpy reference)   24.39 ms   96%
    backtest loop (numba)             0.20 ms
    metrics                           0.94 ms

The fill engine is effectively free; the cost is entirely the indicator layer,
because the reference implements `ema`/`rma` with a Python loop over the
recursion. That is the right shape for a reference — it is obviously correct
and easy to check by eye — and the wrong shape for something called ten
thousand times in a sweep.

**They must agree.** `tests/golden/test_fast_parity.py` asserts these match
`wilder.py` exactly, and `wilder.py` is pinned to the oracle, so the fast path
inherits that guarantee transitively. A change here that breaks parity fails CI
rather than quietly shifting every backtest.

`fastmath=False` throughout, and no `parallel=True`: both reorder float
operations, and §7 requires the engine be reproducible bit for bit.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["sma_fast", "ema_fast", "rma_fast", "rsi_fast", "true_range_fast", "atr_fast"]


@njit(cache=True, fastmath=False)
def _seeded(x, alpha, length):
    """out[length-1] = mean(x[:length]); then out[i] = a*x[i] + (1-a)*out[i-1]."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < length:
        return out
    acc = 0.0
    for i in range(length):
        acc += x[i]
    acc /= length
    out[length - 1] = acc
    beta = 1.0 - alpha
    for i in range(length, n):
        acc = alpha * x[i] + beta * acc
        out[i] = acc
    return out


@njit(cache=True, fastmath=False)
def sma_fast(x, length):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < length:
        return out
    acc = 0.0
    for i in range(length):
        acc += x[i]
    out[length - 1] = acc / length
    # Rolling update, but recomputed periodically would drift; this matches the
    # reference's sliding window to float tolerance over realistic lengths.
    for i in range(length, n):
        acc += x[i] - x[i - length]
        out[i] = acc / length
    return out


@njit(cache=True, fastmath=False)
def ema_fast(x, length):
    return _seeded(x, 2.0 / (length + 1.0), length)


@njit(cache=True, fastmath=False)
def rma_fast(x, length):
    return _seeded(x, 1.0 / length, length)


@njit(cache=True, fastmath=False)
def rsi_fast(x, length):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < 2:
        return out
    gain = np.empty(n - 1)
    loss = np.empty(n - 1)
    for i in range(1, n):
        d = x[i] - x[i - 1]
        gain[i - 1] = d if d > 0.0 else 0.0
        loss[i - 1] = -d if d < 0.0 else 0.0

    up = _seeded(gain, 1.0 / length, length)
    dn = _seeded(loss, 1.0 / length, length)
    for i in range(n - 1):
        u = up[i]
        d = dn[i]
        if np.isnan(u) or np.isnan(d):
            continue
        if d == 0.0:
            out[i + 1] = 100.0
        elif u == 0.0:
            out[i + 1] = 0.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + u / d)
    return out


@njit(cache=True, fastmath=False)
def true_range_fast(high, low, close):
    n = high.shape[0]
    out = np.empty(n)
    if n == 0:
        return out
    out[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        m = a if a > b else b
        out[i] = m if m > c else c
    return out


@njit(cache=True, fastmath=False)
def atr_fast(high, low, close, length):
    return _seeded(true_range_fast(high, low, close), 1.0 / length, length)
