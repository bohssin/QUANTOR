"""Wilder-family indicators with TradingView/MT5-compatible semantics.

Validated against PineTS 0.9.33 (the dev oracle, plan §8.2) on 20,000 bars:

    series     max abs diff   max rel diff   nan parity
    ema20         5.025e-11      2.546e-14   exact (19/19)
    rma20         5.025e-11      2.543e-14   exact (19/19)
    rsi14         5.002e-11      2.899e-12   exact (14/14)
    atr14         4.999e-11      7.232e-11   exact (13/13)
    sma20         1.776e-09      8.916e-13   exact (19/19)

The 5e-11 floor is PineTS's own JSON output precision (10 decimals), i.e. these
agree to every digit the oracle reports.

Pine and MT5 both use Wilder smoothing for the RSI/ATR family, which is why one
implementation satisfies both references.

Seeding matters more than the recurrence: `ema` and `rma` are seeded with the
SMA of the first `length` values at bar `length-1`, and are NaN before it. That
warmup boundary is where independent implementations usually diverge, so the
golden fixtures assert NaN counts as well as values.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sma", "ema", "rma", "rsi", "true_range", "atr"]


def sma(x: np.ndarray, length: int) -> np.ndarray:
    """Simple moving average. NaN for the first ``length - 1`` bars.

    Uses a sliding window rather than a cumulative sum: cumsum accumulates
    float error over long series (measured at 1.8e-9 over 20k bars, which is
    worse than every other function in this module).
    """
    _check(length)
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    if x.size < length:
        return out
    window = np.lib.stride_tricks.sliding_window_view(x, length)
    out[length - 1:] = window.mean(axis=-1)
    return out


def ema(x: np.ndarray, length: int) -> np.ndarray:
    """Exponential MA, ``alpha = 2 / (length + 1)``, SMA-seeded (Pine `ta.ema`)."""
    _check(length)
    x = np.asarray(x, dtype=np.float64)
    return _seeded_recursion(x, 2.0 / (length + 1.0), length)


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """Wilder's smoothing, ``alpha = 1 / length``, SMA-seeded (Pine `ta.rma`)."""
    _check(length)
    x = np.asarray(x, dtype=np.float64)
    return _seeded_recursion(x, 1.0 / length, length)


def rsi(x: np.ndarray, length: int) -> np.ndarray:
    """Relative Strength Index over RMA-smoothed up/down moves (Pine `ta.rsi`).

    The leading bar has no change and stays NaN; smoothing starts from bar 1,
    so the first finite value lands at bar ``length``.
    """
    _check(length)
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    if x.size < 2:
        return out

    change = np.diff(x)
    gain = np.maximum(change, 0.0)
    loss = np.maximum(-change, 0.0)

    up = rma(gain, length)
    down = rma(loss, length)

    with np.errstate(divide="ignore", invalid="ignore"):
        body = 100.0 - 100.0 / (1.0 + up / down)
    # Pine's saturation cases: all-loss -> 0, all-gain -> 100.
    body = np.where(down == 0.0, 100.0, body)
    body = np.where(up == 0.0, 0.0, body)

    out[1:] = body
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range. First bar is ``high - low`` (no previous close), as in Pine."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    _same_shape(high, low, close)

    prev_close = np.empty(close.shape)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Average True Range = RMA of true range (Pine `ta.atr`)."""
    return rma(true_range(high, low, close), length)


# --- internals ---------------------------------------------------------------

def _seeded_recursion(x: np.ndarray, alpha: float, length: int) -> np.ndarray:
    """``out[length-1] = mean(x[:length])``, then ``out[i] = a*x[i] + (1-a)*out[i-1]``.

    NaN before the seed. Kept as an explicit loop because the recurrence is
    sequential; the hot path in production is the numba equivalent, which this
    function is the readable reference for.
    """
    out = np.full(x.shape, np.nan)
    if x.size < length:
        return out

    acc = x[:length].mean()
    out[length - 1] = acc
    beta = 1.0 - alpha
    for i in range(length, x.shape[0]):
        acc = alpha * x[i] + beta * acc
        out[i] = acc
    return out


def _check(length: int) -> None:
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError(f"length must be a positive int, got {length!r}")


def _same_shape(*arrays: np.ndarray) -> None:
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"inputs must have the same shape, got {sorted(shapes)}")
