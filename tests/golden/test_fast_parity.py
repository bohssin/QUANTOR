"""The numba fast path must match the numpy reference. Plan §8.2.

`wilder.py` is pinned to the PineTS oracle by `test_indicator_parity.py`. These
tests pin `fast.py` to `wilder.py`, so the compiled path used by every
optimizer evaluation inherits that guarantee transitively.

Without this, the fast path is an unverified reimplementation sitting on the
hottest code path in the system — the worst possible place for one.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.indicators import (
    atr, atr_fast, ema, ema_fast, rma, rma_fast,
    rsi, rsi_fast, sma, sma_fast, true_range, true_range_fast,
)

TOL = 1e-9


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(17)
    close = 1900.0 + np.cumsum(rng.normal(0, 1.1, size=20_000))
    wick = np.abs(rng.normal(0, 0.7, size=20_000))
    return close, close + wick, close - wick


def _same(name, fast_out, ref_out):
    f_nan, r_nan = np.isnan(fast_out), np.isnan(ref_out)
    assert np.array_equal(f_nan, r_nan), f"{name}: warmup NaN positions differ"
    finite = ~r_nan
    diff = np.abs(fast_out[finite] - ref_out[finite])
    assert diff.max() < TOL, (
        f"{name}: max |fast - reference| = {diff.max():.3e} at bar "
        f"{int(np.argmax(diff))}"
    )


@pytest.mark.parametrize("length", [5, 14, 20, 50, 200])
def test_ema_parity(series, length):
    close, _, _ = series
    _same("ema", ema_fast(close, length), ema(close, length))


@pytest.mark.parametrize("length", [5, 14, 20, 50, 200])
def test_rma_parity(series, length):
    close, _, _ = series
    _same("rma", rma_fast(close, length), rma(close, length))


@pytest.mark.parametrize("length", [5, 14, 20, 50])
def test_sma_parity(series, length):
    close, _, _ = series
    _same("sma", sma_fast(close, length), sma(close, length))


@pytest.mark.parametrize("length", [7, 14, 21])
def test_rsi_parity(series, length):
    close, _, _ = series
    _same("rsi", rsi_fast(close, length), rsi(close, length))


def test_true_range_parity(series):
    close, high, low = series
    _same("true_range", true_range_fast(high, low, close), true_range(high, low, close))


@pytest.mark.parametrize("length", [7, 14, 21])
def test_atr_parity(series, length):
    close, high, low = series
    _same("atr", atr_fast(high, low, close, length), atr(high, low, close, length))


def test_fast_path_is_deterministic(series):
    """fastmath=False and no parallel reductions — plan §7. Two calls must be
    bit-identical, or the engine is not reproducible."""
    close, high, low = series
    for fn, args in ((ema_fast, (close, 20)), (rma_fast, (close, 14)),
                     (atr_fast, (high, low, close, 14))):
        np.testing.assert_array_equal(fn(*args), fn(*args))


def test_short_input_returns_all_nan_not_garbage(series):
    close, _, _ = series
    short = close[:5]
    assert np.isnan(ema_fast(short, 20)).all()
    assert np.isnan(rma_fast(short, 20)).all()
