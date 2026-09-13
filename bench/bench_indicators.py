"""Signal-pass cost: numpy reference vs numba hot path. Plan §2, §11."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.indicators import atr, ema, rsi  # noqa: E402


def synthetic_bars(n: int):
    i = np.arange(n, dtype=np.float64)
    r = np.sin(i * 12.9898) * 43758.5453
    d = (r - np.floor(r) - 0.5) * 2
    close = np.maximum(1.0, 2000.0 + np.cumsum(d))
    open_ = np.concatenate(([2000.0], close[:-1]))
    high = np.maximum(open_, close) + np.abs(d) * 0.5
    low = np.minimum(open_, close) - np.abs(d) * 0.5
    return open_, high, low, close


@njit(cache=True, fastmath=False)  # fastmath=False is load-bearing — plan §7
def ema_nb(x, length):
    out = np.full(x.shape[0], np.nan)
    acc = 0.0
    for i in range(length):
        acc += x[i]
    acc /= length
    out[length - 1] = acc
    a = 2.0 / (length + 1.0)
    b = 1.0 - a
    for i in range(length, x.shape[0]):
        acc = a * x[i] + b * acc
        out[i] = acc
    return out


@njit(cache=True, fastmath=False)
def rma_nb(x, length):
    out = np.full(x.shape[0], np.nan)
    acc = 0.0
    for i in range(length):
        acc += x[i]
    acc /= length
    out[length - 1] = acc
    a = 1.0 / length
    b = 1.0 - a
    for i in range(length, x.shape[0]):
        acc = a * x[i] + b * acc
        out[i] = acc
    return out


def main() -> None:
    print("=== PASS 1: signal computation (3 indicators) ===")
    print(f"{'bars':>10}  {'numpy ref':>12}  {'numba':>10}  {'speedup':>8}")
    for n in (100_000, 370_000, 1_000_000):
        _, high, low, close = synthetic_bars(n)

        t0 = time.perf_counter()
        ema(close, 20)
        rsi(close, 14)
        atr(high, low, close, 14)
        t_np = (time.perf_counter() - t0) * 1000

        ema_nb(close, 20)  # warm the JIT
        rma_nb(close, 14)
        t0 = time.perf_counter()
        ema_nb(close, 20)
        ema_nb(close, 50)
        rma_nb(close, 14)
        t_nb = (time.perf_counter() - t0) * 1000

        print(f"{n:>10,}  {t_np:>10.1f}ms  {t_nb:>8.2f}ms  {t_np / t_nb:>7.0f}x")

    print("\nPlan §11 reference: PineTS 0.9.33 was ~3,300ms at 370k bars.")


if __name__ == "__main__":
    main()
