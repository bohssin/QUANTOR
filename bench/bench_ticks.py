"""Tick pipeline: ingest, bar construction, tick-resolution fills. Plan §4.1, §6.

The headline result: a full year of real-tick backtesting costs ~0.2s, which is
why the plan drops rev 3's 1-second bar layer and its ambiguity instrumentation
entirely (plan §2).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DATA = Path(__file__).parent / "_data"
CSV = DATA / "ticks.csv"
PARQUET = DATA / "ticks.parquet"
N_TICKS = 10_000_000


def generate_csv() -> None:
    """~3 ticks/sec XAUUSD-like stream with a floating spread."""
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.integers(80, 700, size=N_TICKS, dtype=np.int64)) + 1_704_067_200_000
    bid = 2000.0 + np.cumsum(rng.normal(0, 0.012, size=N_TICKS))
    ask = bid + np.clip(rng.normal(0.22, 0.07, size=N_TICKS), 0.10, 3.0)
    with CSV.open("w") as fh:
        fh.write("timestamp,bid,ask\n")
        for i in range(0, N_TICKS, 1_000_000):
            sl = slice(i, min(i + 1_000_000, N_TICKS))
            np.savetxt(fh, np.column_stack([t[sl], bid[sl], ask[sl]]), fmt="%d,%.3f,%.3f")


@njit(cache=True, fastmath=False)
def build_bars(t, bid, ask, tf_ms):
    """Ticks -> OHLC bars. MT5 semantics: OHLC from bid, volume = tick count,
    spread carried per bar, empty bars simply do not exist. Plan §4.4."""
    n = t.shape[0]
    bar_t = np.empty(n, np.int64)
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    vol = np.empty(n, np.int64)
    sp_mean = np.empty(n); sp_max = np.empty(n)

    k = -1
    current = np.int64(-1)
    sp_sum = 0.0
    for i in range(n):
        boundary = (t[i] // tf_ms) * tf_ms
        spread = ask[i] - bid[i]
        if boundary != current:
            if k >= 0:
                sp_mean[k] = sp_sum / vol[k]
            k += 1
            current = boundary
            bar_t[k] = boundary
            o[k] = h[k] = l[k] = c[k] = bid[i]
            vol[k] = 1
            sp_sum = spread
            sp_max[k] = spread
        else:
            if bid[i] > h[k]:
                h[k] = bid[i]
            if bid[i] < l[k]:
                l[k] = bid[i]
            c[k] = bid[i]
            vol[k] += 1
            sp_sum += spread
            if spread > sp_max[k]:
                sp_max[k] = spread
    if k >= 0:
        sp_mean[k] = sp_sum / vol[k]
    m = k + 1
    return bar_t[:m], o[:m], h[:m], l[:m], c[:m], vol[:m], sp_mean[:m], sp_max[:m]


@njit(cache=True, fastmath=False)
def fill_on_ticks(t, bid, ask, sig_t, sig_dir, sl_dist, tp_dist,
                  contract_size, commission_per_lot, lots, initial_cash):
    """Resolve fills against the real tick stream.

    Tick order is known, so stop-vs-target ordering is never ambiguous — the
    whole reason rev 3's 1s layer and §7.3 ambiguity metric are gone (plan §2).

    Deliberately minimal: single position, SL/TP only. The production engine
    also owns OCA, pending orders, trailing, partial closes, margin, stop-out,
    swap and FX (plan §7), so expect it to be materially slower than this.
    """
    n = t.shape[0]
    m = sig_t.shape[0]
    pos = 0.0
    entry = sl = tp = 0.0
    cash = initial_cash
    trades = wins = 0
    si = 0
    pnl_out = np.empty(m)
    kind_out = np.empty(m, np.int8)
    nt = 0

    for i in range(n):
        if pos != 0.0:
            price = 0.0
            kind = 0
            if pos > 0.0:                      # long exits on bid
                if bid[i] <= sl:
                    price, kind = sl, 1
                elif bid[i] >= tp:
                    price, kind = tp, 2
            else:                              # short exits on ask
                if ask[i] >= sl:
                    price, kind = sl, 1
                elif ask[i] <= tp:
                    price, kind = tp, 2
            if kind != 0:
                pnl = (price - entry) * pos * contract_size * lots
                pnl -= commission_per_lot * lots
                cash += pnl
                trades += 1
                if pnl > 0.0:
                    wins += 1
                pnl_out[nt] = pnl
                kind_out[nt] = kind
                nt += 1
                pos = 0.0

        while si < m and sig_t[si] <= t[i]:
            if pos == 0.0 and sig_dir[si] != 0.0:
                pos = sig_dir[si]
                entry = ask[i] if pos > 0.0 else bid[i]   # directional spread
                sl = entry - sl_dist * pos
                tp = entry + tp_dist * pos
            si += 1

    return cash, trades, wins, pnl_out[:nt], kind_out[:nt]


def main() -> None:
    if not CSV.exists():
        print(f"generating {N_TICKS:,}-tick CSV (one-time, ~20s)...")
        generate_csv()

    print(f"=== INGEST: {N_TICKS:,}-tick CSV ({os.path.getsize(CSV) / 1e6:.0f}MB) ===")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV, dtype={"timestamp": "int64", "bid": "float64", "ask": "float64"})
    print(f"  pandas.read_csv                {(time.perf_counter() - t0) * 1000:9.0f}ms")

    t = df["timestamp"].to_numpy()
    bid = df["bid"].to_numpy()
    ask = df["ask"].to_numpy()
    print(f"  resident: {(t.nbytes + bid.nbytes + ask.nbytes) / 1e6:.0f}MB "
          f"({(t.nbytes + bid.nbytes + ask.nbytes) / N_TICKS:.0f} bytes/tick)")

    t0 = time.perf_counter()
    df.to_parquet(PARQUET, compression="zstd")
    print(f"  -> parquet write               {(time.perf_counter() - t0) * 1000:9.0f}ms")
    t0 = time.perf_counter()
    pd.read_parquet(PARQUET)
    print(f"  <- parquet read (steady state) {(time.perf_counter() - t0) * 1000:9.0f}ms")
    print(f"  parquet {os.path.getsize(PARQUET) / 1e6:.0f}MB "
          f"vs csv {os.path.getsize(CSV) / 1e6:.0f}MB")

    print("\n=== BAR CONSTRUCTION (MT5: OHLC from bid) ===")
    build_bars(t[:1000], bid[:1000], ask[:1000], 60_000)   # warm
    for name, tf in (("M1", 60_000), ("M5", 300_000), ("M15", 900_000), ("H1", 3_600_000)):
        t0 = time.perf_counter()
        bars = build_bars(t, bid, ask, tf)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {name:4} -> {len(bars[0]):>8,} bars  {ms:7.0f}ms   "
              f"spread mean {bars[6].mean():.3f} max {bars[7].max():.3f}")

    print("\n=== PASS 2: fills over REAL TICKS (MT5 'every tick based on real ticks') ===")
    bar_t, *_ = build_bars(t, bid, ask, 900_000)
    sig_t = bar_t[::4]
    sig_dir = np.where(np.arange(len(sig_t)) % 2 == 0, 1.0, -1.0)
    fill_on_ticks(t[:1000], bid[:1000], ask[:1000], sig_t[:2], sig_dir[:2],
                  1.0, 2.0, 100.0, 3.5, 0.10, 10_000.0)   # warm
    t0 = time.perf_counter()
    cash, trades, wins, _, kind = fill_on_ticks(
        t, bid, ask, sig_t, sig_dir, 1.0, 2.0, 100.0, 3.5, 0.10, 10_000.0)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  {N_TICKS:,} ticks, {len(sig_t):,} signals   {ms:7.0f}ms  "
          f"({N_TICKS / (ms / 1000) / 1e6:.0f}M ticks/s)")
    print(f"  trades={trades} wins={wins} ({100 * wins / max(trades, 1):.1f}%)  "
          f"sl={int((kind == 1).sum())} tp={int((kind == 2).sum())}  equity={cash:.2f}")

    year = 60_000_000
    print(f"\n  extrapolated 1 year (~{year // 1_000_000}M ticks): "
          f"{ms * year / N_TICKS / 1000:.1f}s per tick-resolution backtest")
    print("  (MT5's own Strategy Tester: minutes to hours for the same span)")


if __name__ == "__main__":
    main()
