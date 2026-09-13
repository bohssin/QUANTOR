"""End-to-end slice: signal -> backtest -> optimize -> walk-forward. Plan §20 step 0.

This is the thin vertical slice the build order asks for first, and it exists to
answer one question with evidence rather than assertion: does the testing and
optimizing design actually hold together in Python?

    python3 bench/demo_loop.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import Bars, Instrument, Signals, compute_metrics, run_backtest  # noqa: E402
from engine.indicators import atr_fast, ema_fast  # noqa: E402
from engine.optimize import FloatParam, IntParam, ParamSpec, grid_search  # noqa: E402
from engine.validate import rolling_folds, walk_forward  # noqa: E402

TIMEFRAME = "M15"
INSTRUMENT = Instrument(
    contract_size=100.0, tick_size=0.001, tick_value=0.10,
    lot_min=0.01, lot_step=0.01, lot_max=100.0,
    commission_per_lot_per_side=3.5, default_spread=0.30, stops_level=0.0,
)


def synthetic_bars(n: int, seed: int = 11) -> Bars:
    """A trending-then-ranging series. Not a market; enough to exercise the loop."""
    rng = np.random.default_rng(seed)
    drift = np.sin(np.arange(n) / 900.0) * 0.04
    step = rng.normal(0, 0.9, size=n) + drift
    close = 1900.0 + np.cumsum(step)
    open_ = np.concatenate(([1900.0], close[:-1]))
    wick = np.abs(rng.normal(0, 0.6, size=n))
    return Bars(
        ms=np.arange(n, dtype=np.int64) * 900_000,
        open=open_, high=np.maximum(open_, close) + wick,
        low=np.minimum(open_, close) - wick, close=close,
        spread=np.full(n, 0.30),
    )


def signal_block(bars: Bars, p: dict) -> Signals:
    """THE GENERATED SLOT (plan §9).

    Computes levels and arm conditions. No order calls, no position state, no
    forward references — every value on bar i uses only bars <= i, and the
    engine acts on it at bar i+1's open.

    Uses the numba indicators: they are pinned to the numpy reference, which is
    pinned to the oracle (§8.2), and the reference's Python-loop recursions were
    96% of an evaluation's cost.
    """
    fast = ema_fast(bars.close, p["fast"])
    slow = ema_fast(bars.close, p["slow"])
    vol = atr_fast(bars.high, bars.low, bars.close, p["atr_len"])

    above = fast > slow
    cross_up = np.zeros(len(bars), bool)
    cross_dn = np.zeros(len(bars), bool)
    cross_up[1:] = above[1:] & ~above[:-1]
    cross_dn[1:] = ~above[1:] & above[:-1]

    warm = np.isnan(fast) | np.isnan(slow) | np.isnan(vol)
    cross_up &= ~warm
    cross_dn &= ~warm

    stop = np.nan_to_num(vol, nan=0.0) * p["stop_mult"]
    return Signals(
        long_entry=cross_up, short_entry=cross_dn,
        stop_distance=stop, target_distance=stop * p["rr"],
    )


def evaluate(bars: Bars, params: dict):
    result = run_backtest(bars, signal_block(bars, params), INSTRUMENT,
                          initial_capital=10_000.0, risk_pct=0.01)
    return compute_metrics(result, TIMEFRAME), result


def main() -> None:
    bars = synthetic_bars(40_000)
    print(f"bars: {len(bars):,} @ {TIMEFRAME}\n")

    base = {"fast": 20, "slow": 50, "atr_len": 14, "stop_mult": 2.0, "rr": 2.0}

    # --- 1. one backtest ---
    print("=" * 72)
    print("1. SINGLE BACKTEST")
    evaluate(bars, base)                                   # warm the JIT
    t0 = time.perf_counter()
    metrics, result = evaluate(bars, base)
    ms = (time.perf_counter() - t0) * 1000
    print(f"   {ms:.1f}ms   params={base}")
    print(f"   trades={metrics.n_trades}  net={metrics.net_profit:,.2f}  "
          f"PF={metrics.profit_factor:.2f}  expectancy={metrics.expectancy_r:+.3f}R")
    print(f"   maxDD={metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1%})  "
          f"sharpe={metrics.sharpe:.2f}  SQN={metrics.sqn:.2f}")
    print(f"   ambiguous exits={result.ambiguous_exits} "
          f"({result.ambiguity_rate():.1%} of trades)  "
          f"size-rejected={result.rejected_zero_lots}")
    reconcile = result.equity[-1] - 10_000.0 - float(result.pnl.sum())
    print(f"   ledger check |equity delta - sum(pnl)| = {abs(reconcile):.2e}")

    # --- 2. grid search ---
    print("\n" + "=" * 72)
    print("2. GRID SEARCH  (objective: return_over_maxdd, min 30 trades)")
    spec = ParamSpec(
        params={
            "fast": IntParam(10, 30, 10),
            "slow": IntParam(40, 80, 20),
            "stop_mult": FloatParam(1.5, 3.0, 0.5),
            "rr": FloatParam(1.5, 3.0, 0.5),
        },
        fixed={"atr_len": 14},
    )
    print(f"   space: {spec.grid_size(10):,} configurations")
    t0 = time.perf_counter()
    search = grid_search(spec, lambda p: evaluate(bars, p)[0], min_trades=30)
    total = time.perf_counter() - t0
    print(f"   {search.comparisons} evals in {total:.2f}s "
          f"({total / max(search.comparisons, 1) * 1000:.1f}ms/eval)")
    print("\n   top 5 (plateau, not just the peak):")
    print(f"   {'fast':>5} {'slow':>5} {'stop':>6} {'rr':>5} {'score':>9} "
          f"{'net':>10} {'maxDD':>9} {'trades':>7}")
    for e in search.top_k(5):
        print(f"   {e.params['fast']:>5} {e.params['slow']:>5} "
              f"{e.params['stop_mult']:>6.1f} {e.params['rr']:>5.1f} "
              f"{e.score:>9.3f} {e.metrics['net_profit']:>10,.0f} "
              f"{e.metrics['max_drawdown']:>9,.0f} {e.metrics['n_trades']:>7}")

    # --- 3. walk-forward ---
    print("\n" + "=" * 72)
    print("3. WALK-FORWARD  (a full search inside every train fold)")
    folds = rolling_folds(len(bars), train=12_000, test=4_000, step=4_000)
    print(f"   {len(folds)} folds, train=12,000 test=4,000 bars")

    small = ParamSpec(
        params={"fast": IntParam(10, 30, 10), "stop_mult": FloatParam(1.5, 3.0, 0.5)},
        fixed={"slow": 50, "atr_len": 14, "rr": 2.0},
    )

    def search_fold(idx, tr_start, tr_end):
        window = bars.slice(tr_start, tr_end)
        res = grid_search(small, lambda p: evaluate(window, p)[0], min_trades=10)
        best = res.best
        return best.params, best.score, res.comparisons

    def test_fold(params, te_start, te_end):
        window = bars.slice(te_start, te_end)
        m, _ = evaluate(window, params)
        return (m.net_profit / m.max_drawdown if m.max_drawdown > 0 else float("-inf")), \
               m.as_dict()

    t0 = time.perf_counter()
    wf = walk_forward(folds, search_fold, test_fold)
    total = time.perf_counter() - t0

    print(f"   {total:.2f}s, {wf.total_comparisons} total comparisons\n")
    print(f"   {'fold':>4} {'fast':>5} {'stop':>6} {'train':>9} {'test':>9} {'trades':>7}")
    for i, f in enumerate(folds):
        p = wf.chosen_params[i]
        print(f"   {i:>4} {p['fast']:>5} {p['stop_mult']:>6.1f} "
              f"{wf.train_scores[i]:>9.3f} {wf.test_scores[i]:>9.3f} "
              f"{wf.test_metrics[i].get('n_trades', 0):>7}")

    print(f"\n   mean out-of-sample score : {wf.mean_test_score:.3f}")
    print(f"   walk-forward efficiency  : {wf.efficiency():.2f}  "
          "(OOS / IS; near 1.0 means it generalized)")
    print("\n   NOTE: synthetic data. These numbers say the machinery works,")
    print("         not that the strategy does.")


if __name__ == "__main__":
    main()
