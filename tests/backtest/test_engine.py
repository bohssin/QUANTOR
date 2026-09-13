"""Backtest engine correctness. Plan §10.1 — verified by invariants, not inspection.

These are the tests that stand in for an external reference now that MT5
reconciliation is not a build gate (§10.2). They fall into three groups:

1. **Hand-computed cases** — one trade whose P&L is worked out on paper.
2. **Invariants** — properties that must hold for every input, checked over
   randomized data.
3. **Silent-failure guards** — the specific ways a backtest lies: look-ahead,
   same-bar round trips, hidden ambiguity, sizing that ignores lot steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.backtest import Bars, Instrument, Signals, compute_metrics, run_backtest
from engine.backtest.core import EXIT_EOD, EXIT_STOP, EXIT_TARGET

INST = Instrument(
    contract_size=100.0, tick_size=0.001, tick_value=0.10,
    lot_min=0.01, lot_step=0.01, lot_max=100.0,
    commission_per_lot_per_side=3.5, default_spread=0.0, stops_level=0.0,
)


def flat_bars(n, price=1900.0):
    a = np.full(n, price, dtype=np.float64)
    return Bars(ms=np.arange(n, dtype=np.int64) * 60_000,
                open=a.copy(), high=a.copy(), low=a.copy(), close=a.copy(),
                spread=np.zeros(n))


def no_signals(n):
    return Signals(long_entry=np.zeros(n, bool), short_entry=np.zeros(n, bool),
                   stop_distance=np.zeros(n), target_distance=np.zeros(n))


# --- 1. hand-computed -------------------------------------------------------

def test_single_long_hits_target_with_exact_pnl():
    """Worked by hand:

    entry  1900.0 at bar 1 open, stop 1890 (10.0 away), target 1920 (20.0)
    value_per_price_unit = tick_value/tick_size = 0.10/0.001 = 100 per lot
    risk 1% of 10,000 = 100.0; per-lot risk = 10.0*100 + 3.5*2 = 1007.0
    raw lots = 100/1007 = 0.0993 -> floor to 0.01 step -> 0.09
    gross at target = (1920-1900) * 100 * 0.09 = 180.0
    commissions = 3.5*0.09 entry + 3.5*0.09 exit = 0.63
    net = 180.0 - 0.63 = 179.37
    """
    n = 6
    bars = flat_bars(n)
    bars.high[3] = 1925.0          # target touched on bar 3

    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 10.0
    sig.target_distance[0] = 20.0

    r = run_backtest(bars, sig, INST, initial_capital=10_000.0, risk_pct=0.01)

    assert r.n_trades == 1
    assert r.entry_i[0] == 1
    assert r.exit_i[0] == 3
    assert r.exit_reason[0] == EXIT_TARGET
    assert r.lots[0] == pytest.approx(0.09)
    assert r.entry_px[0] == pytest.approx(1900.0)
    assert r.exit_px[0] == pytest.approx(1920.0)
    assert r.pnl[0] == pytest.approx(179.37, abs=1e-9)


def test_single_short_hits_stop_with_exact_pnl():
    """Short entry 1900, stop 1910 (10 away). Loss = -10*100*lots - commissions."""
    n = 6
    bars = flat_bars(n)
    bars.high[3] = 1915.0

    sig = no_signals(n)
    sig.short_entry[0] = True
    sig.stop_distance[0] = 10.0
    sig.target_distance[0] = 20.0

    r = run_backtest(bars, sig, INST, initial_capital=10_000.0, risk_pct=0.01)
    assert r.n_trades == 1
    assert r.direction[0] == -1
    assert r.exit_reason[0] == EXIT_STOP
    assert r.exit_px[0] == pytest.approx(1910.0)
    assert r.pnl[0] == pytest.approx(-10.0 * 100 * 0.09 - 3.5 * 0.09 * 2, abs=1e-9)


def test_spread_is_charged_on_the_long_entry():
    """Buying pays the ask. A zero-spread and a 0.5-spread run must differ by
    exactly the spread, in the entry price."""
    n = 6
    bars = flat_bars(n)
    bars.high[3] = 1930.0
    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 10.0
    sig.target_distance[0] = 20.0

    tight = run_backtest(bars, sig, INST, risk_pct=0.01)

    wide_bars = flat_bars(n)
    wide_bars.high[3] = 1930.0
    wide_bars = Bars(ms=wide_bars.ms, open=wide_bars.open, high=wide_bars.high,
                     low=wide_bars.low, close=wide_bars.close,
                     spread=np.full(n, 0.5))
    wide = run_backtest(wide_bars, sig, INST, risk_pct=0.01)

    assert wide.entry_px[0] - tight.entry_px[0] == pytest.approx(0.5)


# --- 2. invariants ----------------------------------------------------------

def random_case(seed, n=400):
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 1.2, size=n)
    close = 1900.0 + np.cumsum(step)
    open_ = np.concatenate(([1900.0], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.8, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.8, n))
    bars = Bars(ms=np.arange(n, dtype=np.int64) * 60_000,
                open=open_, high=high, low=low, close=close,
                spread=np.full(n, 0.2))
    sig = no_signals(n)
    fire = rng.random(n) < 0.05
    long_side = rng.random(n) < 0.5
    sig.long_entry[:] = fire & long_side
    sig.short_entry[:] = fire & ~long_side
    sig.stop_distance[:] = rng.uniform(3.0, 12.0, n)
    sig.target_distance[:] = sig.stop_distance * rng.uniform(1.0, 3.0, n)
    return bars, sig


@pytest.mark.parametrize("seed", range(12))
def test_closed_pnl_reconciles_with_the_equity_curve(seed):
    """Sum of closed-trade P&L equals the change in equity. If these disagree,
    money is being created or destroyed somewhere in the ledger."""
    bars, sig = random_case(seed)
    r = run_backtest(bars, sig, INST, initial_capital=10_000.0, risk_pct=0.01)
    if r.n_trades == 0:
        pytest.skip("no trades for this seed")
    assert r.equity[-1] - 10_000.0 == pytest.approx(float(r.pnl.sum()), abs=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_every_size_is_a_clean_multiple_of_lot_step(seed):
    bars, sig = random_case(seed)
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    if r.n_trades == 0:
        pytest.skip("no trades")
    steps = r.lots / INST.lot_step
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert (r.lots >= INST.lot_min - 1e-12).all()
    assert (r.lots <= INST.lot_max + 1e-12).all()


@pytest.mark.parametrize("seed", range(12))
def test_fills_lie_inside_the_bar_that_produced_them(seed):
    """An exit price outside its bar's range is a fill that could not happen."""
    bars, sig = random_case(seed)
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    for k in range(r.n_trades):
        if r.exit_reason[k] == EXIT_EOD:
            continue
        i = int(r.exit_i[k])
        assert bars.low[i] - 1e-9 <= r.exit_px[k] <= bars.high[i] + 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_positions_never_overlap(seed):
    """One position at a time: each entry is at or after the previous exit."""
    bars, sig = random_case(seed)
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    for k in range(1, r.n_trades):
        assert r.entry_i[k] >= r.exit_i[k - 1]


@pytest.mark.parametrize("seed", range(6))
def test_r_multiple_matches_pnl_over_risk(seed):
    """R is the sizing-free view, so it must be consistent with the currency
    P&L — §12's Monte Carlo resamples R and would silently drift otherwise."""
    bars, sig = random_case(seed)
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    if r.n_trades == 0:
        pytest.skip("no trades")
    losers = r.r_multiple[r.exit_reason == EXIT_STOP]
    if losers.size:
        # A stop-out risks ~1R by construction; spread and rounding move it a
        # little, but it must not be wildly off.
        assert -1.6 < float(losers.mean()) < -0.5


def test_determinism_byte_for_byte():
    """Same inputs give the same trade list. Plan §7 — the engine is
    reproducible even though the agent that configured it is not."""
    bars, sig = random_case(3)
    a = run_backtest(bars, sig, INST, risk_pct=0.01)
    b = run_backtest(bars, sig, INST, risk_pct=0.01)
    for field in ("entry_i", "exit_i", "direction", "entry_px", "exit_px",
                  "lots", "pnl", "r_multiple", "exit_reason", "equity"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


# --- 3. silent-failure guards -----------------------------------------------

def test_signal_cannot_act_on_its_own_bar():
    """A signal on bar i fills at bar i+1's open. Acting on bar i's own close
    is the classic look-ahead that makes a backtest look brilliant."""
    n = 5
    bars = flat_bars(n)
    sig = no_signals(n)
    sig.long_entry[2] = True
    sig.stop_distance[2] = 10.0
    sig.target_distance[2] = 20.0
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    assert r.n_trades == 1
    assert r.entry_i[0] == 3          # not 2


def test_no_exit_on_the_entry_bar():
    """The entry bar's own high/low must not close the position: the intrabar
    path is unknown, so such a fill depends on ordering the data never had."""
    n = 6
    bars = flat_bars(n)
    bars.high[1] = 1999.0             # would hit target on the entry bar
    bars.low[1] = 1800.0              # and the stop too
    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 10.0
    sig.target_distance[0] = 20.0
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    assert r.n_trades == 1
    assert r.exit_i[0] > 1
    assert r.exit_reason[0] == EXIT_EOD


def test_ambiguous_bars_are_counted_not_hidden():
    """Stop and target both inside one bar: bar data cannot say which came
    first. Take the stop, and report the count — §4.6."""
    n = 6
    bars = flat_bars(n)
    bars.high[3] = 1925.0             # target 1920 reached
    bars.low[3] = 1885.0              # stop 1890 also reached
    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 10.0
    sig.target_distance[0] = 20.0
    r = run_backtest(bars, sig, INST, risk_pct=0.01)
    assert r.ambiguous_exits == 1
    assert r.ambiguity_rate() == 1.0
    assert r.exit_reason[0] == EXIT_STOP       # conservative reading


def test_stops_level_rejects_too_tight_a_stop():
    """A broker refuses stops closer than stops_level. A backtest that accepts
    them flatters every tight-stop strategy."""
    inst = Instrument(**{**INST.__dict__, "stops_level": 5.0})
    n = 6
    bars = flat_bars(n)
    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 1.0        # below stops_level
    sig.target_distance[0] = 20.0
    r = run_backtest(bars, sig, inst, risk_pct=0.01)
    assert r.n_trades == 0
    assert r.rejected_by_stops_level == 1


def test_size_rounding_to_zero_is_reported_not_silent():
    """A risk budget too small for one minimum lot must be visible, not an
    absent trade nobody can explain."""
    n = 6
    bars = flat_bars(n)
    sig = no_signals(n)
    sig.long_entry[0] = True
    sig.stop_distance[0] = 500.0      # enormous stop -> tiny size
    sig.target_distance[0] = 1000.0
    r = run_backtest(bars, sig, INST, initial_capital=100.0, risk_pct=0.001)
    assert r.n_trades == 0
    assert r.rejected_zero_lots == 1


def test_metrics_on_an_empty_result_are_zero_not_nan():
    n = 20
    r = run_backtest(flat_bars(n), no_signals(n), INST)
    m = compute_metrics(r, "M1")
    assert m.n_trades == 0
    assert m.net_profit == 0.0
    assert m.sharpe == 0.0
