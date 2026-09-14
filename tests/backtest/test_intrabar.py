"""Finer bars settle what a single bar cannot. Plan §6, §10.

When a signal bar's range contains both the stop and the target, that bar does
not record which was touched first. The engine assumes the stop — safe, and
wrong often enough to matter. Given finer bars underneath it can stop assuming.

These tests are built around hand-made bars where the answer is known by
construction, because "the number changed" is not evidence that it changed to
the right one.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.backtest import Bars, Instrument, Signals, run_backtest
from engine.backtest.core import EXIT_STOP, EXIT_TARGET, align_subbars

M1 = 60_000
BASE = 1_609_722_000_000


def _inst() -> Instrument:
    # No costs, so the trade's P&L is purely the level that was hit.
    return Instrument(commission_per_lot_per_side=0.0, default_spread=0.0)


def _m1(highs, lows, opens=None, closes=None) -> Bars:
    n = len(highs)
    o = np.asarray(opens if opens is not None else [100.0] * n, float)
    c = np.asarray(closes if closes is not None else o, float)
    return Bars(ms=BASE + np.arange(n, dtype=np.int64) * M1,
                open=o, high=np.asarray(highs, float), low=np.asarray(lows, float),
                close=c, spread=np.zeros(n))


def _s1(rows) -> Bars:
    """rows: (offset_seconds, high, low)."""
    ms = np.array([BASE + int(t) * 1000 for t, _, _ in rows], np.int64)
    high = np.array([h for _, h, _ in rows], float)
    low = np.array([l for _, _, l in rows], float)
    return Bars(ms=ms, open=high, high=high, low=low, close=low,
                spread=np.zeros(len(rows)))


def _signals(n, entry_bar=0, stop=1.0, target=1.0) -> Signals:
    long_entry = np.zeros(n, bool)
    long_entry[entry_bar] = True
    return Signals(long_entry=long_entry, short_entry=np.zeros(n, bool),
                   stop_distance=np.full(n, stop), target_distance=np.full(n, target))


# --- the case the whole feature exists for -----------------------------------

def test_finer_bars_decide_the_target_when_it_came_first():
    """Bar 2 contains both levels. The S1 bars say the target came first."""
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    # Entry fills at bar 1's open (100), stop 99, target 101.
    sub = _s1([
        (60, 100.2, 100.0),      # bar 1, nothing
        (120, 101.5, 100.0),     # bar 2 second 0: target first
        (135, 101.5, 98.0),      # then the stop
    ])
    out = run_backtest(bars, _signals(4), _inst(),
                       subbars=sub, subbar_timeframe="S1", bar_ms=M1)

    assert out.n_trades == 1
    assert out.exit_reason[0] == EXIT_TARGET
    assert out.exit_px[0] == pytest.approx(101.0)
    assert out.resolved_intrabar == 1
    assert out.ambiguous_exits == 0
    assert out.intrabar_timeframe == "S1"


def test_finer_bars_confirm_the_stop_when_it_came_first():
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    sub = _s1([
        (60, 100.2, 100.0),
        (120, 100.1, 98.0),      # bar 2 second 0: stop first
        (135, 101.5, 98.0),
    ])
    out = run_backtest(bars, _signals(4), _inst(),
                       subbars=sub, subbar_timeframe="S1", bar_ms=M1)
    assert out.exit_reason[0] == EXIT_STOP
    assert out.resolved_intrabar == 1
    assert out.ambiguous_exits == 0


def test_without_finer_bars_the_stop_is_assumed_and_counted():
    """The pessimistic reading, and it must be reported as an assumption."""
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    out = run_backtest(bars, _signals(4), _inst())

    assert out.exit_reason[0] == EXIT_STOP
    assert out.ambiguous_exits == 1
    assert out.resolved_intrabar == 0
    assert out.intrabar_timeframe == ""
    assert out.intrabar_resolution_rate() == 0.0


def test_the_same_bars_give_opposite_answers_on_opposite_orderings():
    """The whole claim: the finer data, not the engine, decides."""
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    target_first = _s1([(60, 100.0, 100.0), (120, 101.5, 100.0), (150, 101.5, 98.0)])
    stop_first = _s1([(60, 100.0, 100.0), (120, 100.0, 98.0), (150, 101.5, 98.0)])

    a = run_backtest(bars, _signals(4), _inst(), subbars=target_first,
                     subbar_timeframe="S1", bar_ms=M1)
    b = run_backtest(bars, _signals(4), _inst(), subbars=stop_first,
                     subbar_timeframe="S1", bar_ms=M1)

    assert a.exit_reason[0] == EXIT_TARGET
    assert b.exit_reason[0] == EXIT_STOP
    assert a.pnl[0] > 0 > b.pnl[0]


def test_a_finer_bar_holding_both_levels_stays_ambiguous():
    """S1 is finer, not infinitely fine. One second can still hold both."""
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    sub = _s1([(60, 100.0, 100.0), (120, 101.5, 98.0)])   # one second, both levels

    out = run_backtest(bars, _signals(4), _inst(), subbars=sub,
                       subbar_timeframe="S1", bar_ms=M1)
    assert out.exit_reason[0] == EXIT_STOP, "unresolved must stay pessimistic"
    assert out.ambiguous_exits == 1
    assert out.resolved_intrabar == 0


def test_a_gap_in_the_finer_data_falls_back_to_the_assumption():
    """No S1 bars under this M1 bar — the honest answer is 'still unknown'."""
    bars = _m1(highs=[100, 100, 102, 100], lows=[100, 100, 98, 100])
    sub = _s1([(0, 100.0, 100.0), (30, 100.0, 100.0)])    # only under bar 0

    out = run_backtest(bars, _signals(4), _inst(), subbars=sub,
                       subbar_timeframe="S1", bar_ms=M1)
    assert out.exit_reason[0] == EXIT_STOP
    assert out.ambiguous_exits == 1
    assert out.resolved_intrabar == 0


# --- it must change nothing else ---------------------------------------------

def test_unambiguous_trades_are_untouched_by_finer_bars():
    """Only the both-hit branch may differ. Everything else is bit-identical."""
    rng = np.random.default_rng(4)
    n = 3_000
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    high, low = close + 0.08, close - 0.08
    bars = Bars(ms=BASE + np.arange(n, dtype=np.int64) * M1,
                open=close, high=high, low=low, close=close, spread=np.zeros(n))
    # Wide levels: no bar can contain both, so nothing is ambiguous.
    long_entry = np.zeros(n, bool)
    long_entry[::11] = True
    sig = Signals(long_entry=long_entry, short_entry=np.zeros(n, bool),
                  stop_distance=np.full(n, 2.0), target_distance=np.full(n, 2.0))

    sub_ms = np.repeat(bars.ms, 4) + np.tile(np.array([0, 15, 30, 45]) * 1000, n)
    sub = Bars(ms=sub_ms, open=np.repeat(close, 4), high=np.repeat(high, 4),
               low=np.repeat(low, 4), close=np.repeat(close, 4),
               spread=np.zeros(n * 4))

    a = run_backtest(bars, sig, _inst())
    b = run_backtest(bars, sig, _inst(), subbars=sub, subbar_timeframe="S1", bar_ms=M1)

    assert a.ambiguous_exits == 0 and b.resolved_intrabar == 0
    np.testing.assert_array_equal(a.entry_i, b.entry_i)
    np.testing.assert_array_equal(a.exit_i, b.exit_i)
    np.testing.assert_array_equal(a.exit_reason, b.exit_reason)
    np.testing.assert_allclose(a.pnl, b.pnl, rtol=0, atol=0)


def test_subbars_without_bar_ms_is_refused():
    bars = _m1(highs=[100, 101], lows=[99, 100])
    sub = _s1([(0, 100.0, 100.0)])
    with pytest.raises(ValueError, match="bar_ms"):
        run_backtest(bars, _signals(2), _inst(), subbars=sub)


# --- alignment ----------------------------------------------------------------

def test_alignment_puts_each_finer_bar_under_exactly_one_signal_bar():
    bars = _m1(highs=[1, 1, 1], lows=[1, 1, 1])
    sub = _s1([(0, 1, 1), (30, 1, 1), (59, 1, 1),
               (60, 1, 1), (119, 1, 1),
               (120, 1, 1)])
    start, end = align_subbars(bars, sub, M1)

    assert list(start) == [0, 3, 5]
    assert list(end) == [3, 5, 6]
    # Every finer bar is claimed once and only once.
    counts = np.zeros(len(sub.ms), int)
    for a, b in zip(start, end):
        counts[a:b] += 1
    assert (counts == 1).all()


def test_alignment_leaves_an_empty_range_where_finer_data_is_missing():
    bars = _m1(highs=[1, 1, 1], lows=[1, 1, 1])
    sub = _s1([(0, 1, 1), (120, 1, 1)])          # nothing under bar 1
    start, end = align_subbars(bars, sub, M1)
    assert start[1] == end[1], "bar 1 must get an empty range, not a borrowed one"
