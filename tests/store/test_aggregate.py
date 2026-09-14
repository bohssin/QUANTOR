"""Rolling bars up must equal building them from ticks. Plan §4.6.

This is what makes a large archive usable: stream the ticks **once** into M1,
cache that, and derive M5/M15/H1/H4/D1 from the cache. The whole scheme is only
sound if the derived bars are the same bars — otherwise the timeframe a strategy
was optimized on and the timeframe it is traded on quietly disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.store import bars_from_ticks
from engine.store.bars import aggregate_bars

COARSER = ["M5", "M15", "M30", "H1", "H4", "D1"]


def _ticks(n: int = 300_000, seed: int = 11):
    rng = np.random.default_rng(seed)
    ms = 1_609_722_000_000 + np.cumsum(rng.integers(1, 900, n).astype(np.int64))
    bid = 1900.0 + np.cumsum(rng.normal(0, 0.01, n))
    ask = bid + rng.uniform(0.10, 0.40, n)
    return ms, bid, ask


@pytest.mark.parametrize("timeframe", COARSER)
def test_aggregating_m1_equals_building_from_ticks(timeframe):
    ms, bid, ask = _ticks()
    direct = bars_from_ticks(ms, bid, ask, timeframe)
    rolled = aggregate_bars(bars_from_ticks(ms, bid, ask, "M1"), timeframe, "M1")

    np.testing.assert_array_equal(rolled["ms"], direct["ms"])
    np.testing.assert_array_equal(rolled["volume"], direct["volume"])
    for key in ("open", "high", "low", "close", "spread", "spread_max"):
        np.testing.assert_allclose(rolled[key], direct[key], rtol=0, atol=1e-9,
                                   err_msg=f"{key} at {timeframe}")


def test_spread_is_weighted_by_tick_count_not_a_mean_of_means():
    """A 900-tick bar and a 3-tick bar are not equal evidence about the spread."""
    base = 1_609_722_000_000
    bars = {
        "ms": np.array([base, base + 60_000], np.int64),
        "open": np.array([10.0, 10.0]), "high": np.array([10.0, 10.0]),
        "low": np.array([10.0, 10.0]), "close": np.array([10.0, 10.0]),
        "volume": np.array([900, 3], np.int64),
        "spread": np.array([0.10, 1.00]),          # the thin bar is wide
        "spread_max": np.array([0.10, 1.00]),
    }
    out = aggregate_bars(bars, "M5", "M1")
    expected = (0.10 * 900 + 1.00 * 3) / 903
    assert out["spread"][0] == pytest.approx(expected)
    assert out["spread"][0] < 0.2, "a mean of means would have said 0.55"
    assert out["spread_max"][0] == 1.00


def test_rolling_down_is_refused():
    ms, bid, ask = _ticks(10_000)
    h1 = bars_from_ticks(ms, bid, ask, "H1")
    with pytest.raises(ValueError, match="never split"):
        aggregate_bars(h1, "M15", "H1")


def test_same_timeframe_passes_through_unchanged():
    ms, bid, ask = _ticks(10_000)
    m15 = bars_from_ticks(ms, bid, ask, "M15")
    out = aggregate_bars(m15, "M15", "M15")
    for key in m15:
        np.testing.assert_array_equal(out[key], m15[key])


def test_empty_input_stays_empty():
    empty = bars_from_ticks(np.empty(0, np.int64), np.empty(0), np.empty(0), "M1")
    assert len(aggregate_bars(empty, "H1", "M1")["ms"]) == 0


def test_windows_with_no_ticks_produce_no_bar():
    """MT5 skips empty bars; inventing one manufactures an untradeable price."""
    base = 1_609_722_000_000
    ms = np.array([base, base + 60_000, base + 10 * 3_600_000], np.int64)
    bid = np.array([10.0, 11.0, 12.0])
    ask = bid + 0.2
    hourly = aggregate_bars(bars_from_ticks(ms, bid, ask, "M1"), "H1", "M1")
    assert len(hourly["ms"]) == 2, "the ten empty hours must not become bars"
