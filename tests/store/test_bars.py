"""Bar construction from ticks. Plan §4.4 — MT5 semantics."""

from __future__ import annotations

import numpy as np
import pytest

from engine.store import bars_from_ticks, build_bars, read_csv, SourceSpec

TICKS = """timestamp,bidPrice,askPrice
2021-01-04 01:00:00.100,1900.000,1900.300
2021-01-04 01:00:30.000,1901.500,1901.900
2021-01-04 01:00:45.000,1899.000,1899.200
2021-01-04 01:00:59.999,1900.500,1900.700
2021-01-04 01:01:00.000,1902.000,1902.400
2021-01-04 01:01:30.000,1903.000,1903.200
2021-01-04 01:05:00.000,1890.000,1890.500
"""


def _ticks():
    ms = np.array([0, 30_000, 45_000, 59_999, 60_000, 90_000, 300_000], dtype=np.int64)
    bid = np.array([1900.0, 1901.5, 1899.0, 1900.5, 1902.0, 1903.0, 1890.0])
    ask = bid + np.array([0.3, 0.4, 0.2, 0.2, 0.4, 0.2, 0.5])
    return ms, bid, ask


def test_ohlc_comes_from_bid_and_is_correct():
    ms, bid, ask = _ticks()
    b = bars_from_ticks(ms, bid, ask, "M1")
    # first minute: ticks at 0, 30s, 45s, 59.999s
    assert b["open"][0] == 1900.0
    assert b["high"][0] == 1901.5
    assert b["low"][0] == 1899.0
    assert b["close"][0] == 1900.5


def test_volume_is_tick_count():
    ms, bid, ask = _ticks()
    b = bars_from_ticks(ms, bid, ask, "M1")
    assert b["volume"][0] == 4
    assert b["volume"][1] == 2


def test_empty_minutes_do_not_exist():
    """Ticks jump from 01:01:30 to 01:05:00. MT5 skips the gap; so do we —
    forward-filling invents a price that could not be traded."""
    ms, bid, ask = _ticks()
    b = bars_from_ticks(ms, bid, ask, "M1")
    assert len(b["ms"]) == 3                      # not 6
    assert b["ms"].tolist() == [0, 60_000, 300_000]


def test_spread_is_carried_per_bar():
    ms, bid, ask = _ticks()
    b = bars_from_ticks(ms, bid, ask, "M1")
    assert b["spread"][0] == pytest.approx((0.3 + 0.4 + 0.2 + 0.2) / 4)
    assert b["spread_max"][0] == pytest.approx(0.4)


def test_boundaries_align_to_the_timeframe():
    ms, bid, ask = _ticks()
    for tf, expect in (("M1", 60_000), ("M5", 300_000)):
        b = bars_from_ticks(ms, bid, ask, tf)
        assert all(int(t) % expect == 0 for t in b["ms"])


def test_coarser_timeframe_merges_bars():
    ms, bid, ask = _ticks()
    m5 = bars_from_ticks(ms, bid, ask, "M5")
    assert len(m5["ms"]) == 2                     # 01:00-01:05, then 01:05
    assert m5["volume"][0] == 6
    assert m5["high"][0] == 1903.0
    assert m5["low"][0] == 1899.0


def test_empty_input_returns_empty_not_a_crash():
    b = bars_from_ticks(np.empty(0, np.int64), np.empty(0), np.empty(0), "M1")
    assert len(b["ms"]) == 0


def test_build_bars_reads_a_tick_csv_end_to_end(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(TICKS)
    md = read_csv(p)
    assert md.kind == "tick"
    b = build_bars(md, "M1")
    assert len(b["ms"]) == 3
    assert b["open"][0] == pytest.approx(1900.0)
    assert b["volume"][0] == 4


def test_a_coarser_source_is_refused_not_faked(tmp_path):
    """§4.6: aggregate up, never split down."""
    p = tmp_path / "h1.csv"
    p.write_text(
        "timestamp,open,high,low,close\n"
        "2021-01-04 01:00:00,1.0,2.0,0.5,1.5\n"
        "2021-01-04 02:00:00,1.5,2.5,1.0,2.0\n"
        "2021-01-04 03:00:00,2.0,3.0,1.5,2.5\n"
    )
    md = read_csv(p)
    with pytest.raises(ValueError, match="never be split down|never split down"):
        build_bars(md, "M5")


def test_bar_source_at_its_own_timeframe_passes_through(tmp_path):
    p = tmp_path / "m1.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2021-01-04 01:00:00,1.0,2.0,0.5,1.5,10\n"
        "2021-01-04 01:01:00,1.5,2.5,1.0,2.0,20\n"
        "2021-01-04 01:02:00,2.0,3.0,1.5,2.5,30\n"
    )
    md = read_csv(p)
    b = build_bars(md, "M1")
    assert len(b["ms"]) == 3
    assert b["volume"].tolist() == [10, 20, 30]
