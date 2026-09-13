"""Bar sources, auto-detection, and the resolution rule. Plan §4.2, §4.6."""

from __future__ import annotations

import numpy as np
import pytest

from engine.store import (
    SourceSpec,
    assert_serves_timeframe,
    detect_decimals,
    detect_resolution_ms,
    read_csv,
    timeframe_ms,
)

M1_BARS = """timestamp,open,high,low,close,volume
2021-01-04 01:00:00,1904.998,1905.400,1904.200,1905.100,318
2021-01-04 01:01:00,1905.100,1905.900,1905.000,1905.750,402
2021-01-04 01:02:00,1905.750,1906.100,1905.200,1905.300,289
2021-01-04 01:03:00,1905.300,1905.500,1904.800,1904.900,254
"""

TICKS = """timestamp,bidPrice,askPrice
2021-01-04 01:00:00.413,1904.998,1905.366
2021-01-04 01:00:00.464,1905.248,1905.492
2021-01-04 01:00:00.514,1904.664,1905.12
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# --- kind detection ---------------------------------------------------------

def test_detects_bar_source(tmp_path):
    data = read_csv(_write(tmp_path, "m1.csv", M1_BARS))
    assert data.kind == "bar"
    assert len(data) == 4
    np.testing.assert_allclose(data.to_float("close"),
                               [1905.100, 1905.750, 1905.300, 1904.900])
    assert data.volume.tolist() == [318, 402, 289, 254]


def test_detects_tick_source(tmp_path):
    data = read_csv(_write(tmp_path, "t.csv", TICKS))
    assert data.kind == "tick"
    assert data.resolution_ms == 0


def test_ambiguous_file_refuses_rather_than_guessing(tmp_path):
    both = ("timestamp,open,high,low,close,bid,ask\n"
            "2021-01-04 01:00:00,1.0,2.0,0.5,1.5,1.4,1.6\n")
    with pytest.raises(ValueError, match="both OHLC and bid/ask"):
        read_csv(_write(tmp_path, "both.csv", both))
    # ...but an explicit kind resolves it.
    assert read_csv(_write(tmp_path, "b2.csv", both), SourceSpec(kind="bar")).kind == "bar"


def test_column_aliases_are_accepted(tmp_path):
    aliased = ("Date_Time,O,H,L,C,Vol\n"
               "2021-01-04 01:00:00,1904.9,1905.4,1904.2,1905.1,318\n")
    data = read_csv(_write(tmp_path, "alias.csv", aliased))
    assert data.kind == "bar"
    assert data.detected["columns"]["close"] == "C"


# --- precision read from the file, not declared ------------------------------

def test_decimals_detected_from_column_text():
    assert detect_decimals(["1904.998", "1905.12", "1905"]) == 3
    assert detect_decimals(["1.10234", "1.1"]) == 5
    assert detect_decimals(["100", "200"]) == 0


def test_tick_size_comes_from_the_csv(tmp_path):
    """3-decimal metal and 5-decimal FX, same code path, nothing declared."""
    gold = read_csv(_write(tmp_path, "gold.csv", TICKS))
    assert gold.scale == 1000
    assert gold.tick_size == pytest.approx(0.001)
    assert gold.bid.tolist() == [1904998, 1905248, 1904664]

    fx = ("timestamp,bidPrice,askPrice\n"
          "2021-01-04 01:00:00.100,1.22345,1.22351\n"
          "2021-01-04 01:00:00.200,1.22348,1.22354\n")
    eurusd = read_csv(_write(tmp_path, "fx.csv", fx))
    assert eurusd.scale == 100_000
    assert eurusd.bid.tolist() == [122345, 122348]
    np.testing.assert_allclose(eurusd.to_float("bid"), [1.22345, 1.22348])


def test_explicit_tick_size_overrides_detection(tmp_path):
    data = read_csv(_write(tmp_path, "t.csv", TICKS), SourceSpec(tick_size=0.01))
    assert data.scale == 100


# --- resolution -------------------------------------------------------------

def test_resolution_detected_from_spacing(tmp_path):
    assert read_csv(_write(tmp_path, "m1.csv", M1_BARS)).resolution_ms == 60_000


def test_resolution_uses_the_mode_not_the_minimum():
    """Session gaps and a stray duplicate must not skew the estimate."""
    ms = np.array([0, 60_000, 120_000, 120_000, 180_000, 9_000_000, 9_060_000],
                  dtype=np.int64)
    assert detect_resolution_ms(ms) == 60_000


def test_explicit_timeframe_overrides_detection(tmp_path):
    data = read_csv(_write(tmp_path, "m1.csv", M1_BARS), SourceSpec(timeframe="M5"))
    assert data.resolution_ms == timeframe_ms("M5")


# --- the rule: aggregate up, never invent down -------------------------------

def test_source_serves_equal_and_coarser_timeframes(tmp_path):
    m1 = read_csv(_write(tmp_path, "m1.csv", M1_BARS))
    assert m1.serves("M1")
    assert m1.serves("M15")
    assert m1.serves("H4")
    assert not m1.serves("TICK")


def test_ticks_serve_everything(tmp_path):
    ticks = read_csv(_write(tmp_path, "t.csv", TICKS))
    assert all(ticks.serves(tf) for tf in ("TICK", "M1", "M15", "H1", "D1"))


def test_too_coarse_a_source_is_refused_with_a_useful_message(tmp_path):
    h1 = ("timestamp,open,high,low,close\n"
          "2021-01-04 01:00:00,1.0,2.0,0.5,1.5\n"
          "2021-01-04 02:00:00,1.5,2.5,1.0,2.0\n"
          "2021-01-04 03:00:00,2.0,3.0,1.5,2.5\n")
    data = read_csv(_write(tmp_path, "h1.csv", h1))
    assert data.resolution_ms == timeframe_ms("H1")
    assert_serves_timeframe(data, "H4")            # fine: aggregate up
    with pytest.raises(ValueError, match="never be split down|never split down"):
        assert_serves_timeframe(data, "M5")
