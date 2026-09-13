"""Ingest tests against the owner's real tick format. Plan §4.2, §4.4."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from engine.store import (
    SourceSpec,
    assert_serves_timeframe,
    detect_decimals,
    detect_resolution_ms,
    infer_utc_offset_hours,
    read_csv,
    read_tick_csv,
)

# Exactly the owner's sample, including the ragged decimals (1905.12).
SAMPLE = """timestamp,bidPrice,askPrice
2021-01-04 01:00:00.413,1904.998,1905.366
2021-01-04 01:00:00.464,1905.248,1905.492
2021-01-04 01:00:00.514,1904.664,1905.12
"""


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "ticks.csv"
    p.write_text(SAMPLE)
    return p


def test_reads_owner_format(sample_csv):
    chunk = read_tick_csv(sample_csv)
    assert len(chunk) == 3
    assert chunk.ms.dtype == np.int64
    # Ragged decimals must survive: 1905.12 is 1905120 at tick_size 0.001.
    assert chunk.ask[2] == 1905120
    np.testing.assert_allclose(chunk.to_float("bid"), [1904.998, 1905.248, 1904.664])
    np.testing.assert_allclose(chunk.to_float("ask"), [1905.366, 1905.492, 1905.12])


def test_timestamps_are_milliseconds_not_seconds(sample_csv):
    """Guards the pandas-3.0 resolution trap described in the module docstring."""
    chunk = read_tick_csv(sample_csv)
    expected = int(
        dt.datetime(2021, 1, 4, 1, 0, 0, 413_000, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    assert int(chunk.ms[0]) == expected
    assert int(chunk.ms[1]) - int(chunk.ms[0]) == 51      # .413 -> .464


def test_prices_stored_as_exact_scaled_integers(sample_csv):
    chunk = read_tick_csv(sample_csv)
    assert np.issubdtype(chunk.bid.dtype, np.integer)
    # Exactness is the point: stop/target comparisons must not depend on epsilon.
    assert chunk.bid.tolist() == [1904998, 1905248, 1904664]
    assert chunk.scale == 1000


def test_utc_offset_is_applied(sample_csv):
    utc = read_tick_csv(sample_csv, SourceSpec())
    eet = read_tick_csv(sample_csv, SourceSpec(utc_offset_hours=2.0))
    assert int(utc.ms[0]) - int(eet.ms[0]) == 2 * 3_600_000


def test_missing_column_is_refused_not_guessed(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("time,foo,baz\n2021-01-04 01:00:00.413,1904.998,1905.366\n")
    with pytest.raises(ValueError, match="needs bid and ask"):
        read_tick_csv(p)


def test_quality_report_flags_same_millisecond_ticks(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text(
        "timestamp,bidPrice,askPrice\n"
        "2021-01-04 01:00:00.100,1904.998,1905.366\n"
        "2021-01-04 01:00:00.100,1905.001,1905.370\n"
        "2021-01-04 01:00:00.100,1905.004,1905.372\n"
        "2021-01-04 01:00:00.200,1905.010,1905.380\n"
    )
    q = read_tick_csv(p).quality
    assert q.duplicate_timestamps == 2
    assert q.max_duplicate_run == 2
    assert q.rows == 4


def test_same_millisecond_ticks_keep_source_order(tmp_path):
    """Stable sort — fills depend on this order, so it must be deterministic."""
    p = tmp_path / "stable.csv"
    p.write_text(
        "timestamp,bidPrice,askPrice\n"
        "2021-01-04 01:00:00.100,1904.900,1905.100\n"
        "2021-01-04 01:00:00.100,1904.800,1905.000\n"
        "2021-01-04 01:00:00.100,1904.700,1904.900\n"
    )
    chunk = read_tick_csv(p)
    assert chunk.bid.tolist() == [1904900, 1904800, 1904700]


def test_quality_report_flags_bad_spreads(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "timestamp,bidPrice,askPrice\n"
        "2021-01-04 01:00:00.100,1905.000,1904.000\n"   # crossed
        "2021-01-04 01:00:00.200,1905.000,1905.000\n"   # zero
        "2021-01-04 01:00:00.300,1905.000,1905.300\n"
    )
    q = read_tick_csv(p).quality
    assert q.negative_spread == 1
    assert q.zero_spread == 1


def _weekly_stream(weeks: int = 3, tick_ms: int = 60_000) -> np.ndarray:
    """Ticks every minute across 5 trading days, then a weekend gap. Opens
    Sunday 22:00 UTC, which is the nominal FX/metals reopen."""
    week_open = int(
        dt.datetime(2021, 1, 3, 22, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    session_ms = 5 * 86_400_000
    out: list[int] = []
    for week in range(weeks):
        start = week_open + week * 7 * 86_400_000
        out.extend(range(start, start + session_ms, tick_ms))
    return np.array(sorted(out), dtype=np.int64)


def test_infer_utc_offset_from_weekend_gap():
    """A stream reopening at 22:00 in its own clock is UTC; shift it and the
    inference must follow."""
    ms = _weekly_stream()
    assert infer_utc_offset_hours(ms) == 0.0
    assert infer_utc_offset_hours(ms + 2 * 3_600_000) == 2.0
    assert infer_utc_offset_hours(ms + 3 * 3_600_000) == 3.0
    assert infer_utc_offset_hours(ms - 5 * 3_600_000) == -5.0


def test_infer_utc_offset_is_robust_to_a_thin_friday_close():
    """The estimate reads the reopen, so a Friday close that trails off early
    must not bias it — the reason the inference does not use the gap start."""
    ms = _weekly_stream()
    # Drop the last 90 minutes of every trading week: a soft, early close.
    week_open = ms[0]
    keep = [t for t in ms.tolist()
            if (t - week_open) % (7 * 86_400_000) < 5 * 86_400_000 - 90 * 60_000]
    thin = np.array(keep, dtype=np.int64)
    assert infer_utc_offset_hours(thin) == 0.0


def test_infer_utc_offset_returns_none_on_short_data(sample_csv):
    chunk = read_tick_csv(sample_csv)
    assert infer_utc_offset_hours(chunk.ms) is None
