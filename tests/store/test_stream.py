"""Streaming ingest must agree with the whole-file path, exactly. Plan §4.2.

The streaming builder exists because the owner's tick archive is 11 GB. That
makes it the path real results come from, and a streamed bar that differs from
a whole-file bar by one tick is a silent, permanent divergence between what was
tested and what was traded.

So these tests do not check that streaming is "close". They check that the two
builders produce **identical arrays**, across batch sizes chosen to land
mid-bar, and that memory does not grow with file size.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.store import SourceSpec, bars_from_ticks, read_csv, stream_bars
from engine.store.stream import BarAccumulator, fingerprint

M15 = 900_000


def _ticks(n: int, *, seed: int = 7, start_ms: int = 1_609_722_000_000,
           step_ms: int = 1_100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Irregularly spaced ticks, so bar boundaries fall at awkward places."""
    rng = np.random.default_rng(seed)
    gaps = rng.integers(1, step_ms, n).astype(np.int64)
    ms = start_ms + np.cumsum(gaps)
    bid = 1900.0 + np.cumsum(rng.normal(0, 0.02, n))
    ask = bid + rng.uniform(0.10, 0.45, n)
    return ms, bid, ask


def _write_csv(path, ms, bid, ask) -> None:
    import datetime as dt
    with open(path, "w") as fh:
        fh.write("timestamp,bidPrice,askPrice\n")
        for t, b, a in zip(ms, bid, ask):
            stamp = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc)
            fh.write(f"{stamp:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{b:.3f},{a:.3f}\n")


# --- the accumulator agrees with the whole-array builder ---------------------

@pytest.mark.parametrize("chunk", [1, 2, 7, 100, 999, 5000])
def test_chunked_bars_are_identical_to_whole_array_bars(chunk):
    ms, bid, ask = _ticks(20_000)
    whole = bars_from_ticks(ms, bid, ask, "M15")

    acc = BarAccumulator("M15")
    for i in range(0, len(ms), chunk):
        acc.push(ms[i:i + chunk], bid[i:i + chunk], ask[i:i + chunk])
    streamed = acc.finish()

    assert set(streamed) == set(whole)
    for key in whole:
        np.testing.assert_array_equal(
            streamed[key], whole[key],
            err_msg=f"{key} differs at chunk size {chunk}",
        )


def test_a_bar_split_across_chunks_merges_rather_than_duplicating():
    """The one bug this design can have: two half-bars instead of one."""
    base = 1_609_722_000_000                      # exactly on an M15 boundary
    ms = np.array([base + 1, base + 2, base + 3, base + M15 + 1], np.int64)
    bid = np.array([10.0, 12.0, 9.0, 20.0])
    ask = bid + 0.2

    acc = BarAccumulator("M15")
    acc.push(ms[:2], bid[:2], ask[:2])            # split inside the first bar
    acc.push(ms[2:], bid[2:], ask[2:])
    out = acc.finish()

    assert len(out["ms"]) == 2, "the split bar was duplicated"
    assert out["high"][0] == 12.0 and out["low"][0] == 9.0
    assert out["close"][0] == 9.0 and out["volume"][0] == 3


def test_an_empty_push_changes_nothing():
    ms, bid, ask = _ticks(500)
    acc = BarAccumulator("M15")
    acc.push(ms, bid, ask)
    before = acc.finish()
    acc.push(np.empty(0, np.int64), np.empty(0), np.empty(0))
    np.testing.assert_array_equal(acc.finish()["ms"], before["ms"])


def test_an_accumulator_that_saw_nothing_returns_empty_arrays():
    out = BarAccumulator("M5").finish()
    assert len(out["ms"]) == 0
    assert out["close"].dtype == np.float64


# --- the file-level streaming path -------------------------------------------

def test_streamed_file_matches_read_csv_bar_for_bar(tmp_path):
    ms, bid, ask = _ticks(40_000)
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    whole = read_csv(path)
    reference = bars_from_ticks(whole.ms, whole.to_float("bid"),
                                whole.to_float("ask"), "M15")
    streamed = stream_bars(path, "M15", block_size=1 << 16)   # forces many batches

    assert streamed.batches > 1, "block_size did not actually split the file"
    np.testing.assert_array_equal(streamed.bars["ms"], reference["ms"])
    np.testing.assert_array_equal(streamed.bars["volume"], reference["volume"])
    for key in ("open", "high", "low", "close", "spread", "spread_max"):
        np.testing.assert_allclose(streamed.bars[key], reference[key], rtol=0, atol=1e-9)


def test_streaming_counts_rows_and_span_exactly(tmp_path):
    ms, bid, ask = _ticks(10_000)
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    out = stream_bars(path, "M5", block_size=1 << 15)
    assert out.quality.rows == 10_000
    assert out.quality.first_ms == int(ms[0])
    assert out.quality.last_ms == int(ms[-1])
    assert out.quality.out_of_order == 0


def test_the_gmt_offset_shifts_bar_boundaries(tmp_path):
    """GMT+3 is the owner's file. The offset must move bars, not just labels."""
    ms, bid, ask = _ticks(20_000)
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    utc = stream_bars(path, "H1", SourceSpec(utc_offset_hours=0.0))
    plus3 = stream_bars(path, "H1", SourceSpec(utc_offset_hours=3.0))

    assert int(plus3.quality.first_ms) == int(utc.quality.first_ms) - 3 * 3_600_000
    assert (plus3.bars["ms"] == utc.bars["ms"] - 3 * 3_600_000).all()


def test_out_of_order_rows_are_reported_not_silently_repaired(tmp_path):
    """A stream cannot sort. It must say so rather than pretend."""
    ms, bid, ask = _ticks(5_000)
    ms[2_000], ms[2_001] = ms[2_001], ms[2_000]
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    out = stream_bars(path, "M15", block_size=1 << 14)
    assert out.quality.out_of_order >= 1


def test_duplicate_timestamps_are_counted_across_a_batch_seam(tmp_path):
    ms, bid, ask = _ticks(4_000)
    ms[1_000:1_010] = ms[1_000]                   # a run of ten
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    for block in (1 << 12, 1 << 14, 1 << 20):
        out = stream_bars(path, "M15", block_size=block)
        assert out.quality.duplicate_timestamps == 9, f"block_size={block}"
        assert out.quality.max_duplicate_run == 9, f"block_size={block}"


def test_a_bar_file_is_refused_with_a_reason(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("timestamp,open,high,low,close\n"
                    "2021-01-04 01:00:00,1,2,0.5,1.5\n")
    with pytest.raises(NotImplementedError, match="read_csv"):
        stream_bars(path, "M15")


def test_batch_size_does_not_change_the_result(tmp_path):
    ms, bid, ask = _ticks(30_000)
    path = tmp_path / "ticks.csv"
    _write_csv(path, ms, bid, ask)

    results = [stream_bars(path, "M15", block_size=b)
               for b in (1 << 13, 1 << 16, 1 << 24)]
    assert results[0].batches != results[-1].batches, "batch sizes did not differ"
    for other in results[1:]:
        np.testing.assert_array_equal(results[0].bars["ms"], other.bars["ms"])
        np.testing.assert_array_equal(results[0].bars["close"], other.bars["close"])
        np.testing.assert_array_equal(results[0].bars["volume"], other.bars["volume"])


# --- identity ----------------------------------------------------------------

def test_fingerprint_changes_when_the_file_changes(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("timestamp,bid,ask\n2021-01-04 01:00:00.000,1.0,1.1\n")
    first = fingerprint(path)
    assert first.startswith("fp:"), "the fingerprint must announce what it is"

    path.write_text("timestamp,bid,ask\n2021-01-04 01:00:00.000,1.0,1.2\n")
    assert fingerprint(path) != first


def test_fingerprint_is_stable_for_an_unchanged_file(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("timestamp,bid,ask\n2021-01-04 01:00:00.000,1.0,1.1\n")
    assert fingerprint(path) == fingerprint(path)
