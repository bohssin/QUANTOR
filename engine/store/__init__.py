"""Market data store: ingest, quality checks, bar construction. Plan §4."""
from .bars import bars_from_ticks, build_bars
from .ingest import (
    TIMEFRAMES,
    MarketData,
    QualityReport,
    SourceSpec,
    assert_serves_timeframe,
    detect_decimals,
    detect_resolution_ms,
    infer_utc_offset_hours,
    read_csv,
    read_tick_csv,
    timeframe_ms,
)
from .stream import (
    BarAccumulator,
    StreamResult,
    fingerprint,
    stream_bars,
    stream_batches,
)

__all__ = [
    "TIMEFRAMES",
    "BarAccumulator",
    "StreamResult",
    "bars_from_ticks",
    "build_bars",
    "fingerprint",
    "stream_bars",
    "stream_batches",
    "MarketData",
    "QualityReport",
    "SourceSpec",
    "assert_serves_timeframe",
    "detect_decimals",
    "detect_resolution_ms",
    "infer_utc_offset_hours",
    "read_csv",
    "read_tick_csv",
    "timeframe_ms",
]
