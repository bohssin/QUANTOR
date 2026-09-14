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

__all__ = [
    "TIMEFRAMES",
    "bars_from_ticks",
    "build_bars",
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
