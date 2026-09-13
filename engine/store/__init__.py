"""Market data store: ingest, quality checks, bar construction. Plan §4."""
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
