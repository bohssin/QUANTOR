"""Tick store: ingest, quality checks, bar construction. Plan §4."""
from .ingest import (
    QualityReport,
    SourceSpec,
    TickChunk,
    infer_utc_offset_hours,
    read_tick_csv,
)

__all__ = [
    "QualityReport",
    "SourceSpec",
    "TickChunk",
    "infer_utc_offset_hours",
    "read_tick_csv",
]
