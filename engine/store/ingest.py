"""Tick CSV ingest. Plan §4.2.

Targets the owner's format:

    timestamp,bidPrice,askPrice
    2021-01-04 01:00:00.413,1904.998,1905.366

but is driven by an explicit column mapping (`SourceSpec`), never by format
sniffing — a store that silently reinterprets a column is worse than one that
refuses to load.

Three decisions this module encodes, each measured rather than assumed:

**pyarrow, not pandas.** `pyarrow.csv` parses this format at ~23M rows/s vs
pandas' ~0.6M rows/s — a 39x difference `[measured]`, which is the gap between
~18 s and ~12 min for a year of ticks at the sampled density.

**Timestamp resolution is pinned explicitly.** pandas 3.0 infers datetime
resolution from the data, so `to_datetime(...).astype('int64')` silently yields
*seconds* for some inputs and *milliseconds* for others. That bug produced
timestamps off by a factor of 1000 in testing and raised no error. Arrow is
told `timestamp('ms')` and the result is checked against a stdlib parse.

**Prices are stored as integers scaled by tick size.** At tick_size=0.001 a
XAUUSD quote of 1905.366 stores as 1905366. This halves memory (24 -> 16 bytes
per tick `[measured]`) and, more importantly, makes stop/target comparisons
*exact* — no float epsilon at the one place in the system where a wrong
comparison silently changes a trade.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

__all__ = ["SourceSpec", "TickChunk", "QualityReport", "read_tick_csv", "infer_utc_offset_hours"]

_MS_PER_HOUR = 3_600_000
_MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class SourceSpec:
    """How to read one vendor's tick files. Explicit, never sniffed."""

    timestamp_col: str = "timestamp"
    bid_col: str = "bidPrice"
    ask_col: str = "askPrice"
    volume_col: str | None = None
    #: Timezone the timestamp strings are expressed in, as a UTC offset in
    #: hours. Tick files rarely say. Use `infer_utc_offset_hours` to check the
    #: declared value against where the weekend gap actually falls (plan §4.3).
    utc_offset_hours: float = 0.0
    #: Price quantum. Prices are stored as round(price / tick_size).
    tick_size: float = 0.001

    def scale(self) -> int:
        inv = round(1.0 / self.tick_size)
        if abs(inv * self.tick_size - 1.0) > 1e-12:
            raise ValueError(f"tick_size {self.tick_size!r} is not a clean reciprocal")
        return inv


@dataclass
class QualityReport:
    """What the store learned about a file. Plan §4.4 — validate before trusting."""

    rows: int
    file_sha256: str
    first_ms: int
    last_ms: int
    out_of_order: int
    same_ms_pairs: int
    max_same_ms_run: int
    negative_spread: int
    zero_spread: int
    crossed_or_zero_price: int
    spread_mean: float
    spread_p50: float
    spread_p99: float
    largest_gaps_ms: list[int] = field(default_factory=list)

    def summary(self) -> str:
        span_days = (self.last_ms - self.first_ms) / _MS_PER_DAY
        return (
            f"{self.rows:,} ticks over {span_days:.1f}d | "
            f"same-ms {self.same_ms_pairs:,} (max run {self.max_same_ms_run}) | "
            f"out-of-order {self.out_of_order:,} | "
            f"spread p50 {self.spread_p50:.3f} p99 {self.spread_p99:.3f} | "
            f"neg {self.negative_spread} zero {self.zero_spread} "
            f"bad-price {self.crossed_or_zero_price}"
        )


@dataclass
class TickChunk:
    """Normalized ticks. Prices are integers in tick_size units (see module docstring)."""

    ms: np.ndarray           # int64, UTC
    bid: np.ndarray          # int32/int64, scaled
    ask: np.ndarray          # int32/int64, scaled
    scale: int
    quality: QualityReport

    def bid_float(self) -> np.ndarray:
        return self.bid / self.scale

    def ask_float(self) -> np.ndarray:
        return self.ask / self.scale

    def __len__(self) -> int:
        return int(self.ms.shape[0])


def read_tick_csv(path: str | Path, spec: SourceSpec | None = None) -> TickChunk:
    """Read one tick CSV into normalized arrays.

    Ordering: ticks are sorted by timestamp with a **stable** sort, so ticks
    sharing a millisecond keep source-file order. Fill results depend on that
    order, so it must be deterministic (plan §4.4).
    """
    spec = spec or SourceSpec()
    path = Path(path)

    types: dict[str, pa.DataType] = {
        spec.timestamp_col: pa.timestamp("ms"),
        spec.bid_col: pa.float64(),
        spec.ask_col: pa.float64(),
    }
    if spec.volume_col:
        types[spec.volume_col] = pa.float64()

    table = pacsv.read_csv(
        path, convert_options=pacsv.ConvertOptions(column_types=types)
    )
    _require_columns(table, spec, path)

    ms = table.column(spec.timestamp_col).cast(pa.int64()).to_numpy()
    bid_f = table.column(spec.bid_col).to_numpy(zero_copy_only=False)
    ask_f = table.column(spec.ask_col).to_numpy(zero_copy_only=False)

    _verify_timestamp_parse(path, table, spec, ms)

    if spec.utc_offset_hours:
        ms = ms - int(round(spec.utc_offset_hours * _MS_PER_HOUR))

    scale = spec.scale()
    bid = _to_scaled(bid_f, scale)
    ask = _to_scaled(ask_f, scale)

    order = np.argsort(ms, kind="stable")
    out_of_order = int((np.diff(ms) < 0).sum())
    ms, bid, ask = ms[order], bid[order], ask[order]

    quality = _assess(ms, bid, ask, scale, out_of_order, _sha256(path))
    return TickChunk(ms=ms, bid=bid, ask=ask, scale=scale, quality=quality)


#: Nominal Sunday FX/metals reopen, in UTC. The reference the inference below
#: measures against; override per broker if theirs differs.
NOMINAL_SUNDAY_OPEN_UTC_HOUR = 22.0


def infer_utc_offset_hours(ms: np.ndarray, *, top_n: int = 12) -> float | None:
    """Infer the file's UTC offset from where the weekly market gap ends.

    FX/metals close Friday evening and reopen Sunday evening, so the largest
    recurring gaps in a tick stream are weekends, and the hour at which the
    market *reopens* reveals the clock the file is keyed to.

    Measured at the gap's **end**, not its start. The reopen is a hard
    boundary — the feed switches on at a precise time and the first tick lands
    within milliseconds. The Friday close is soft: liquidity thins out and the
    last tick can precede the official close by an arbitrary margin, which
    biases the estimate early by exactly that margin.

    Returns None when the data is too short to contain clear weekend gaps —
    always a hint to confirm with the owner, never an authority. Plan §4.3.
    """
    if ms.shape[0] < 2:
        return None

    gaps = np.diff(ms)
    candidates = np.argsort(gaps)[-top_n:]
    weekend = [i for i in candidates if gaps[i] > 24 * _MS_PER_HOUR]
    if not weekend:
        return None

    # Hour-of-day at which trading resumes, in the file's own clock.
    reopen_hours = [(int(ms[i + 1]) % _MS_PER_DAY) / _MS_PER_HOUR for i in weekend]
    median_reopen = float(np.median(reopen_hours))

    offset = median_reopen - NOMINAL_SUNDAY_OPEN_UTC_HOUR
    if offset > 12.0:
        offset -= 24.0
    elif offset < -12.0:
        offset += 24.0
    return round(offset * 4.0) / 4.0      # snap to quarter-hour


# --- internals ---------------------------------------------------------------

def _require_columns(table: pa.Table, spec: SourceSpec, path: Path) -> None:
    wanted = [spec.timestamp_col, spec.bid_col, spec.ask_col]
    missing = [c for c in wanted if c not in table.column_names]
    if missing:
        raise ValueError(
            f"{path.name}: missing column(s) {missing}; file has "
            f"{table.column_names}. Set them explicitly on SourceSpec — this "
            "loader never guesses."
        )


def _verify_timestamp_parse(path: Path, table: pa.Table, spec: SourceSpec,
                            ms: np.ndarray) -> None:
    """Cross-check Arrow's first timestamp against a stdlib parse.

    Cheap insurance against a silent unit error: an ingest that is wrong by a
    factor of 1000 produces a store that looks superficially fine and is
    entirely useless.
    """
    if ms.shape[0] == 0:
        return
    raw = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(skip_rows_after_names=0),
        convert_options=pacsv.ConvertOptions(column_types={spec.timestamp_col: pa.string()}),
    ).column(spec.timestamp_col)[0].as_py()

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            parsed = dt.datetime.strptime(raw, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        expected = int(parsed.timestamp() * 1000)
        if int(ms[0]) != expected:
            raise ValueError(
                f"{path.name}: timestamp parse mismatch on row 0 — arrow gave "
                f"{int(ms[0])} ms, stdlib gave {expected} ms for {raw!r}. "
                "Refusing to build a store on an ambiguous time unit."
            )
        return
    # Unrecognized layout: Arrow parsed it, we simply could not double-check.


def _to_scaled(prices: np.ndarray, scale: int) -> np.ndarray:
    scaled = np.rint(prices * scale)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("non-finite price in source data")
    lo, hi = scaled.min(), scaled.max()
    if lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
        return scaled.astype(np.int32)
    return scaled.astype(np.int64)


def _assess(ms, bid, ask, scale, out_of_order, digest) -> QualityReport:
    same = np.diff(ms) == 0
    spread = (ask - bid) / scale
    gaps = np.diff(ms)
    largest = sorted(gaps[np.argsort(gaps)[-5:]].tolist(), reverse=True) if gaps.size else []

    return QualityReport(
        rows=int(ms.shape[0]),
        file_sha256=digest,
        first_ms=int(ms[0]) if ms.size else 0,
        last_ms=int(ms[-1]) if ms.size else 0,
        out_of_order=out_of_order,
        same_ms_pairs=int(same.sum()),
        max_same_ms_run=int(_max_run(same)),
        negative_spread=int((ask < bid).sum()),
        zero_spread=int((ask == bid).sum()),
        crossed_or_zero_price=int((bid <= 0).sum() + (ask <= 0).sum()),
        spread_mean=float(spread.mean()) if spread.size else 0.0,
        spread_p50=float(np.percentile(spread, 50)) if spread.size else 0.0,
        spread_p99=float(np.percentile(spread, 99)) if spread.size else 0.0,
        largest_gaps_ms=largest,
    )


def _max_run(flags: np.ndarray) -> int:
    """Longest run of True in a bool array, as a count of consecutive pairs."""
    if flags.size == 0 or not flags.any():
        return 0
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
