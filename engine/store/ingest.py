"""Market data ingest. Plan §4.2.

Accepts **tick CSVs or bar CSVs**, because the owner's data is whatever the
owner has:

    timestamp,bidPrice,askPrice                 <- ticks
    2021-01-04 01:00:00.413,1904.998,1905.366

    timestamp,open,high,low,close,volume        <- bars (M1, M5, H1, ...)
    2021-01-04 01:00:00,1904.998,1905.4,1904.2,1905.1,318

The file's own resolution decides which modeling modes are available (§6) and
which strategy timeframes it can serve (§4.6): a source can always be
aggregated *up* to a coarser timeframe, never invented *down* to a finer one.

Four decisions this module encodes, each measured rather than assumed:

**pyarrow, not pandas.** `pyarrow.csv` parses this format at ~23M rows/s vs
pandas' ~0.6M rows/s — a 39x difference `[measured]`.

**Timestamp resolution is pinned explicitly.** pandas 3.0 infers datetime
resolution from the data, so `to_datetime(...).astype('int64')` silently yields
*seconds* for some inputs and *milliseconds* for others. That bug produced
timestamps off by a factor of 1000 in testing and raised no error. Arrow is
told `timestamp('ms')` and the result is checked against a stdlib parse.

**Price precision is read from the file, not declared.** Digits differ per
asset and per feed; the CSV already states the precision by how it is written.
`1904.998` and `1905.12` in one column means 3 decimals, so tick_size 0.001.
Declaring it per instrument was one more thing to get wrong.

**Prices are stored as integers scaled by tick size.** 1905.366 at tick_size
0.001 stores as 1905366. Cuts memory by a third `[measured]` and, more
importantly, makes stop/target comparisons *exact* — no float epsilon at the
one place in the system where a wrong comparison silently changes a trade.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

__all__ = [
    "SourceSpec",
    "MarketData",
    "QualityReport",
    "read_csv",
    "read_tick_csv",
    "detect_decimals",
    "detect_resolution_ms",
    "infer_utc_offset_hours",
    "timeframe_ms",
    "assert_serves_timeframe",
]

_MS_PER_HOUR = 3_600_000
_MS_PER_DAY = 86_400_000

#: Nominal Sunday FX/metals reopen, in UTC. Reference for offset inference.
NOMINAL_SUNDAY_OPEN_UTC_HOUR = 22.0

#: MT5 timeframe names -> milliseconds.
TIMEFRAMES: dict[str, int] = {
    "TICK": 0,
    "M1": 60_000, "M2": 120_000, "M3": 180_000, "M4": 240_000, "M5": 300_000,
    "M6": 360_000, "M10": 600_000, "M12": 720_000, "M15": 900_000,
    "M20": 1_200_000, "M30": 1_800_000,
    "H1": 3_600_000, "H2": 7_200_000, "H3": 10_800_000, "H4": 14_400_000,
    "H6": 21_600_000, "H8": 28_800_000, "H12": 43_200_000,
    "D1": 86_400_000, "W1": 604_800_000,
}

_TICK_ALIASES = {
    "bid": ("bid", "bidprice", "bid_price", "b"),
    "ask": ("ask", "askprice", "ask_price", "a"),
}
_BAR_ALIASES = {
    "open": ("open", "o", "openprice"),
    "high": ("high", "h", "highprice"),
    "low": ("low", "l", "lowprice"),
    "close": ("close", "c", "closeprice"),
    "volume": ("volume", "vol", "v", "tickvolume", "tick_volume"),
    "spread": ("spread", "spr"),
}
_TIME_ALIASES = ("timestamp", "time", "datetime", "date_time", "gmt time", "date")


def timeframe_ms(tf: str) -> int:
    """'M15' -> 900000. Raises on an unknown name."""
    key = tf.strip().upper()
    if key not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}; known: {', '.join(TIMEFRAMES)}")
    return TIMEFRAMES[key]


@dataclass(frozen=True)
class SourceSpec:
    """How to read one file.

    Everything defaults to auto-detection, because the owner's answer to "what
    is in this CSV" is reasonably "look at it". Detection never guesses
    silently: what it found is reported on `MarketData.detected`, and anything
    ambiguous raises rather than picking a side. Any field set explicitly wins.
    """

    kind: Literal["auto", "tick", "bar"] = "auto"
    timestamp_col: str | None = None
    bid_col: str | None = None
    ask_col: str | None = None
    open_col: str | None = None
    high_col: str | None = None
    low_col: str | None = None
    close_col: str | None = None
    volume_col: str | None = None
    spread_col: str | None = None
    #: Clock the timestamps are written in, as hours east of UTC. Set per
    #: import — vendors differ and files rarely say. Cross-check with
    #: `infer_utc_offset_hours`. Plan §4.3.
    utc_offset_hours: float = 0.0
    #: Price quantum. None = read the decimals out of the file.
    tick_size: float | None = None
    #: Bar sources only. None = infer from timestamp spacing.
    timeframe: str | None = None


@dataclass
class QualityReport:
    """What the store learned about a file. Plan §4.4 — validate before trusting."""

    rows: int
    file_sha256: str
    first_ms: int
    last_ms: int
    out_of_order: int
    duplicate_timestamps: int
    max_duplicate_run: int
    negative_spread: int
    zero_spread: int
    bad_price: int
    spread_mean: float
    spread_p50: float
    spread_p99: float
    largest_gaps_ms: list[int] = field(default_factory=list)

    def summary(self) -> str:
        days = (self.last_ms - self.first_ms) / _MS_PER_DAY
        return (
            f"{self.rows:,} rows over {days:.1f}d | "
            f"dup-ts {self.duplicate_timestamps:,} (max run {self.max_duplicate_run}) | "
            f"out-of-order {self.out_of_order:,} | "
            f"spread p50 {self.spread_p50:.5f} p99 {self.spread_p99:.5f} | "
            f"neg {self.negative_spread} zero {self.zero_spread} bad {self.bad_price}"
        )


@dataclass
class MarketData:
    """Normalized source data. Prices are integers in `tick_size` units."""

    kind: Literal["tick", "bar"]
    ms: np.ndarray                      # int64, UTC
    scale: int                          # price integers are value * scale
    resolution_ms: int                  # 0 for ticks
    quality: QualityReport
    detected: dict[str, object]         # what auto-detection concluded

    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    open: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None
    close: np.ndarray | None = None
    volume: np.ndarray | None = None
    spread: np.ndarray | None = None

    @property
    def tick_size(self) -> float:
        return 1.0 / self.scale

    def to_float(self, name: str) -> np.ndarray:
        arr = getattr(self, name)
        if arr is None:
            raise AttributeError(f"{name!r} not present on a {self.kind} source")
        return arr / self.scale

    def serves(self, timeframe: str) -> bool:
        """Can this source drive a strategy on `timeframe`?

        Aggregating up is always fine; inventing detail is not. Ticks serve
        everything. Equal resolution is fine — an M5 file drives an M5
        strategy, though only at bar-close fidelity (§6).
        """
        return self.resolution_ms <= timeframe_ms(timeframe)

    def __len__(self) -> int:
        return int(self.ms.shape[0])


def read_csv(path: str | Path, spec: SourceSpec | None = None) -> MarketData:
    """Read a tick or bar CSV into normalized arrays.

    Ordering: rows are sorted by timestamp with a **stable** sort, so rows
    sharing a timestamp keep source-file order. Fill results depend on that
    order, so it must be deterministic (plan §4.4).
    """
    spec = spec or SourceSpec()
    path = Path(path)

    header = _read_header(path)
    cols, kind = _resolve_columns(header, spec, path)

    raw = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in cols.values()}
        ),
    )

    scale = _resolve_scale(raw, cols, spec)
    ms = _read_timestamps(path, raw, cols["timestamp"])
    if spec.utc_offset_hours:
        ms = ms - int(round(spec.utc_offset_hours * _MS_PER_HOUR))

    order = np.argsort(ms, kind="stable")
    out_of_order = int((np.diff(ms) < 0).sum())
    ms = ms[order]

    fields: dict[str, np.ndarray] = {}
    for name, col in cols.items():
        if name == "timestamp":
            continue
        values = _to_float(raw.column(col))
        if name == "volume":
            fields[name] = np.rint(values)[order].astype(np.int64)
        else:
            fields[name] = _to_scaled(values, scale)[order]

    resolution = 0 if kind == "tick" else (
        timeframe_ms(spec.timeframe) if spec.timeframe else detect_resolution_ms(ms)
    )
    quality = _assess(ms, fields, scale, out_of_order, _sha256(path))
    detected = {
        "kind": kind,
        "columns": cols,
        "tick_size": 1.0 / scale,
        "resolution_ms": resolution,
        "utc_offset_hours": spec.utc_offset_hours,
    }
    return MarketData(
        kind=kind, ms=ms, scale=scale, resolution_ms=resolution,
        quality=quality, detected=detected, **fields,
    )


def read_tick_csv(path: str | Path, spec: SourceSpec | None = None) -> MarketData:
    """Read a file already known to be ticks. Thin wrapper over `read_csv`."""
    spec = spec or SourceSpec()
    if spec.kind == "auto":
        spec = SourceSpec(**{**spec.__dict__, "kind": "tick"})
    data = read_csv(path, spec)
    if data.kind != "tick":
        raise ValueError(f"{Path(path).name}: expected tick data, detected {data.kind}")
    return data


def assert_serves_timeframe(data: MarketData, timeframe: str) -> None:
    """Raise unless `data` is fine enough to drive a strategy on `timeframe`."""
    if data.serves(timeframe):
        return
    have = _name_for_ms(data.resolution_ms)
    raise ValueError(
        f"source resolution {have} cannot drive a {timeframe.upper()} strategy — "
        f"bars can be aggregated up to a coarser timeframe, never split down to a "
        f"finer one. Supply {timeframe.upper()} or finer data, or run the strategy "
        f"at {have} or coarser."
    )


def detect_decimals(text_values, sample: int = 50_000) -> int:
    """Largest number of decimal places used in a price column.

    The file states its own precision: a column containing both `1904.998` and
    `1905.12` is a 3-decimal column whose trailing zeros were trimmed.
    """
    most = 0
    for i, value in enumerate(text_values):
        if i >= sample:
            break
        if value is None:
            continue
        s = value.decode() if isinstance(value, bytes) else str(value)
        dot = s.rfind(".")
        if dot >= 0:
            most = max(most, len(s) - dot - 1)
    return most


def detect_resolution_ms(ms: np.ndarray) -> int:
    """Bar spacing, as the most common positive gap between timestamps.

    The mode, not the minimum or the mean: sessions have gaps and weekends,
    and a single duplicated timestamp would drag a minimum to zero.
    """
    if ms.shape[0] < 2:
        return 0
    gaps = np.diff(ms)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 0
    values, counts = np.unique(gaps, return_counts=True)
    return int(values[int(np.argmax(counts))])


def infer_utc_offset_hours(ms: np.ndarray, *, top_n: int = 12) -> float | None:
    """Infer the file's UTC offset from where the weekly market gap ends.

    FX/metals close Friday evening and reopen Sunday evening, so the largest
    recurring gaps are weekends, and the hour at which the market *reopens*
    reveals the clock the file is keyed to.

    Measured at the gap's **end**, not its start. The reopen is a hard
    boundary — the feed switches on at a precise time and the first row lands
    immediately. The Friday close is soft: liquidity thins and the last row can
    precede the official close by an arbitrary margin, biasing the estimate
    early by exactly that margin.

    Returns None when the data is too short to contain clear weekend gaps —
    always a cross-check on a declared value, never an authority. Plan §4.3.
    """
    if ms.shape[0] < 2:
        return None

    gaps = np.diff(ms)
    weekend = [i for i in np.argsort(gaps)[-top_n:] if gaps[i] > 24 * _MS_PER_HOUR]
    if not weekend:
        return None

    reopen_hours = [(int(ms[i + 1]) % _MS_PER_DAY) / _MS_PER_HOUR for i in weekend]
    offset = float(np.median(reopen_hours)) - NOMINAL_SUNDAY_OPEN_UTC_HOUR
    if offset > 12.0:
        offset -= 24.0
    elif offset < -12.0:
        offset += 24.0
    return round(offset * 4.0) / 4.0      # snap to quarter-hour


# --- internals ---------------------------------------------------------------

def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as fh:
        line = fh.readline().strip()
    if not line:
        raise ValueError(f"{path.name}: file is empty")
    return [c.strip() for c in line.split(",")]


def _resolve_columns(header, spec, path) -> tuple[dict[str, str], str]:
    lower = {c.lower().replace(" ", "").replace("_", ""): c for c in header}

    def pick(explicit, aliases):
        if explicit:
            if explicit not in header:
                raise ValueError(
                    f"{path.name}: column {explicit!r} not in file; has {header}"
                )
            return explicit
        for a in aliases:
            key = a.replace(" ", "").replace("_", "")
            if key in lower:
                return lower[key]
        return None

    ts = pick(spec.timestamp_col, _TIME_ALIASES)
    if ts is None:
        raise ValueError(
            f"{path.name}: no timestamp column found in {header}. "
            "Set SourceSpec(timestamp_col=...) explicitly."
        )

    bid = pick(spec.bid_col, _TICK_ALIASES["bid"])
    ask = pick(spec.ask_col, _TICK_ALIASES["ask"])
    ohlc = {n: pick(getattr(spec, f"{n}_col"), _BAR_ALIASES[n])
            for n in ("open", "high", "low", "close")}
    has_tick = bid is not None and ask is not None
    has_bar = all(ohlc.values())

    kind = spec.kind
    if kind == "auto":
        if has_bar and has_tick:
            raise ValueError(
                f"{path.name}: file has both OHLC and bid/ask columns ({header}); "
                "set SourceSpec(kind='tick'|'bar') to say which to use."
            )
        if has_bar:
            kind = "bar"
        elif has_tick:
            kind = "tick"
        else:
            raise ValueError(
                f"{path.name}: found neither bid/ask nor open/high/low/close in "
                f"{header}. Name the columns explicitly on SourceSpec."
            )

    if kind == "tick":
        if not has_tick:
            raise ValueError(f"{path.name}: tick source needs bid and ask; has {header}")
        cols = {"timestamp": ts, "bid": bid, "ask": ask}
    else:
        missing = [n for n, c in ohlc.items() if c is None]
        if missing:
            raise ValueError(f"{path.name}: bar source missing {missing}; has {header}")
        cols = {"timestamp": ts, **ohlc}
        vol = pick(spec.volume_col, _BAR_ALIASES["volume"])
        if vol:
            cols["volume"] = vol
        spr = pick(spec.spread_col, _BAR_ALIASES["spread"])
        if spr:
            cols["spread"] = spr
    return cols, kind


def _resolve_scale(raw: pa.Table, cols: dict[str, str], spec: SourceSpec) -> int:
    if spec.tick_size is not None:
        inv = round(1.0 / spec.tick_size)
        if abs(inv * spec.tick_size - 1.0) > 1e-12:
            raise ValueError(f"tick_size {spec.tick_size!r} is not a clean reciprocal")
        return inv
    price_cols = [c for n, c in cols.items() if n not in ("timestamp", "volume")]
    decimals = max(detect_decimals(raw.column(c).to_pylist()) for c in price_cols)
    return 10 ** decimals


def _read_timestamps(path: Path, raw: pa.Table, col: str) -> np.ndarray:
    parsed = pacsv.read_csv(
        path, convert_options=pacsv.ConvertOptions(column_types={col: pa.timestamp("ms")})
    ).column(col)
    ms = parsed.cast(pa.int64()).to_numpy()
    _verify_timestamp_parse(path, raw.column(col)[0].as_py(), ms)
    return ms


def _verify_timestamp_parse(path: Path, first_raw, ms: np.ndarray) -> None:
    """Cross-check Arrow's first timestamp against a stdlib parse.

    Cheap insurance against a silent unit error: an ingest wrong by a factor of
    1000 produces a store that looks superficially fine and is entirely useless.
    """
    if ms.shape[0] == 0 or first_raw is None:
        return
    text = str(first_raw)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        expected = int(parsed.timestamp() * 1000)
        if int(ms[0]) != expected:
            raise ValueError(
                f"{path.name}: timestamp parse mismatch on row 0 — arrow gave "
                f"{int(ms[0])} ms, stdlib gave {expected} ms for {text!r}. "
                "Refusing to build a store on an ambiguous time unit."
            )
        return
    # Unrecognized layout: Arrow parsed it, we simply could not double-check.


def _to_float(col: pa.ChunkedArray) -> np.ndarray:
    return col.cast(pa.float64()).to_numpy(zero_copy_only=False)


def _to_scaled(prices: np.ndarray, scale: int) -> np.ndarray:
    scaled = np.rint(prices * scale)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("non-finite price in source data")
    if scaled.min() >= np.iinfo(np.int32).min and scaled.max() <= np.iinfo(np.int32).max:
        return scaled.astype(np.int32)
    return scaled.astype(np.int64)


def _assess(ms, fields, scale, out_of_order, digest) -> QualityReport:
    dup = np.diff(ms) == 0 if ms.size > 1 else np.array([], dtype=bool)
    gaps = np.diff(ms) if ms.size > 1 else np.array([], dtype=np.int64)

    if "bid" in fields:
        spread = (fields["ask"] - fields["bid"]) / scale
        lo, hi = fields["bid"], fields["ask"]
    elif "spread" in fields:
        spread = fields["spread"] / scale
        lo = hi = fields["close"]
    else:
        spread = np.zeros(0)
        lo = hi = fields["close"]

    return QualityReport(
        rows=int(ms.shape[0]),
        file_sha256=digest,
        first_ms=int(ms[0]) if ms.size else 0,
        last_ms=int(ms[-1]) if ms.size else 0,
        out_of_order=out_of_order,
        duplicate_timestamps=int(dup.sum()),
        max_duplicate_run=int(_max_run(dup)),
        negative_spread=int((hi < lo).sum()),
        zero_spread=int((hi == lo).sum()) if "bid" in fields else 0,
        bad_price=int((lo <= 0).sum() + (hi <= 0).sum()),
        spread_mean=float(spread.mean()) if spread.size else 0.0,
        spread_p50=float(np.percentile(spread, 50)) if spread.size else 0.0,
        spread_p99=float(np.percentile(spread, 99)) if spread.size else 0.0,
        largest_gaps_ms=sorted(gaps[np.argsort(gaps)[-5:]].tolist(), reverse=True)
        if gaps.size else [],
    )


def _max_run(flags: np.ndarray) -> int:
    if flags.size == 0 or not flags.any():
        return 0
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def _name_for_ms(ms: int) -> str:
    if ms == 0:
        return "TICK"
    for name, value in TIMEFRAMES.items():
        if value == ms:
            return name
    return f"{ms}ms"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
