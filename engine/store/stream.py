"""Constant-memory ingest for files too large to hold. Plan §4.2, §4.4.

`read_csv` materializes the whole file — twice, in fact, once for the values and
once to parse timestamps. That is fine for a year of M15 bars and impossible for
the owner's 11 GB XAUUSD tick file: ~280M rows, 4.5-6.7 GB resident before
Arrow's own parse buffers, and then again for the second pass.

This module reads the same files in **bounded memory**: `pyarrow.csv.open_csv`
yields record batches, each batch is folded into the bar accumulator, and the
batch is released. Peak RSS is set by `block_size`, not by file size. A 100 MB
file and a 100 GB file cost the same.

What that costs, stated plainly:

**No global sort.** `read_csv` stable-sorts by timestamp; a stream cannot,
because row 280,000,000 could in principle belong at the front. Tick exports are
chronological by construction, so out-of-order rows are **counted and reported**
rather than silently repaired. A file with many of them is a broken file and the
quality report says so.

**Percentiles from a sample.** Exact p50/p99 of 280M spreads needs 280M spreads.
A fixed-size reservoir (deterministic, seeded) gives them to within sampling
error. The count-based fields — rows, duplicates, out-of-order, bad prices — are
exact.

**Fingerprint, not full hash, by default.** sha256 of 11 GB means reading 11 GB
again. The default identity is size plus the head and tail blocks, which is
enough to notice a changed file and honest about what it is. `full_sha256=True`
when the real digest is wanted.

**Prices stay floats.** The scaled-integer store (§4.2) exists to make stop and
target comparisons exact on data held in memory. Here the ticks are consumed and
dropped — only bars survive — and bars are floats everywhere downstream.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
from numba import njit

from .ingest import (
    QualityReport,
    SourceSpec,
    _read_header,
    _resolve_columns,
    detect_decimals,
    timeframe_ms,
)

__all__ = [
    "BarAccumulator",
    "StreamResult",
    "stream_bars",
    "stream_batches",
    "fingerprint",
]

_MS_PER_HOUR = 3_600_000
_MS_PER_DAY = 86_400_000

#: Bytes per record batch, and the single number that decides whether this
#: module keeps its promise. `[measured]`, folding a 0.4 GB and a 1.6 GB tick
#: file with `bench/bench_stream.py`:
#:
#:     block   threads   0.4 GB    1.6 GB    ratio
#:      64 MB     on      918 MB   1962 MB    2.14
#:      64 MB     off     953 MB   2051 MB    2.15
#:      16 MB     off     774 MB    984 MB    1.27
#:       8 MB     off     674 MB    673 MB    1.00
#:       4 MB     off     486 MB    484 MB    1.00
#:
#: The growth was never this module's state — it is Arrow's CSV readahead,
#: which queues blocks in proportion to `block_size` and thread count. At 4 MB
#: with readahead off, peak RSS is flat in file size *and* the fastest of the
#: six, so a 11 GB file costs exactly what a 0.4 GB file costs.
DEFAULT_BLOCK_SIZE = 4 << 20

#: Reservoir size for spread percentiles. 200k samples put the p99 estimate
#: within ~0.3% of exact at any file size.
RESERVOIR = 200_000


# --- the incremental kernel --------------------------------------------------

@njit(cache=True, fastmath=False)
def _build_chunk(t, bid, ask, tf_ms,
                 c_boundary, c_o, c_h, c_l, c_c, c_vol, c_spsum, c_spmax):
    """Fold one batch of ticks into bars, carrying a partial bar in and out.

    Returns every bar that is **provably complete** — a later tick opened a new
    boundary — plus the still-open bar as loose scalars. The open bar is not
    emitted, because the next batch may add ticks to it. That is the whole
    correctness question in this module: a bar split across a batch boundary
    must merge, never duplicate.

    Mirrors `bars.py::_build` exactly, including which side of a tie wins and
    the order of the running-sum updates, so that the streamed result is
    bit-identical to the whole-file result. `tests/store/test_stream.py` asserts
    that rather than trusting it.
    """
    n = t.shape[0]
    cap = n + 1
    bar_t = np.empty(cap, np.int64)
    o = np.empty(cap, np.float64)
    h = np.empty(cap, np.float64)
    low = np.empty(cap, np.float64)
    c = np.empty(cap, np.float64)
    vol = np.empty(cap, np.int64)
    sp_mean = np.empty(cap, np.float64)
    sp_max = np.empty(cap, np.float64)

    k = -1
    current = c_boundary
    sp_sum = 0.0

    if c_boundary >= 0:                       # resume the carried-in bar
        k = 0
        bar_t[0] = c_boundary
        o[0] = c_o
        h[0] = c_h
        low[0] = c_l
        c[0] = c_c
        vol[0] = c_vol
        sp_max[0] = c_spmax
        sp_sum = c_spsum

    for i in range(n):
        boundary = (t[i] // tf_ms) * tf_ms
        spread = ask[i] - bid[i]
        if boundary != current:
            if k >= 0:
                sp_mean[k] = sp_sum / vol[k]
            k += 1
            current = boundary
            bar_t[k] = boundary
            o[k] = bid[i]
            h[k] = bid[i]
            low[k] = bid[i]
            c[k] = bid[i]
            vol[k] = 1
            sp_sum = spread
            sp_max[k] = spread
        else:
            if bid[i] > h[k]:
                h[k] = bid[i]
            if bid[i] < low[k]:
                low[k] = bid[i]
            c[k] = bid[i]
            vol[k] += 1
            sp_sum += spread
            if spread > sp_max[k]:
                sp_max[k] = spread

    # Everything below k is closed; k itself stays open and goes out as carry.
    if k < 0:                                  # empty batch, nothing to carry
        return (bar_t[:0], o[:0], h[:0], low[:0], c[:0], vol[:0],
                sp_mean[:0], sp_max[:0],
                np.int64(-1), 0.0, 0.0, 0.0, 0.0, np.int64(0), 0.0, 0.0)

    return (bar_t[:k], o[:k], h[:k], low[:k], c[:k], vol[:k],
            sp_mean[:k], sp_max[:k],
            bar_t[k], o[k], h[k], low[k], c[k], vol[k], sp_sum, sp_max[k])


class BarAccumulator:
    """Builds bars from ticks arriving in arbitrary chunks.

    The stateful twin of `bars_from_ticks`. Feed it any partition of a tick
    series and `finish()` returns exactly what the whole-array builder would
    have returned for the concatenation.
    """

    _FIELDS = ("ms", "open", "high", "low", "close", "volume", "spread", "spread_max")

    def __init__(self, timeframe: str) -> None:
        self.tf_ms = timeframe_ms(timeframe)
        if self.tf_ms <= 0:
            raise ValueError(f"{timeframe!r} is not a bar timeframe")
        self.timeframe = timeframe.strip().upper()
        self._closed: list[tuple[np.ndarray, ...]] = []
        self._carry = (np.int64(-1), 0.0, 0.0, 0.0, 0.0, np.int64(0), 0.0, 0.0)

    def push(self, ms: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> int:
        """Add one chunk. Returns how many bars closed as a result."""
        if ms.shape[0] == 0:
            return 0
        out = _build_chunk(
            np.ascontiguousarray(ms, dtype=np.int64),
            np.ascontiguousarray(bid, dtype=np.float64),
            np.ascontiguousarray(ask, dtype=np.float64),
            np.int64(self.tf_ms),
            *self._carry,
        )
        closed, carry = out[:8], out[8:]
        if closed[0].shape[0]:
            self._closed.append(tuple(np.copy(a) for a in closed))
        self._carry = carry
        return int(closed[0].shape[0])

    def finish(self) -> dict[str, np.ndarray]:
        """Close the open bar and return the full series."""
        pieces = list(self._closed)
        boundary = self._carry[0]
        if boundary >= 0:
            vol = self._carry[5]
            pieces.append((
                np.array([boundary], np.int64),
                np.array([self._carry[1]], np.float64),
                np.array([self._carry[2]], np.float64),
                np.array([self._carry[3]], np.float64),
                np.array([self._carry[4]], np.float64),
                np.array([vol], np.int64),
                np.array([self._carry[6] / vol], np.float64),   # sp_sum -> mean
                np.array([self._carry[7]], np.float64),
            ))
        if not pieces:
            empty = np.empty(0)
            return {"ms": np.empty(0, np.int64), "open": empty, "high": empty,
                    "low": empty, "close": empty, "volume": np.empty(0, np.int64),
                    "spread": empty, "spread_max": empty}
        return {name: np.concatenate([p[i] for p in pieces])
                for i, name in enumerate(self._FIELDS)}

    @property
    def n_closed(self) -> int:
        return sum(int(p[0].shape[0]) for p in self._closed)


# --- streaming quality -------------------------------------------------------

class _Quality:
    """Exact counts, sampled percentiles. Folded one batch at a time."""

    def __init__(self, seed: int = 12345) -> None:
        self.rows = 0
        self.first_ms = 0
        self.last_ms = 0
        self.out_of_order = 0
        self.duplicates = 0
        self.max_dup_run = 0
        self._dup_run = 0
        self.negative_spread = 0
        self.zero_spread = 0
        self.bad_price = 0
        self._spread_sum = 0.0
        self._reservoir = np.empty(RESERVOIR, np.float64)
        self._seen = 0
        self._rng = np.random.default_rng(seed)
        self._gaps: list[int] = []
        self._prev_ms: int | None = None

    def push(self, ms: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        n = ms.shape[0]
        if n == 0:
            return
        first_batch = self.rows == 0
        if first_batch:
            self.first_ms = int(ms[0])
        self.last_ms = int(ms[-1])

        # Diffs measured across the batch seam as well as inside it: a weekend
        # gap that straddles two batches is one gap, not two halves of nothing.
        # On the very first batch there is no predecessor, so row 0 is skipped.
        if first_batch:
            gaps = np.diff(ms)
        else:
            gaps = np.empty(n, np.int64)
            gaps[0] = ms[0] - self._prev_ms
            gaps[1:] = np.diff(ms)
        self._prev_ms = int(ms[-1])

        self.out_of_order += int((gaps < 0).sum())
        dup = gaps == 0
        self.duplicates += int(dup.sum())
        self.max_dup_run = max(self.max_dup_run, self._run_of_duplicates(dup))

        if gaps.size:
            top = gaps[np.argsort(gaps)[-5:]].tolist()
            self._gaps = sorted(self._gaps + top, reverse=True)[:5]

        spread = hi - lo
        self.negative_spread += int((spread < 0).sum())
        self.zero_spread += int((spread == 0).sum())
        self.bad_price += int((lo <= 0).sum() + (hi <= 0).sum())
        self._spread_sum += float(spread.sum())
        self._sample(spread)
        self.rows += n

    def _run_of_duplicates(self, dup: np.ndarray) -> int:
        """Longest run of equal timestamps, continuing across batch seams."""
        best = run = self._dup_run
        for f in dup:
            run = run + 1 if f else 0
            best = max(best, run)
        self._dup_run = run
        return best

    def _sample(self, values: np.ndarray) -> None:
        """Reservoir sampling, so percentiles do not need the whole column."""
        n = values.shape[0]
        room = RESERVOIR - self._seen
        if room > 0:
            take = min(room, n)
            self._reservoir[self._seen:self._seen + take] = values[:take]
            self._seen += take
            values, n = values[take:], n - take
            if n == 0:
                return
        # Each later value replaces a random slot with probability k/i.
        idx = np.arange(self._seen + 1, self._seen + n + 1, dtype=np.float64)
        keep = self._rng.random(n) < (RESERVOIR / idx)
        if keep.any():
            slots = self._rng.integers(0, RESERVOIR, int(keep.sum()))
            self._reservoir[slots] = values[keep]
        self._seen += n

    def report(self, digest: str) -> QualityReport:
        sample = self._reservoir[:min(self._seen, RESERVOIR)]
        return QualityReport(
            rows=self.rows,
            file_sha256=digest,
            first_ms=self.first_ms,
            last_ms=self.last_ms,
            out_of_order=self.out_of_order,
            duplicate_timestamps=self.duplicates,
            max_duplicate_run=self.max_dup_run,
            negative_spread=self.negative_spread,
            zero_spread=self.zero_spread,
            bad_price=self.bad_price,
            spread_mean=self._spread_sum / self.rows if self.rows else 0.0,
            spread_p50=float(np.percentile(sample, 50)) if sample.size else 0.0,
            spread_p99=float(np.percentile(sample, 99)) if sample.size else 0.0,
            largest_gaps_ms=self._gaps,
        )


# --- the public streaming API ------------------------------------------------

@dataclass
class StreamResult:
    """Bars plus everything learned on the way past."""

    bars: dict[str, np.ndarray]
    quality: QualityReport
    kind: str
    timeframe: str
    resolution_ms: int
    tick_size: float
    detected: dict[str, object] = field(default_factory=dict)
    batches: int = 0
    seconds: float = 0.0

    def __len__(self) -> int:
        return int(self.bars["ms"].shape[0])

    def summary(self) -> str:
        span_days = (self.quality.last_ms - self.quality.first_ms) / _MS_PER_DAY
        return (
            f"{self.quality.rows:,} {self.kind} rows -> {len(self):,} "
            f"{self.timeframe} bars over {span_days:.1f}d "
            f"in {self.seconds:.1f}s ({self.batches} batches)\n"
            f"{self.quality.summary()}"
        )


def stream_batches(
    path: str | Path,
    spec: SourceSpec | None = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray], dict]]:
    """Yield `(ms, columns, meta)` per record batch. Never holds the file.

    `meta` is constant across batches and carries what was detected from the
    header and the first batch: column mapping, kind, tick size.
    """
    spec = spec or SourceSpec()
    path = Path(path)
    header = _read_header(path)
    cols, kind = _resolve_columns(header, spec, path)
    ts_col = cols["timestamp"]
    value_cols = {n: c for n, c in cols.items() if n != "timestamp"}

    types: dict[str, pa.DataType] = {ts_col: pa.timestamp("ms")}
    for name, col in value_cols.items():
        types[col] = pa.int64() if name == "volume" else pa.float64()

    reader = pacsv.open_csv(
        path,
        # `use_threads=False` is not about CPU — it is about memory. Threaded
        # reading keeps several blocks in flight, and the queue grows with the
        # file. Off, it is both bounded and (measurably) faster here, because
        # the work downstream is the bar fold, not the parse.
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=False),
        convert_options=pacsv.ConvertOptions(
            column_types=types, include_columns=list(cols.values())
        ),
    )

    shift = int(round(spec.utc_offset_hours * _MS_PER_HOUR)) if spec.utc_offset_hours else 0
    meta = {"kind": kind, "columns": cols, "tick_size": None, "checked_first_row": False}
    first = True

    try:
        for batch in reader:
            ms = batch.column(ts_col).cast(pa.int64()).to_numpy(zero_copy_only=False)
            if shift:
                ms = ms - shift
            values = {
                name: batch.column(col).to_numpy(zero_copy_only=False)
                for name, col in value_cols.items()
            }
            if first:
                _verify_first_row(path, ms, shift)
                meta["checked_first_row"] = True
                meta["tick_size"] = _tick_size_from(path, spec, cols)
                first = False
            yield ms, values, meta
    finally:
        reader.close()


def stream_bars(
    path: str | Path,
    timeframe: str,
    spec: SourceSpec | None = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    full_sha256: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> StreamResult:
    """Build bars at `timeframe` from a CSV of any size, in bounded memory.

    This is the path for real tick archives. `progress(rows, bars)` is called
    once per batch, which for an 11 GB file is roughly every 64 MB.
    """
    import time

    spec = spec or SourceSpec()
    path = Path(path)
    t0 = time.perf_counter()

    # Refuse before opening anything, rather than partway through a 64 MB batch.
    _, kind = _resolve_columns(_read_header(path), spec, path)
    if kind != "tick":
        raise NotImplementedError(
            f"{path.name} is a bar source; streaming exists for tick archives that "
            "do not fit in memory. Bar files do — use read_csv for those."
        )

    acc = BarAccumulator(timeframe)
    quality = _Quality()
    batches = 0
    meta: dict = {}

    for ms, values, meta in stream_batches(path, spec, block_size=block_size):
        bid, ask = values["bid"], values["ask"]
        acc.push(ms, bid, ask)
        quality.push(ms, bid, ask)
        batches += 1
        if progress is not None:
            progress(quality.rows, acc.n_closed)

    bars = acc.finish()
    digest = _sha256_full(path) if full_sha256 else fingerprint(path)
    tick_size = float(meta.get("tick_size") or 0.0) or 1e-5

    return StreamResult(
        bars=bars,
        quality=quality.report(digest),
        kind=kind,
        timeframe=acc.timeframe,
        resolution_ms=0 if kind == "tick" else timeframe_ms(acc.timeframe),
        tick_size=tick_size,
        detected={
            "kind": kind,
            "columns": meta.get("columns", {}),
            "tick_size": tick_size,
            "utc_offset_hours": spec.utc_offset_hours,
            "digest_kind": "sha256" if full_sha256 else "fingerprint",
        },
        batches=batches,
        seconds=time.perf_counter() - t0,
    )


# --- identity ----------------------------------------------------------------

def fingerprint(path: str | Path, block: int = 1 << 20) -> str:
    """Cheap content key: size plus the head and tail blocks.

    Not a digest of the file and never presented as one — `StreamResult.detected`
    records which was used. It changes whenever the file changes at either end or
    in length, which is what identity in the library actually needs, and it costs
    two seeks instead of reading 11 GB.
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"size={size}\n".encode())
    with path.open("rb") as fh:
        h.update(fh.read(block))
        if size > block:
            fh.seek(max(0, size - block))
            h.update(fh.read(block))
    return "fp:" + h.hexdigest()


def _sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --- internals ---------------------------------------------------------------

def _verify_first_row(path: Path, ms: np.ndarray, shift: int) -> None:
    """Re-parse row 0 with the stdlib and compare. Plan §4.2.

    The same cross-check `read_csv` does, against the same failure: a timestamp
    column read in the wrong unit produces a store that looks fine and is off by
    a factor of 1000. Only row 0 is checked, which is all it takes to catch a
    unit error — they are never per-row.
    """
    if ms.shape[0] == 0:
        return
    with path.open("r", encoding="utf-8-sig") as fh:
        fh.readline()
        line = fh.readline().strip()
    if not line:
        return
    text = line.split(",")[0].strip().strip('"')
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        expected = int(parsed.timestamp() * 1000) - shift
        if int(ms[0]) != expected:
            raise ValueError(
                f"{path.name}: timestamp parse mismatch on row 0 — arrow gave "
                f"{int(ms[0])} ms, stdlib gave {expected} ms for {text!r}. "
                "Refusing to build bars on an ambiguous time unit."
            )
        return
    # Unrecognized layout: Arrow parsed it, we simply could not double-check.


def _tick_size_from(path: Path, spec: SourceSpec, cols: dict[str, str]) -> float:
    """Precision from the first rows of the file, read as text.

    `read_csv` scans the whole price column; here the first 50k rows decide.
    A feed does not change its precision partway through a file, and if one did,
    the bars are floats either way — this value is reported, not relied on.
    """
    if spec.tick_size is not None:
        return float(spec.tick_size)
    price_cols = [c for n, c in cols.items() if n not in ("timestamp", "volume")]
    idx = {name: i for i, name in enumerate(_read_header(path))}
    best = 0
    with path.open("r", encoding="utf-8-sig") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= 50_000:
                break
            parts = line.rstrip("\n").split(",")
            for c in price_cols:
                j = idx.get(c, -1)
                if 0 <= j < len(parts):
                    best = max(best, detect_decimals([parts[j]]))
    return 1.0 / (10 ** best)
