"""Prove the streaming ingest is constant-memory. Plan §4.2.

The claim that matters for the owner's 11 GB file cannot be tested by loading an
11 GB file — it is tested by showing that **peak RSS does not grow with file
size**. If 0.5 GB and 2 GB cost the same resident memory, 11 GB costs the same
resident memory, and the whole-file path's 4.5-6.7 GB never happens.

    python3 bench/bench_stream.py            # ~0.5 GB and ~2 GB
    python3 bench/bench_stream.py --sizes 0.25 1 4

Each size is measured in a **fresh subprocess**, because peak RSS is a
high-water mark: measuring both in one process would report the larger for both
and prove nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "bench" / "_data"

#: Average bytes per row of `2021-01-04 01:00:00.413,1904.998,1905.366`.
BYTES_PER_ROW = 42


def generate(path: Path, target_bytes: int, seed: int = 3) -> int:
    """Write a XAUUSD-shaped tick CSV of roughly `target_bytes`."""
    if path.exists() and abs(path.stat().st_size - target_bytes) < target_bytes * 0.05:
        return path.stat().st_size

    rows = int(target_bytes / BYTES_PER_ROW)
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    chunk = 1_000_000
    t = 1_609_722_000_000
    price = 1900.0
    with path.open("w", buffering=1 << 22) as fh:
        fh.write("timestamp,bidPrice,askPrice\n")
        written = 0
        while written < rows:
            n = min(chunk, rows - written)
            gaps = rng.integers(1, 900, n).astype(np.int64)
            ms = t + np.cumsum(gaps)
            t = int(ms[-1])
            bid = price + np.cumsum(rng.normal(0, 0.01, n))
            price = float(bid[-1])
            ask = bid + rng.uniform(0.10, 0.40, n)
            # Formatting dominates generation cost; build the block as one string.
            stamps = (np.datetime64(0, "ms") + ms.astype("timedelta64[ms]"))
            body = "\n".join(
                f"{s},{b:.3f},{a:.3f}"
                for s, b, a in zip(np.datetime_as_string(stamps, unit="ms"), bid, ask)
            )
            fh.write(body.replace("T", " ") + "\n")
            written += n
    return path.stat().st_size


def measure_one(path: Path, timeframe: str, block_mb: int) -> dict:
    """Run the stream in this process and report peak RSS. Called via --measure."""
    from engine.store import stream_bars

    t0 = time.perf_counter()
    out = stream_bars(path, timeframe, block_size=block_mb << 20)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "file_mb": round(path.stat().st_size / (1 << 20), 1),
        "rows": out.quality.rows,
        "bars": len(out),
        "batches": out.batches,
        "seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(peak_kb / 1024, 1),
        "rows_per_s": int(out.quality.rows / max(time.perf_counter() - t0, 1e-9)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=float, default=[0.5, 2.0],
                    help="file sizes in GB")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--block-mb", type=int, default=64)
    ap.add_argument("--measure", help=argparse.SUPPRESS)
    ap.add_argument("--keep", action="store_true", help="do not delete the files")
    args = ap.parse_args()

    if args.measure:                              # the child process
        print(json.dumps(measure_one(Path(args.measure), args.timeframe, args.block_mb)))
        return 0

    print(f"\nstreaming ingest — peak RSS vs file size  "
          f"(block_size={args.block_mb} MB, {args.timeframe})\n")
    print(f"  {'file':>10} {'rows':>14} {'bars':>9} {'batches':>8} "
          f"{'sec':>7} {'rows/s':>11} {'peak RSS':>10}")

    results = []
    made = []
    for gb in args.sizes:
        path = DATA / f"stream_{gb:g}gb.csv"
        size = generate(path, int(gb * (1 << 30)))
        made.append(path)
        proc = subprocess.run(
            [sys.executable, __file__, "--measure", str(path),
             "--timeframe", args.timeframe, "--block-mb", str(args.block_mb)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            return 1
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"  {r['file_mb']:>9.0f}M {r['rows']:>14,} {r['bars']:>9,} "
              f"{r['batches']:>8} {r['seconds']:>7.1f} {r['rows_per_s']:>11,} "
              f"{r['peak_rss_mb']:>9.0f}M")

    if len(results) >= 2:
        size_ratio = results[-1]["file_mb"] / results[0]["file_mb"]
        rss_ratio = results[-1]["peak_rss_mb"] / results[0]["peak_rss_mb"]
        print(f"\n  file grew {size_ratio:.1f}x, peak RSS grew {rss_ratio:.2f}x")
        verdict = "CONSTANT-MEMORY" if rss_ratio < 1.35 else "GROWS WITH FILE SIZE"
        print(f"  verdict: {verdict}")
        biggest = max(r["file_mb"] for r in results)
        eleven_gb = 11 * 1024
        slowest = min(r["rows_per_s"] for r in results)
        rows_11gb = int(eleven_gb * (1 << 20) / BYTES_PER_ROW)
        print(f"\n  extrapolating to 11 GB ({rows_11gb:,} rows) at the measured "
              f"{slowest:,} rows/s:")
        print(f"    ~{rows_11gb / slowest / 60:.0f} min, peak RSS still "
              f"~{results[-1]['peak_rss_mb']:.0f} MB")
        print(f"    (largest file actually measured: {biggest:.0f} MB)")

    if not args.keep:
        for p in made:
            p.unlink(missing_ok=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
