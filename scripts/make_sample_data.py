"""Generate a XAUUSD-shaped tick CSV for testing the whole pipeline.

Why this exists: the owner's real archive is 11 GB on a Windows machine and the
build has to be verifiable without it. This produces a file with the same
*shape* — GMT+3 timestamps, `timestamp,bidPrice,askPrice`, three decimals,
weekend gaps, session-varying spread and tick rate — so the ingest, the streaming
path, the bar builder and the GMT offset all get exercised on something
realistic.

**It is synthetic, and nothing measured on it says anything about markets.**
Two structures are deliberately built in so the optimizer and walk-forward have
something real to find rather than pure noise:

  1. **Intraday mean reversion**, strongest in the thin Asian session — an
     Ornstein-Uhlenbeck pull toward a slow-moving anchor.
  2. **Trending regimes** that switch on a slow Markov chain, so a single
     parameter set cannot be right everywhere and walk-forward has a real
     question to answer.

Any edge found here is an edge someone put here on purpose. The pipeline is what
is being proven, not the strategy.

    python3 scripts/make_sample_data.py --out data/xau_sample.csv --days 730
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

#: The file's clock, matching the owner's export.
GMT_OFFSET_HOURS = 3.0

#: Ticks per minute by session, in the file's own (GMT+3) hours. Gold is thin
#: overnight and busy once London and then New York are open.
SESSION_RATE = {
    "asia":   (1, 8, 6.0),      # 01:00-08:00 GMT+3
    "london": (8, 16, 22.0),
    "ny":     (16, 23, 30.0),
    "late":   (23, 25, 4.0),
}

#: Spread in dollars by session. Wider when thin, which is what actually costs.
SESSION_SPREAD = {"asia": 0.34, "london": 0.18, "ny": 0.15, "late": 0.45}


def session_for(hour: int) -> str:
    for name, (lo, hi, _) in SESSION_RATE.items():
        if lo <= hour < hi:
            return name
    return "late"


def generate(out: Path, days: int, seed: int, start: str,
             ticks_scale: float) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    begin = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

    price = 1830.0
    anchor = price
    regime = 0                      # 0 = range, 1 = trend up, 2 = trend down
    drift = 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    minutes_written = 0

    with out.open("w", buffering=1 << 22) as fh:
        fh.write("timestamp,bidPrice,askPrice\n")

        for day in range(days):
            date = begin + dt.timedelta(days=day)
            weekday = date.weekday()                 # 0 = Monday
            if weekday == 5:                          # Saturday: closed
                continue

            for hour in range(24):
                # Friday closes at 23:00 in the file's clock; Sunday opens 01:00.
                if weekday == 4 and hour >= 23:
                    continue
                if weekday == 6 and hour < 1:
                    continue

                name = session_for(hour)
                rate = SESSION_RATE[name][2] * ticks_scale
                base_spread = SESSION_SPREAD[name]

                for minute in range(60):
                    n = int(rng.poisson(rate))
                    if n <= 0:
                        continue

                    # Regime switches roughly every few days.
                    if rng.random() < 1 / (60 * 24 * 3):
                        regime = int(rng.integers(0, 3))
                        drift = {0: 0.0, 1: 1.0, 2: -1.0}[regime] * rng.uniform(0.4, 1.1)

                    # The anchor drifts slowly; price is pulled toward it. The
                    # pull is strongest when the book is thin, which is the
                    # documented shape of intraday mean reversion.
                    anchor += rng.normal(drift * 0.0016, 0.010)
                    pull = {"asia": 0.055, "late": 0.050,
                            "london": 0.012, "ny": 0.008}[name]

                    vol = {"asia": 0.020, "london": 0.048,
                           "ny": 0.055, "late": 0.016}[name]
                    if regime:
                        vol *= 1.25

                    steps = rng.normal(0.0, vol, n)
                    steps += pull * (anchor - price) / max(n, 1)
                    bids = price + np.cumsum(steps)
                    price = float(bids[-1])

                    spread = np.abs(rng.normal(base_spread, base_spread * 0.22, n))
                    spread = np.clip(spread, 0.05, 3.0)

                    stamp = date.replace(hour=hour, minute=minute)
                    offsets = np.sort(rng.integers(0, 60_000, n))
                    ms = int(stamp.timestamp() * 1000) + offsets

                    stamps = np.datetime_as_string(
                        np.datetime64(0, "ms") + ms.astype("timedelta64[ms]"), unit="ms")
                    fh.write("\n".join(
                        f"{s.replace('T', ' ')},{b:.3f},{b + sp:.3f}"
                        for s, b, sp in zip(stamps, bids, spread)) + "\n")
                    rows += n
                    minutes_written += 1

    return rows, minutes_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/xau_sample.csv")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=20240115)
    ap.add_argument("--start", default="2022-01-03")
    ap.add_argument("--ticks-scale", type=float, default=1.0,
                    help="scale the tick rate; 0.25 makes a much smaller file")
    args = ap.parse_args()

    out = Path(args.out)
    rows, minutes = generate(out, args.days, args.seed, args.start, args.ticks_scale)
    size = out.stat().st_size
    print(f"wrote {out}")
    print(f"  {rows:,} ticks over {args.days} calendar days "
          f"({minutes:,} populated minutes)")
    print(f"  {size / (1 << 20):,.0f} MB  ({size / max(rows, 1):.0f} bytes/row)")
    print(f"  clock: GMT+{GMT_OFFSET_HOURS:g} — load with utc_offset_hours={GMT_OFFSET_HOURS:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
