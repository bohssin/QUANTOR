# Benchmarks

Reproduce the `[measured]` numbers in the plan. Run on a 4-core / 16 GB Linux box
against synthetic data; re-measure on real tick files before capacity planning
(real ticks cluster around news, synthetic ones are uniform — see plan §18,
Probe 2).

    python3 bench/bench_indicators.py    # plan §2, §11 — signal-pass cost
    python3 bench/bench_ticks.py         # plan §4.1, §6 — ingest, bars, fills

`bench_ticks.py` generates its own ~306 MB tick CSV on first run (~20 s) and
caches it. Delete `bench/_data/` to regenerate.
