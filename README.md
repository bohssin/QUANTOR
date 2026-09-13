# QUANTOR

Local AI trading agent: chart on the left, agent sidebar on the right. The agent writes
strategy logic in Python, backtests it against your own data, optimizes and validates it, and
writes the result out as MQL5 or Pine Script on request.

A **platform, not a script runner** — strategies, versions, runs, verdicts, exports and
conversations are all catalogued and browsable later. Nothing generated is thrown away.

**Start here: [`trading-agent-plan.md`](trading-agent-plan.md).** Nothing is built yet beyond
the indicator layer and its parity harness — the plan is the design handoff, and §0 asks you
to argue with it.

**Reference platform is MetaTrader 5** throughout: tick history → bars at a chosen timeframe,
MT5 modeling modes as the fidelity dial, MT5 Strategy Tester layout for the UI, and the
owner's own MT5 backtests as the verification ground truth.

## What exists

| Path | State |
|---|---|
| `trading-agent-plan.md` | Design plan, rev 4 |
| `engine/indicators/` | Wilder-family indicators, validated against a PineTS oracle |
| `engine/store/` | CSV ingest — ticks or bars, precision and resolution read from the file |
| `tests/` | 30 passing: indicator parity + ingest |
| `tools/pinets_oracle/` | Dev-only fixture generator (AGPL, never shipped) |
| `bench/` | Reproduces every `[measured]` number in the plan |

## Quick start

    pip install numpy pandas numba pyarrow pytest
    python3 -m pytest tests/ -v            # indicator parity + tick ingest
    python3 bench/bench_indicators.py      # signal-pass cost
    python3 bench/bench_ticks.py           # ingest, bars, tick-resolution fills

**Data in:** your own CSVs — ticks or bars, any instrument, any precision, each with its own
GMT offset. The file's resolution decides which strategy timeframes it can serve.

**Export:** the agent writes MQL5 (to run) or Pine Script (to view on TradingView) on request.
Every export carries a measured parity verdict — an AI translation is a claim until checked.
See plan §22.

## Licence note

The shipped system has no copyleft: Vela is Apache-2.0, the Python stack is BSD/MIT/Apache,
everything else MIT. PineTS (AGPL-3.0-only) is a **development-only** test oracle under
`tools/pinets_oracle/` and is never imported by `engine/` or distributed. See plan §3.
