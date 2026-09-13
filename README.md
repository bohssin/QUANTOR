# QUANTOR

Local AI trading agent: chart on the left, agent sidebar on the right. The agent writes
strategy logic in Python, backtests it against your own tick data, optimizes and validates
it, across any instrument.

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
| `engine/store/` | Tick CSV ingest — reads the real feed format, verifies the time unit |
| `tests/` | 17 passing: indicator parity + ingest |
| `tools/pinets_oracle/` | Dev-only fixture generator (AGPL, never shipped) |
| `bench/` | Reproduces every `[measured]` number in the plan |

## Quick start

    pip install numpy pandas numba pyarrow pytest
    python3 -m pytest tests/ -v            # indicator parity + tick ingest
    python3 bench/bench_indicators.py      # signal-pass cost
    python3 bench/bench_ticks.py           # ingest, bars, tick-resolution fills

Strategies export to **MQL5** (to run) and **Pine** (to view on TradingView), each with a
parity verdict rather than on trust — see plan §21.

## Licence note

The shipped system has no copyleft: Vela is Apache-2.0, the Python stack is BSD/MIT/Apache,
everything else MIT. PineTS (AGPL-3.0-only) is a **development-only** test oracle under
`tools/pinets_oracle/` and is never imported by `engine/` or distributed. See plan §3.
