# QUANTOR

A local AI trading agent. You talk to it; it writes strategies in Python,
backtests them against your own tick or bar data, optimizes and validates them,
and draws the result on a chart so you can see where it actually traded.

**MCP-first, no API keys.** Claude Code runs on your own subscription and
reaches everything through MCP: the LuxAlgo Library, Edge Stats, and QUANTOR's
own engine. One artifact per strategy — a Python signal block.

**A platform, not a script runner.** Strategies, versions, runs, optimization
tables, verdicts and the conversations that produced them are all catalogued.
Nothing generated is thrown away, including the rejections — the failure record
is what stops the agent re-proposing a dead end in three weeks.

**Reference platform is MetaTrader 5**: tick history → bars at a chosen
timeframe, MT5 modeling modes as the fidelity dial, MT5 Strategy Tester layout
for the UI.

The design document is [`trading-agent-plan.md`](trading-agent-plan.md), and its
§0 asks you to argue with it. The engine, the MCP server, the library, the dark
UI and the chart are built and tested end to end.

## What exists

| Path | State |
|---|---|
| `trading-agent-plan.md` | The design document — read this first |
| `scripts/setup.sh` | One-command local setup |
| `quantor_mcp/doctor.py` | Health check that names the exact fix |
| `engine/indicators/` | Wilder-family indicators, validated against a PineTS oracle |
| `engine/store/` | CSV ingest + tick→bar construction (MT5 semantics) |
| `engine/backtest/` | Bar-mode fill engine + metrics |
| `engine/optimize/` | Grid and random search, objectives with guards |
| `engine/validate/` | Fold geometry, walk-forward |
| `engine/signal/` | Static validator for generated signal blocks (§9) |
| `quantor_mcp/` | The engine as an MCP server — agent-tested end to end (§14.2) |
| `engine/optimize/plateau.py` | Neighbourhood robustness — plateaus over peaks (§11) |
| `bench/demo_loop.py` | End-to-end: signal → backtest → optimize → walk-forward |
| `bench/probe_vectorbt.py` | Reproduces the VectorBT parameter-grid trap (§3.1) |
| `engine/store/stream.py` | Constant-memory ingest for tick archives of any size |
| `engine/library/` | SQLite + content-addressed artifacts — everything survives a restart (§16) |
| `engine/service.py` | One implementation of every operation; MCP and HTTP are adapters over it |
| `app/api/` | FastAPI backend + the agent WebSocket |
| `app/ui/` | Dark UI: strategies, chart, data, history, assistant sidebar |
| `scripts/run.sh` | Start the app |
| `scripts/drive_ui.py` | Drives the UI in Chromium and fails on any console error |
| `tests/` | 244 passing |
| `tools/pinets_oracle/` | Dev-only fixture generator (AGPL, never shipped) |
| `bench/` | Reproduces every `[measured]` number in the plan |

## The app

```bash
./scripts/run.sh        # then open http://127.0.0.1:8000
```

Four pages and a sidebar, all dark:

- **Strategies** — the version tree, the signal block, and one panel that runs
  everything: backtest, grid sweep, walk-forward validation, and the control
  test. Results land inline and in the library.
- **Chart** — candles with the run's **own** trade list drawn on them, the
  equity curve panned in step below, and a trade table that scrolls the chart
  to any trade. The markers are the engine's trades, never re-derived in the
  browser, so the picture and the number cannot disagree.
- **Data** — load a tick or bar CSV, read its quality report, see which
  timeframes it can serve.
- **History** — every run ever made, reopenable, including the failed ones.
- **Assistant** — Claude Code over a WebSocket, driving the same MCP tools.
  No API key: it runs on your own subscription.

Large files stream. A 295 MB tick CSV loads in ~24 s at flat memory; bars are
cached at M1, so every coarser timeframe afterwards is free.

A worked example — hypothesis, rejection, revision, control test and
walk-forward, with the numbers and the bugs it exposed — is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

### The control test

The cheapest way to tell a real edge from a bug: re-run the strategy on
**shuffled returns**. Same distribution, same costs, same bar count — the order
destroyed. An edge that survives that is look-ahead or a sizing artifact, not an
edge. It is a button in the UI and an MCP tool (`control_test`); run it before
believing any result.

## Quick start — run it locally

```bash
git clone https://github.com/bohssin/QUANTOR && cd QUANTOR
./scripts/setup.sh
```

That creates a venv, installs the engine and MCP dependencies, writes a
`.mcp.json` pointing at **that** interpreter with absolute paths, runs the
tests, and finishes with a health check.

Then sign in to LuxAlgo (free account — the token is stored **per machine**, so
it has to be the machine running the server):

```bash
npx -y @luxalgo/mcp login
```

And start a session:

```bash
claude --mcp-config .mcp.json \
  --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"
```

Ask it to do the loop:

> Load `ticks.csv` as 'xau' at M15, write an EMA-crossover strategy with an ATR
> stop, backtest it, then `validate_run` it and tell me honestly whether it
> generalized.

### When something breaks

```bash
.venv/bin/python quantor_mcp/doctor.py
```

Every check ends in a command you can paste:

```
  [PASS] Python >= 3.11            3.12.3 at /home/you/QUANTOR/.venv/bin/python
  [PASS] engine + MCP packages     all present
  [PASS] numba JIT compiles        ema_fast returned finite values
  [PASS] MCP server handshake      initialize answered over stdio
  [PASS] end-to-end backtest       103 trades, ledger reconciles=True
  [PASS] .mcp.json                 quantor-engine configured
  [WARN] LuxAlgo MCP (optional)    npx found; signed in: no
```

The doctor exists for one failure in particular. **If `.mcp.json` names a Python
without `mcp` installed, the server exits instantly and Claude Code reports only
`CONNECTION_CLOSED`** — no traceback, nothing naming the cause. The doctor names
it. Always point `command` at the venv's python, which `setup.sh` does for you.

## What's here

## Licence note

The shipped system has no copyleft: Vela is Apache-2.0, the Python stack is BSD/MIT/Apache,
everything else MIT. PineTS (AGPL-3.0-only) is a **development-only** test oracle under
`tools/pinets_oracle/` and is never imported by `engine/` or distributed. See plan §3.
