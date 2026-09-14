# QUANTOR — notes for whoever works on this next

A local backtesting platform for one person's own research. Python engine
(numpy + numba), MCP server, FastAPI + a dark browser UI. No API keys anywhere:
the assistant is the Claude Code CLI the owner is already signed into.

Read `trading-agent-plan.md` first — it is the design document and every module
docstring cites its sections. `docs/FINDINGS.md` is the record of the first real
research pass, including what turned out to be wrong.

## Running it

```bash
./scripts/setup.sh            # Linux/macOS      setup.cmd     on Windows
./scripts/run.sh              #                  run.cmd       on Windows
.venv/bin/python -m pytest tests/ -q
.venv/bin/python quantor_mcp/doctor.py      # names the exact fix for each failure
.venv/bin/python scripts/drive_ui.py --base http://127.0.0.1:2026   # real browser
```

The app serves **http://localhost:2026** (or `quantor:2026` with a hosts entry).
`scripts/serverctl.sh start|stop|restart` manages it via a pidfile — do **not**
`pkill -f uvicorn`, the pattern matches the calling shell's own command line and
kills the caller.

## Shape

**`engine/service.py` is the only implementation of anything.** `quantor_mcp/server.py`
is an MCP adapter over it and `app/api/main.py` is an HTTP adapter over it.
`tests/integration/test_cross_surface.py` runs the same strategy through both and
compares every field, because two code paths computing "the same" number diverge
eventually, and the way it shows up is a strategy measured at Sharpe 1.4
displaying 0.9 with no error anywhere.

| | |
|---|---|
| `engine/store/` | CSV ingest, tick→bar, constant-memory streaming, S1..W1 |
| `engine/backtest/` | Bar-mode fills + metrics. `subbars` settle intrabar order |
| `engine/optimize/` | Grid/random search, plateau (neighbourhood) robustness |
| `engine/validate/` | Fold geometry, walk-forward |
| `engine/signal/` | AST validator for generated signal blocks (§9) |
| `engine/library/` | SQLite + content-addressed artifacts. Survives restart (§16) |
| `app/ui/` | Plain ES modules, no build step, one dark theme |

## Rules that came from something breaking

Each of these is a bug that shipped once. They are not style preferences.

- **Never default a run's timeframe to the cache base.** A source caches bars at
  M1 (or S1) and is *loaded as* M15. Defaulting a backtest to the base reports an
  M1 Sharpe for an M15 strategy — the same trades scaled by √15, silently, and in
  the flattering direction. `default_timeframe` is a separate column for this.
- **No silent fallbacks on lookup tables.** `BARS_PER_YEAR.get(tf, 6_192.0)` made
  every timeframe outside its eight entries annualize as H1. Derive, and raise on
  what you do not know.
- **A ratio is only defined where its denominator has a sign.** `plateau.robustness`
  = mean(neighbours)/peak inverted below zero: a peak of −0.91 with neighbours at
  −0.96 scored 1.06 and read as the flattest possible plateau.
- **Run the control test before believing a number.** `control_test` re-runs on
  shuffled returns: same distribution, order destroyed. An edge that survives is
  look-ahead or a sizing artifact. It is twenty lines and it is the cheapest
  check in the system.
- **Report expectancy in R, not net profit.** Sizing is fixed-fractional, so
  profit compounds and its headline mostly measures how many trades there were.
- **Surface the error text.** The MCP SDK swallows exceptions into
  `UnexpectedToolError`; an agent that sees an identical opaque failure for every
  input correctly concludes the tool is broken and stops. `surfacing_errors`
  returns the type and message. Same reasoning for the HTTP 400s.
- **A period is a whole number of bars.** A float reached numba as a forty-line
  `TypingError`. Validated at the indicator boundary now.

## Packaging traps (all of these bit)

- **Anchor `.gitignore` directory patterns.** `library/` matches that name at any
  depth and silently excluded `engine/library/` from every clone. Use `/library/`.
  `tests/test_repo_is_complete.py` asks git directly and fails on untracked source.
- **`.ps1` files with non-ASCII need a UTF-8 BOM.** PowerShell 5.1 reads a
  BOM-less script as ANSI, mangling it at parse time.
- **`cmd.exe` does not execute `.ps1`.** Hence `setup.cmd` / `run.cmd` at the root.
- **Large artifacts go to Parquet, not gzip.** gzip wrote 429 MB of bar arrays in
  109 s; Parquet+zstd does it in 8.2 s for the same size.
- **Long loads must be jobs.** `POST /api/data` is synchronous and a browser will
  not wait fifteen minutes. The UI uses `POST /api/data/start` and polls.

## Conventions

- Module docstrings say **why**, cite the plan section, and name the failure the
  module prevents. Comments explain decisions, never restate the code.
- Test names are sentences: `test_a_bar_split_across_chunks_merges_rather_than_duplicating`.
  Assert the property, not the implementation.
- Measured numbers are marked `[measured]` and are reproducible from `bench/`.
- Signal blocks (§9): `signal(bars, p)` returning `long_entry`, `short_entry`,
  `stop_distance`, `target_distance`. No imports, no order placement, no forward
  indexing. The validator rejects `bars.close[1:]` outright — build the
  full-length boolean, then slice the **local**.

## Where this is

Built and tested: engine, library, MCP (14 tools), HTTP API, dark UI, chart,
agent sidebar, streaming ingest, S1 intrabar fills. 317 tests.

**Not done.** `real_ticks` fills do not exist — every number is bar-mode with S1
as the finest resolution. No deflated Sharpe yet, though the cumulative
comparison count it needs is recorded per family.

**Never done on real data.** Everything measured so far is synthetic
(`scripts/make_sample_data.py`). The owner's 11 GB XAUUSD tick archive at
`C:\Users\HP\Documents\Téléchargements MEGA\xau.csv` (GMT+3) has never been
read. Loading it is the next task, and every figure about it — S1 bar count,
load time, memory — is currently an extrapolation from a 300 MB file. Treat them
as guesses until measured.
