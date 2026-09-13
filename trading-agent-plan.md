# Local AI Trading Agent — Design Plan (rev 4)

**Status:** design handoff. Nothing built yet.

**Shape:** a local application. Chart on the left, **an AI agent chat on the right** — that
conversation is the primary interface (§15). You brainstorm strategies with it, it writes
them, applies them to the chart, backtests, optimizes and validates them, and you keep talking
to refine them. Across any instrument.

**A platform, not a script runner.** Strategies, versions, runs, optimization tables, verdicts
and the conversations that produced them are all catalogued and browsable later (§16). Nothing
generated is thrown away.

**Data in:** the owner's own CSVs — **ticks or bars**, any instrument, any precision, each
with its own GMT offset. The file's resolution decides what it can be used for (§4.6), and
its precision is read out of the file rather than declared (§4.2).

**Reference platform: MetaTrader 5.** One reference, everywhere. The data model is MT5's
(tick history → bars at a chosen timeframe), the fidelity dial is MT5's (modeling modes), the
UI is MT5's (Strategy Tester layout), and the live execution path is the owner's existing MQL5
EA. Nothing in this system targets TradingView.

**Engine:** Python — numpy + numba. No Pine Script transpiler, no 1-second intermediate bars.

**Agent runtime:** Claude Code, on the owner's own subscription. **No API keys** — everything
beyond files goes through MCP: the LuxAlgo Library, Edge Stats, Prop Firm Sim, and QUANTOR's
own engine. One emitted artifact per strategy: a **Python** signal block. See §14.

---

## 0. To the coding agent reading this

**This is a starting point, not a specification. You are expected to improve it.**

What is expected of you:

- **Challenge the architecture.** If §7's split is wrong, or there is a cleaner way to decouple
  signal from execution, say so with reasoning. Do not implement something you think is a
  mistake.
- **Brainstorm before building.** Discuss with the owner first. Propose alternatives. Surface
  trade-offs the plan misses.
- **Research further.** Read the actual repos, the actual source, the actual issue trackers.
  Re-verify anything stated as fact — facts here were gathered at a point in time.
- **Extend it.** Whole areas are sketched rather than designed: the UI, the optimizer
  internals, live-forward testing, multi-strategy portfolios, regime detection, the exemplar
  corpus. Add what is missing.
- **Question the sequencing.** §20's build order is a guess at dependency structure. If a
  different order gets to a working loop faster, propose it.

What is *not* up for renegotiation, because these are correctness properties rather than
preferences:

- **§5's instrument specs drive all sizing and P&L.** No hardcoded contract sizes, ever.
- **§4.3's server-timezone rule.** Bar boundaries are defined in broker server time. Get this
  wrong and every session- and daily-level signal is wrong, silently.
- **§4.4's tick ordering rule.** Identical-millisecond ticks must have a deterministic order,
  or results are not reproducible.
- **§8's golden-fixture rule.** No indicator ships without a frozen numerical fixture.
- **§10.1's engine invariants.** The fill engine is verified by property tests, not by
  inspection. These carry more weight now that MT5 reconciliation is not a build gate (§10.2).
- **§14's no-API-keys rule.** The agent is Claude Code on the owner's own subscription,
  reaching everything else through MCP. No `ANTHROPIC_API_KEY`, no raw API client. This
  constrains reproducibility in a specific way (§14.6) — accept the constraint, do not route
  around it.
- **§16's keep-everything rule.** Strategies, versions, runs and rejected candidates are
  catalogued, not discarded. §9's failure taxonomy and §13's holdout tracking are queries
  over that history; delete it and neither can exist.
- **§13's holdout-touched-once and comparison-count logging.**
- **§17's policy file: no threshold anywhere else in the codebase as a literal.**

Everything else is a proposal. Argue with it.

---

## 1. How to read this

- **§2** is what changed from rev 3 and why. Read it if you have seen the earlier plan;
  skip it if you have not.
- **§3–4** are the stack and the data layer. §4 is the foundation — nothing above it matters
  until the data is trustworthy.
- **§5–13** are design decisions.
- **§14–15** are the agent runtime and the application.
- **§17** is the policy file.
- **§18–20** are repo layout, probes, build order and open questions.
- **§22** is the execution boundary.

Measured numbers appear inline as `[measured]`. They were produced on a 4-core / 16 GB Linux
box against a synthetic 10M-tick dataset and are recorded so you do not re-litigate settled
questions. **Reproduce them yourself** — `python3 bench/bench_indicators.py` and
`python3 bench/bench_ticks.py`. Re-measure on the owner's real data before trusting them for
capacity planning: synthetic ticks are uniformly distributed and real ones cluster hard around
news.

---

## 2. What changed from rev 3, and why

Rev 3 built signal computation on **PineTS**, a Pine Script → JS transpiler, and aggregated
ticks into **1-second bars** as a universal execution resolution. Both are gone.

### Why PineTS is gone

It was only ever doing one narrow job — computing indicator series and arm conditions on
signal-timeframe bars. Measured against that job:

| | PineTS 0.9.33 | Python (numpy/numba) |
|---|---|---|
| Signal pass, 100k bars | 905 ms | 0.94 ms `[measured]` |
| Signal pass, 370k bars (1 yr @ M1) | ~3,300 ms | 6.75 ms `[measured]` |
| Licence | AGPL-3.0-only | BSD / MIT throughout |
| Numerical reference | TradingView | **MT5** — the platform we actually trade on |

Three findings settled it:

1. **Pine's `ta.*` semantics are reproducible in ~60 lines of numpy.** `ta.ema`, `ta.rma`,
   `ta.rsi` and `ta.atr` were reimplemented and compared against PineTS on identical data:
   max absolute difference **5.0e-11**, which is PineTS's own JSON output precision, with
   warmup-NaN counts matching exactly. `[measured]` The claim that leaving Pine costs you
   indicator fidelity is false for the primitives.
2. **Pine could not express the intent contract.** Pine emits parallel numeric series, not
   event records. Rev 3's intent schema (heterogeneous fields, string enums, nested trailing
   config) would have had to be encoded as integer-coded plot channels with a hard ceiling of
   one intent per bar per channel group. In Python an intent is a dataclass. Rev 3's riskiest
   probe simply stops existing.
3. **PineTS silently returns stale results when reused.** `Indicator.prepare()` snapshots
   resolved inputs on first call, so mutating `ind.input[...]` between runs is validated,
   accepted, and then **ignored** — every subsequent evaluation returns the first one's
   numbers. `[measured]` An optimizer written the obvious way would have produced a complete,
   plausible, entirely wrong results table. This is the class of bug that ends research
   projects.

**PineTS is not deleted, it is demoted.** It stays as a **dev-only test oracle** for
generating golden indicator fixtures (§8.2). AGPL is unencumbered for a dependency that is
never distributed, and it gives us an external reference implementation to check against
without any runtime coupling.

### Why the 1-second layer is gone

Rev 3 aggregated ticks → 1s bars → M1 → M5 … and treated 1s as a fixed execution resolution.
That was a compute-driven compromise, and it was buying inaccuracy:

- **1s bars reintroduce the exact ambiguity they were meant to solve.** When stop and target
  both sit inside one 1-second bar's range, the engine cannot know which was touched first.
  Rev 3 responded with a whole subsystem — ambiguity instrumentation, a tolerance
  threshold in policy, and a spot-check workflow.
- **It was never necessary.** The owner has ticks. A numba fill loop resolves fills against
  **real ticks** at **360M ticks/s** `[measured]` — a full year of tick-resolution backtest in
  about **0.2 s**.

So: ticks in, bars at the chosen timeframe for signals, fills against the ticks themselves.
The 1s layer, the rollup pyramid, the ambiguity metric, its policy threshold and its
spot-check workflow are all **deleted**. There is no ambiguity to instrument, because tick
order is known.

### Everything rev 3 got right, kept

The harness / signal-block split (§9), the instrument registry (§5), duration-vs-bar-count
discipline (§4.5), research integrity mode (§13), the policy file (§17), output-is-parameters
(§22), and §0's "argue with this" posture. Those were the good ideas and they survive intact.

### Two numbers this plan got wrong, and fixed

Recorded because the *method* of the error matters more than the error.

- **Tick volume, out by 11×.** A per-tick rate extrapolated from three consecutive sample rows
  gave 414M ticks/year and a declaration that chunking was mandatory. The owner's actual file
  size — ~1.5 GB/year — says ~38M ticks, which fits in memory. Tick arrival is violently
  non-uniform; measure spans, never samples (§4.1).
- **Instrument precision, wrongly declared.** §5 originally hardcoded XAUUSD at 2 decimals,
  then 3. Both were guesses at something the CSV already states. Precision is now read from the
  file (§4.2) and the spec owns only what the file cannot tell you.

### Consequences worth naming

- **All AGPL dependencies are gone.** This matters more than it looks: §15 serves the UI to a
  browser, which is the shape that flirts with AGPL's network-service trigger.
- **Three rev-3 open questions are resolved.** Runtime split → all Python. App shell → local
  web. Harness composition → a Python function, no textual splice step.
- **Two of four probes are dead.** The bar-count ceiling and intent-extraction probes are
  obsolete; throughput is measured above.
- **What did *not* get smaller:** the store, the fill engine's correctness, the instrument
  registry, the validation suite, research integrity. That is roughly 80% of the work and this
  revision does not touch a line of it. Rev 4 is a better *shape*, not a much smaller project.

---

## 3. The stack

### Engine (Python)

| Component | Choice | Licence | Role |
|---|---|---|---|
| Arrays / math | **numpy** | BSD | all series computation |
| Hot loops | **numba** (`njit`) | BSD | bar construction, sweep fill engine |
| Exploration / stats | **VectorBT** (+ `plotly<6`) | Apache-2.0 + CC | metrics, plotting, cross-check (§3.1) |
| Columnar store | **pyarrow** + Parquet | Apache-2.0 | tick and bar persistence |
| Tabular / joins | **pandas** or **polars** | BSD / MIT | ingest, reporting (pick one, §21) |
| Optimization | **Optuna** 5.x | MIT | TPE, NSGA-II, pruning, trial storage |
| Metrics | **QuantStats** / **empyrical** | Apache-2.0 | Sharpe, Sortino, Calmar, drawdown |
| Property tests | **hypothesis** | MIT | engine invariants (§10.1) |
| API | **FastAPI** + uvicorn | MIT | serves the UI and run control |

### 3.1 Build versus buy

The owner uses this personally and does not sell it, so copyleft and
Commons-Clause restrictions are not constraints. Margin, swap and FX lot
modelling are explicitly not priorities — **the priority is finding profitable
strategies**, which means parameter search and honest validation.

That settles the earlier NautilusTrader decision: it was adopted for tick-level
margin/swap/FX execution, and that requirement is withdrawn. **Dropped.**

**Chosen from the comparison article: VectorBT** over Backtesting.py.
Backtesting.py is a per-bar Python loop, single-asset, with a weaker optimizer;
VectorBT is numba-vectorized, built for exactly the parameter sweeps this
project runs, and ships 28 portfolio metrics and a plotting suite. Verified
here on 40,000 bars `[measured]`: `sl_stop` / `sl_trail` / `tp_stop`, `fees`,
`slippage`, and a full `stats()` table including profit factor, expectancy,
Sharpe, Sortino, Calmar and win rate.

**Where it fits, and where it does not.** Two measurements decide this.

- **Speed: 89.7 ms per combination, against our numba engine's 2.2 ms** — 41x
  slower, on a *simpler* strategy (percentage stops rather than ATR-based). For
  the sweep hot path the engine we already have is faster and is covered by 138
  tests.
- **A silent correctness trap.** Asking VectorBT for a 5x5 indicator grid
  returns **5 columns, not 25** — `param_product=True` applies within one
  indicator's parameters, not across two separately-run indicators, so you get
  the diagonal `(10,40), (15,50), (20,60), (25,70), (30,80)`. No warning. The
  results table looks correct, every row correctly labelled, and it silently
  contains a fraction of the configurations you asked for. Reproduce with
  `bench/probe_vectorbt.py`. This is the same class of failure as rev 3's
  PineTS input-caching bug (§2), and it lands on the optimizer again.

So VectorBT is adopted as the **exploration and analysis layer**, not the sweep
engine:

| Use | Why |
|---|---|
| Its `stats()` suite | Replaces hand-rolled `metrics.py`, which was commodity work |
| Quick exploration of an idea | Better ergonomics than wiring a full run |
| Plotting and portfolio analysis | Genuinely good, and we would not write it |
| **Independent cross-check (§10.4)** | Two implementations should agree on a simple strategy. Divergence means one has a bug — the external reference we otherwise lack |

Sweeps, walk-forward and anything whose numbers get reported stay on our engine.
It also needs `plotly<6` pinned: current VectorBT fails to import against
plotly 6 (`scattermapbox` was removed).

**Still built, and why nothing substitutes.**

- **The store (§4)** — nothing reads the owner's CSV with per-import GMT offset,
  precision detected from the file, and the source-resolution rule.
- **The indicator layer (§8)** — and this matters *more* now, not less. The agent
  emits a Pine script and a Python script for the same strategy (§14.3); they
  agree only if our Python indicators are Pine-exact. The golden fixtures are
  what make the two outputs comparable at all.
- **The bar-mode engine (§7)** — 41x faster than VectorBT, and tested.
- **Fold geometry (§12)** — entry-attribution, purge-from-trade-duration and the
  nested search are not what general CV libraries do.

**Also adopted:** Optuna (MIT) behind the existing `ParamSpec` for TPE and
NSGA-II.

**Rejected, with reasons, since the question recurs.**

| Library | Why not |
|---|---|
| **Backtesting.py** | Per-bar Python loop, single-asset, weaker optimizer. AGPL is fine here, speed is not |
| **Backtrader** | GPLv3 and **last released 2023-04-19**. The article rates it the realism pick without noting it is unmaintained |
| **NautilusTrader** | Excellent for tick-level margin/swap/FX — a requirement now withdrawn. Revisit only if live-execution fidelity becomes the goal |
| **Zipline Reloaded** | Daily US equities behind a bundle pipeline. Wrong market and resolution |
| **bt** | Portfolio rebalancing across many assets. Wrong shape |
| **QuantConnect / LEAN** | C# core, cloud-oriented, heavy lock-in for a local-first tool |
| **pysystemtrade** | Opinionated around one futures methodology — you trade its way or fight it |
| **Fastquant** | A wrapper for quick looks, not an engine |

### External (kept from rev 3)

| Project | Licence | Role |
|---|---|---|
| **Vela** | Apache-2.0 | The chart. WebGL2/Canvas2D renderer, plugin SDK. Fed plain series over the API — the Pine plugin is not used |
| **LuxAlgo MCP** | MIT | Concept reference. 850+ concepts, Pine source on demand. Keyless. **Now the only Pine touchpoint** |
| **Edge Stats** | MIT | Conditional-frequency engine over M1 bars. N + Wilson CI on every answer. 10 MCP tools |
| **Prop Firm Sim** | MIT | Block-bootstrap Monte Carlo over an R-multiple series. MCP package |
| **Trade Journal** | MIT | Live-vs-backtest reconciliation, SQLite, self-hosted |
| **Broker SDK** | MIT | Read-only account/trade sync, tolerant statement-CSV parser |

Edge Stats and Prop Firm Sim are **MCP servers**. They are language-agnostic and unaffected by
the engine being Python — the agent talks to them over MCP, with no in-process coupling.

### Dev-only

| Project | Licence | Role |
|---|---|---|
| **PineTS** | AGPL-3.0-only | **Test oracle only.** Generates golden indicator fixtures (§8.2). Never imported by the engine, never distributed |

### The Library's role

**Reference, not strategy source.** The agent uses it to (a) understand a named concept and
(b) read real working Pine as *semantic* grounding, then implements in Python. The owner will
often ask for strategies not in the Library at all. **The system must never require a Library
match to proceed.**

### Licensing

Vela is Apache-2.0. The Python stack is BSD/MIT/Apache. Everything else MIT. There is no
copyleft in the shipped system, so distribution and network-service deployment are both
unencumbered.

---

## 4. Data layer — tick CSV is the foundation

### 4.1 Pipeline

```
tick CSV  →  normalized tick store (Parquet)  →  bars at the run's timeframe
                     │                                      │
                     └──────────── fills (§7) ──────────────┘
```

That is the whole pipeline. No 1s layer, no rollup pyramid. Bars are built on demand from
ticks at whatever timeframe the run asks for; the tick store is the single source of truth.

**Measured against the owner's actual format** — `timestamp,bidPrice,askPrice` with
millisecond datetime strings — at 5M rows: `[measured]`

| Stage | Time | Rate |
|---|---|---|
| **`pyarrow.csv`** parse | **0.21 s** | **23.4M rows/s** |
| `pandas.read_csv` + `to_datetime` | 5.5–8.4 s | 0.6–0.9M rows/s |
| Parquet write (zstd) | ~1.5 s | 320 MB → 58 MB |
| Parquet read | ~1.3 s | the steady-state path |
| Ticks → M1 bars (numba) | 92 ms | 64,915 bars from 10M ticks |

**Use `pyarrow.csv`, not pandas.** It is 39× faster here, and at the owner's observed tick
density that is the difference between ~18 s and ~12 minutes to parse one year.

**Two traps found while measuring, both silent:**

1. **pandas 3.0 infers datetime resolution from the data**, so
   `to_datetime(col).astype('int64')` yields *seconds* for some inputs and *milliseconds* for
   others. It produced timestamps off by a factor of 1000 with no error raised. Pin the unit
   explicitly (`pa.timestamp('ms')`) and cross-check the first row against a stdlib parse. The
   ingest does both.
2. `to_datetime(..., format='%Y-%m-%d %H:%M:%S.%f')` is **3× slower** than letting pandas
   infer, because an explicit format forces the slow `strptime` path and bypasses the ISO8601
   fast path. The obvious "optimization" is a pessimization.

**Volume, from the owner's actual files: ~1.5 GB of tick CSV per year.** At 42 bytes per row
that is **~38M ticks/year**, averaging ~1.9 ticks/second.

An earlier revision of this section said ~414M ticks/year and declared chunking mandatory.
That was wrong by 11×: it extrapolated a per-tick rate from three consecutive sample rows that
happened to fall inside an active burst. The file size is the honest measure, and it says
something much more comfortable:

| Storage | bytes/tick | 1 year (≈38M ticks) |
|---|---|---|
| int64 ms + 2× float64 | 24 | 0.92 GB |
| int64 ms + 2× scaled int32 | 16 | **0.61 GB** |

**A full year fits in memory.** Partition by `symbol/year/month` anyway — it keeps loads
scoped to the run's span and makes re-ingest incremental — but streaming across partitions
with carried position state is an optimization, not a requirement. Do not design for it up
front.

The lesson generalizes: extrapolating rates from a handful of rows is unreliable in market
data, because tick arrival is violently non-uniform. Measure spans, not samples.

**Prices are stored as integers scaled by tick size** — 1905.366 at `tick_size=0.001` becomes
`1905366`. Cuts memory by a third, round-trips exactly `[measured]`, and makes stop/target
comparisons *exact integer* comparisons. That last point is the real reason: price-level
comparison is the one place in the engine where a float epsilon silently changes which trade
happened. int32 covers prices to 2,147,483 at 3 decimals; fall back to int64 beyond that.

### 4.2 Ingest

The owner's CSVs are the universal data path, and tick CSV formats vary by vendor. Ingest is
**tolerant and explicit**: a per-source YAML mapping, never format sniffing.

**Ticks or bars — the store takes either.** Not everyone has tick history, and for many
strategies M1 bars are plenty. Both shapes normalize to the same `MarketData`:

```
timestamp,bidPrice,askPrice                  <- ticks
2021-01-04 01:00:00.413,1904.998,1905.366

timestamp,open,high,low,close,volume         <- bars (M1, M5, H1, ...)
2021-01-04 01:00:00,1904.998,1905.4,1904.2,1905.1,318
```

The source's own resolution then decides which modeling modes it can support (§6) and which
strategy timeframes it can serve (§4.6).

**Precision is read out of the file, never declared.** Digits differ per asset and per feed —
XAUUSD at 3 decimals here, EURUSD at 5, an index at 1 — and the CSV already states its own
precision by how it is written. A column holding both `1904.998` and `1905.12` is a 3-decimal
column whose trailing zero was trimmed, so `tick_size = 0.001`. Requiring the owner to declare
this per instrument was one more thing to get wrong for no benefit.

**Detection reports, it does not guess silently.** Column names are matched against a small
alias set (`bidPrice`/`bid`/`b`, `close`/`c`, `Date_Time`/`timestamp`, …), the source kind
follows from which family is present, and whatever was concluded is exposed on
`MarketData.detected` and printed at import. A file carrying *both* OHLC and bid/ask columns
raises rather than picking a side; so does one where no price family is recognizable. Any
field can be pinned explicitly on `SourceSpec`, which always wins.

**GMT offset is set per import.** Vendors differ and files rarely say, so every import carries
its own `utc_offset_hours` — cross-checked against §4.3's weekend-gap inference, never
inferred silently. Two files of the same symbol on different clocks normalize correctly
because each declares its own.

Also support at minimum: epoch ms/s and MT5's exported tick format
(`<DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>`). Implemented in
`engine/store/ingest.py`; `tests/store/` covers ticks, bars, alias matching, precision
detection and the refusals.

Records the file's SHA-256 and row count in the store manifest. A re-ingest that produces a
different hash is a new store version, stamped into every run that used it.

### 4.3 Server timezone — a correctness property

**Bar boundaries are defined in broker server time, not UTC.** MT5 builds bars on the server's
clock, which for most retail brokers is EET (UTC+2, UTC+3 in summer) — so the D1 bar breaks at
00:00 EET, not 00:00 UTC, and the difference moves with DST.

Get this wrong and every daily-level signal (previous day's high/low, daily open, session
ranges) is computed on the wrong window, silently, producing a plausible backtest that cannot
be reproduced on the owner's platform.

Rules:

- The instrument spec (§5) declares the broker's server timezone **including its DST rule**.
- Bar alignment is computed in server time, then stored as UTC ms.
- A run artifact records the server timezone used. Two runs with different server timezones are
  not comparable and the report must say so.
- Broker server timezones differ between brokers for the same symbol. Treat that as real.

**Verify the declared offset empirically.** Tick files rarely state their timezone, and the
owner's sample opens at `01:00:00.413` on Monday 2021-01-04 — which is consistent with UTC,
with a broker clock, or with an arbitrary file start. Three rows cannot settle it.

The data can. FX and metals close Friday evening and reopen Sunday evening, so the largest
recurring gaps in a tick stream are weekends, and the reopen hour reveals the clock the file
is keyed to. `infer_utc_offset_hours()` reports it.

Measure at the gap's **end**, not its start: the reopen is a hard boundary (the feed switches
on and the first tick lands within milliseconds) whereas the Friday close is soft (liquidity
thins and the last tick may precede the official close by an arbitrary margin, biasing the
estimate early by exactly that margin).

Treat the result as a **cross-check on a declared value, never as the authority.** Confirm
with the owner, then pin it in the source spec.

### 4.4 Bar construction — MT5 semantics

- **OHLC from bid.** Ask is reconstructed as bid + spread. This is MT5's default and it is what
  the owner's MQL5 results were produced with.
- **Spread carried as first-class columns** per bar: mean and max, in price units. Needed by
  the cost model and by the `m1_ohlc` modeling mode.
- **Volume is tick count** when building from ticks, matching MT5.
- **A bar with no ticks does not exist.** Never forward-fill. MT5 skips empty bars; so do we. A
  missing bar means the market could not be traded there.
- **Tick ordering is deterministic.** Real tick data contains multiple ticks in the same
  millisecond, and their order changes fills. Rule: sort by timestamp with a **stable** sort so
  ties preserve source-file order, and record the file hash. Without this the engine is not
  reproducible.
- Sessions, rollover windows, Sunday opens and holidays are tagged from the instrument spec.

**Store validation before trusting anything:** tick-count histogram per hour (feed outages),
spread distribution per session, bar count vs. expected, gap inventory, and a monotonic-
timestamp check with a report of same-ms clusters.

### 4.5 Durations vs. bar counts

Rev 3 made this unconditional because it swept signal timeframe. MT5 users think in bars on a
fixed timeframe, which is simpler and more familiar. So the rule becomes **conditional**, which
preserves the correctness property exactly where it bites:

- **Default (MT5-style):** timeframe is fixed for the run; indicator lengths are in bars.
  `ta.ema(close, 20)` means 20 bars. Familiar, and correct because the timeframe does not move.
- **If `signal_timeframe` is swept, lengths MUST be declared as durations.** A strategy written
  at M15 and re-run at H1 must mean the same thing; with bar counts it does not. The harness
  **rejects a timeframe sweep on a strategy with bar-count lengths** rather than silently
  producing nonsense.
- Level-based signals (session highs/lows, previous daily H/L, order blocks, FVGs, swing
  structure) are maxima over a clock window and are resolution-invariant either way.
- Note the residual: with empty bars skipped (§4.4), a bar-count indicator spans a different
  clock duration in a thin session than a liquid one. The duration→bars conversion uses nominal
  timeframe, so it is approximate on gappy data. Accept this explicitly; do not pretend
  otherwise.

### 4.6 Source resolution vs strategy timeframe

**A source can be aggregated up to a coarser timeframe. It can never be split down to a finer
one.** Detail that was never recorded cannot be recovered, and inventing it is how a backtest
starts reporting fills that could not have happened.

| Source | Serves | Best available modeling mode (§6) |
|---|---|---|
| Ticks | every timeframe | `real_ticks` |
| M1 bars | M1 and coarser | `m1_ohlc` |
| M5 bars | M5 and coarser | `open_prices` / bar-close |
| H1 bars | H1 and coarser | `open_prices` / bar-close |

Equal resolution is allowed — an M5 file drives an M5 strategy — but only at bar-close
fidelity: intrabar order is unknown, so stop-versus-target within the signal bar cannot be
resolved.

**So ambiguity instrumentation returns — for bar sources only.** Rev 3's ambiguity metric was
deleted because real ticks make the question unanswerable-by-construction rather than
unanswered (§2). On *bar* data the question is back: when a bar's range contains both the stop
and the target, the engine takes the stop (the conservative reading) and **counts it**.
`BacktestResult.ambiguity_rate()` reports the share of trades affected, and a high rate means
the bar resolution is too coarse for the strategy's stop distance. The report states the rate
rather than presenting the pessimistic number as fact.

A second rule falls out of the same reasoning: **a position is never closed by its own entry
bar's high/low.** The intrabar path is unknown, so such a fill depends on an ordering the data
never recorded.

The check runs at load (`assert_serves_timeframe`) and refuses with a message naming both
resolutions and the two ways out: supply finer data, or run the strategy at the source's
resolution or coarser. Never a warning — a run on impossible data is not a run.

---

## 5. Instrument specification registry

**What makes "compatible with everything" true.** One spec per instrument, loaded by the
execution engine. Nothing about sizing, P&L or order validity may be hardcoded.

```yaml
symbol: XAUUSD
broker: <str>               # specs differ between brokers for the same symbol
asset_class: metal          # forex | metal | index_cfd | future | crypto | equity
base_currency: XAU
quote_currency: USD
contract_size: 100          # units per 1.00 lot
lot: {min: 0.01, step: 0.01, max: 100}
# NOTE: tick_size / digits are NOT declared here — they are read from the
# imported CSV (§4.2), because precision varies by asset and by feed. What
# the spec owns is what the file cannot tell you: contract size, costs,
# margin, sessions.
tick_value: 0.10            # quote currency, per 1.00 lot, per tick_size
pip_definition: 0.1         # explicit — "pip" is ambiguous on metals

# MT5 order-validity constraints (rev 3 missed these)
stops_level: 0              # min SL/TP distance in points; broker rejects closer
freeze_level: 0             # distance within which orders cannot be modified

margin: {mode: leverage, leverage: 500}   # leverage | fixed_per_lot | percent_notional
margin_call_pct: 100
stop_out_pct: 50

costs:
  commission: {mode: per_lot_per_side, value: <...>}
  swap_long: <...>
  swap_short: <...>
  swap_3day: wednesday
  swap_mode: <points|percent_annual|money>

server:
  timezone: "EET"           # §4.3 — with DST rule
  dst_rule: <str>
  rollover_time: "00:00"    # server time; swap applied here
sessions:
  schedule: [...]           # incl. rollover window, Sunday open
  holidays: [...]
```

Futures minis/micros are separate specs with their own multipliers — ES (50) vs MES (5),
NQ (20) vs MNQ (2), GC (100) vs MGC (10). Seed from broker contract specs.

**`stops_level` matters.** Backtests that ignore it accept stops the broker would reject, which
inflates results for tight-stop strategies. The engine enforces it and reports rejections.

**Currency conversion:** when quote currency ≠ account currency, P&L needs an FX rate series.
The store must carry the pair, and the spec names it.

**Specs are versioned and stamped into every run.** Costs and swap are values you will tune;
if tuning touches the holdout window you have leaked through the spec rather than through the
parameters (§13).

---

## 6. Modeling modes — the MT5 fidelity dial

MT5's Strategy Tester exposes an explicit accuracy/speed trade-off instead of a hidden
assumption. Adopt it directly, with MT5's vocabulary, because the owner already reasons in it.

| Mode | Engine | Fills resolved against | Cost (1 yr) | Use |
|---|---|---|---|---|
| **`real_ticks`** | ours (numba) | real tick stream | ~0.1 s `[measured]` | **System of record.** Every reported result |
| **`m1_ohlc`** | ours (numba) | M1 OHLC, assumed intrabar path | ~0.1 s | Spans with missing tick data |
| **`open_prices`** | ours (numba) | bar open | **2.2 ms** `[measured]` | **Optimizer sweeps** |

All three run on our engine. Sweep in `open_prices`; report in `real_ticks`.

**Availability follows the source (§4.6), not preference.** `real_ticks` requires a tick
source; `m1_ohlc` requires M1 or finer; `open_prices` works on anything at or below the
strategy timeframe. The UI offers only the modes the loaded data can actually support, and
says why the others are unavailable — rather than accepting a selection it will quietly
approximate.

`real_ticks` is MT5's "Every tick based on real ticks" and is the only mode whose numbers are
reportable. `open_prices` exists because it is ~30× cheaper, which is what makes large sweeps
affordable (§11). MT5's interpolated "Every tick" mode is deliberately **not** implemented —
the owner has real ticks, and synthesising ticks from M1 bars manufactures fills.

**Discipline, enforced in code:**

- Sweep in `open_prices`, then **re-validate the surviving candidates in `real_ticks`**. This is
  the standard MT5 workflow and it falls out of the cost table naturally.
- **A result produced in any mode other than `real_ticks` is labelled as such in every report**,
  the same way an in-sample-only result is (§12).
- The cross-mode delta is itself a diagnostic: `open_prices` vs `real_ticks` should differ in a
  bounded, explainable direction (worse, by roughly spread + slippage). A strategy whose edge
  *disappears* between modes was living inside the bar, and a strategy whose edge *improves*
  indicates a bug in one of the two paths.

---

## 7. Architecture — signals on bars, fills on ticks

### Design

```
PASS 1 — SIGNAL                   PASS 2 — EXECUTION (two engines, one contract)
numpy/numba on signal-TF bars  →  sweep:     our numba bar engine     2.2 ms
emits INTENTS with timestamps     validate:  NautilusTrader on ticks  ~4 min
```

**One intent contract, two consumers.** Pass 1 does not know which engine will run its
intents; that is a per-run choice (§6). The adapter that feeds intents into NautilusTrader is
ours, and it is the piece to get right — the rest of the tick engine is bought (§3.1).

This is how MT5 works, and it is no longer an exotic architecture. Pass 1 is pure, vectorized
and stateless. Pass 2 is a single sequential pass that owns all broker emulation.

**Note what changed:** in rev 3 this split was partly a *performance* necessity. It is not any
more — a numba fill loop handles 6M bars in 80 ms `[measured]`. The split survives because
signal timeframe and execution fidelity are genuinely different concerns and the fill engine
needs a sequential pass anyway. Treat it as separation of concerns, and stop contorting around
it.

### Intent schema

Because pass 1 is Python, an intent is just a record. No numeric-channel encoding, no
one-per-bar ceiling.

```python
@dataclass(frozen=True, slots=True)
class Intent:
    t: int                       # ms, UTC
    kind: Literal["place", "cancel", "modify", "close"]
    id: str
    side: Literal["long", "short"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    price: float | None
    sl: float | None
    tp: float | None
    trail: TrailSpec | None
    size: SizeIntent             # risk_pct | fixed_lots | notional
    oca_group: str | None
    valid_until: int | None      # ms
```

Pass 1 emits `list[Intent]`. Multiple intents per bar are fine. A same-bar cancel-then-place is
fine. Ordering within a timestamp is the list order, which is deterministic.

### What pass 2 owns

- **OCA properly** — sibling cancel/reduce on fill
- **Lot rounding** to `lot.step`, min/max clamping, and honest reporting when a risk-based size
  rounds to zero
- **Contract multipliers**, tick value, per-instrument P&L
- **Spread applied directionally** from the real tick stream — buy at ask, sell at bid
- **Slippage**, configurable: fixed / spread-proportional / gap-based (fill at the worse of
  level and next print — the realistic model for stop entries)
- **`stops_level` / `freeze_level` enforcement**, with rejections reported not swallowed
- **Margin and free-margin checks**, margin call, stop-out, leverage
- **Swap** at the server-time rollover, triple-swap day per spec; **commission** per spec
- **Currency conversion** to account currency
- **Pending order expiry** (`valid_until`)
- **No fill when there are no ticks** — a gap is a gap

### Determinism

The agent layer is not reproducible. **The engine must be, bit for bit.**

- `numba.njit(fastmath=False)` everywhere. `fastmath=True` reorders float operations.
- No `parallel=True` on any float reduction. Parallelize *across* evaluations, never inside one.
- Fixed dtypes: `int64` ms for time, `float64` for prices and accounting.
- Accumulate P&L in a documented, fixed order.
- Stable tick sort with source-order tie-break (§4.4).
- Same inputs + same seed ⇒ byte-identical trade list. There is a test for this.

---

## 8. Indicator layer and parity

This section replaces rev 3's PineTS-capabilities section. The engine owns its indicator
semantics now, so the semantics must be pinned rather than inherited.

### 8.1 Two implementations, one contract

`engine/indicators/` ships each indicator twice, deliberately:

- **`wilder.py`** — readable numpy reference. Obviously correct, easy to check by eye, pinned
  to the PineTS oracle by `tests/golden/`.
- **`fast.py`** — the same recurrences in numba, for the optimizer's inner loop. Pinned to the
  reference by `tests/golden/test_fast_parity.py`, so it inherits the oracle guarantee
  transitively.

The reason is measured. Profiling one evaluation over 40,000 bars `[measured]`:

| | time | share |
|---|---|---|
| signal block (numpy reference) | 24.39 ms | **96%** |
| backtest loop (numba) | 0.20 ms | 1% |
| metrics | 0.94 ms | 4% |

The fill engine is effectively free; the cost was entirely the reference's Python-loop
recursions. Swapping to the compiled path took one evaluation from **25.5 ms to 2.2 ms**
`[measured]` with bit-identical results — which is exactly what the parity test guarantees. A
fast path not pinned to a verified reference is an unverified reimplementation sitting on the
hottest code path in the system.

The Wilder family is implemented and validated:

- `sma`, `ema` (SMA-seeded, `alpha = 2/(n+1)`)
- `rma` (Wilder, SMA-seeded, `alpha = 1/n`)
- `rsi` (RMA of up/down moves), `atr` (RMA of true range)
- true range with first-bar `high - low`

Pine and MT5 both use Wilder smoothing for the RSI/ATR family, which is why one implementation
satisfies both.

### 8.2 Golden fixtures — the rule

**No indicator ships without a frozen numerical fixture.** For each indicator:

1. Generate reference values with the **PineTS dev oracle** (§3) on a fixed synthetic series.
2. Freeze inputs and outputs as a Parquet/CSV fixture, checked into the repo.
3. A test asserts the Python implementation matches to `1e-9` absolute, **and that warmup NaN
   counts match exactly** — warmup semantics are where implementations usually diverge.
4. Where the owner has an MT5 reference for the same indicator, add a second fixture from MT5
   and reconcile. Where MT5 and Pine disagree, **MT5 wins** and the divergence is documented.

Current status: `ema`, `rma`, `rsi`, `atr` match PineTS to **5.0e-11** with exact NaN parity
`[measured]`. `sma` initially drifted to 1.8e-9 via a cumulative-sum implementation — float
error in *our* code, not a semantic divergence — and now clears the 1e-9 bound using a sliding
window. All six golden tests pass (`tests/golden/`). That drift is a useful reminder: the
failure mode when you own the semantics is usually numerical, not conceptual.

### 8.3 Adding concepts

The LuxAlgo concepts the owner actually wants — order blocks, FVGs, session levels, swing
structure — are read from Pine via MCP and implemented in Python. This work existed in rev 3
too (the agent wrote signal blocks from concept understanding either way); only the target
language changed. **The agent is what writes them** — that is the point of the system.

Do **not** adopt a Pine-semantics-in-Python framework. Pine's per-bar execution model is why it
is slow; numpy's vectorized model is why this is ~500× faster. Importing Pine's model reimports
the constraint we just escaped.

---

## 9. Harness / signal-block split

The LLM does not write whole strategies. The strategy is composed.

**Harness** — hand-written, version-controlled, never generated. Owns the decision gate, intent
emission, duration→bars conversion, session helpers, the parameter-spec convention, and the
structural guarantee that a signal block cannot touch order state.

**Signal block** — generated, narrow slot, a typed function:

```python
def signal(bars: Bars, p: Params, h: Helpers) -> SignalOutput:
    """MUST return: long_level, short_level, long_armed, short_armed,
                    stop_distance, target_distance
       MAY use:     bars.*, indicators.*, h.session_*, p.*
       MUST NOT:    place orders, hold position state, read future bars"""
```

**Why:** most LLM-generated strategy code that "sometimes doesn't work" fails in order
management, not indicator logic — and it fails *silently*, producing a plausible backtest rather
than an error. Removing that surface eliminates the majority of failure modes and enforces the
contract structurally rather than by prompting. It also keeps the system general: any concept,
Library or not, reduces to "compute levels, arm conditions."

**Python makes this stronger than rev 3 could:**

- **Enforcement by namespace, not by prompt.** The block receives only `bars`, `p` and `h`.
  Order functions are not in scope, so calling one is a `NameError`, not a silent bug.
- **Static validator — built** (`engine/signal/validator.py`, 20 tests). An AST walk before
  execution rejecting imports, `open`/`eval`/`exec`/`compile`, dunder attribute access, and
  forward or negative-start indexing on `bars` (the look-ahead check). It also checks the
  function's shape: `signal(bars, p)` returning all four required keys.
- **Honest limit, found while building it.** A restricted namespace is *not* a sandbox. Python
  cannot be sandboxed by withholding names, and numba's dispatchers need `__import__` at call
  time regardless — so the namespace has to expose it. Withholding names still catches
  *mistakes* (a block reaching for an order function gets a `NameError`), and that is worth
  having, but **the AST validator is the enforcement** and neither layer is a security
  boundary. Never run a signal block the owner did not initiate.
- **Signal blocks are unit-testable.** Call the function on a fixture and assert on its output.
  Rev 3 could only test through a CLI subprocess.

**Repair loop:** generate → validate → run on a small fixture → on failure the agent reads the
real Python traceback and retries, up to the owner's limit → on exhaustion, mark dead and log
the failure pattern. Tracebacks are considerably more useful than transpiler errors.

**The failure taxonomy is the highest-value output of month one** — it tells you what to put in
the skills. See §20, which is sequenced so this is actually possible in month one.

---

## 10. Execution engine and how it is verified

The fill engine is the hardest correct-by-construction piece in the system: a stateful
sequential loop owning lots, margin, swap, OCA, FX, stop-out and order validity. Rev 3 named it
as such and then offered no verification strategy. This is that strategy.

### 10.1 Property-based invariants

Under `hypothesis`, over randomized tick streams and intent sequences, the following must hold
at every tick:

- `equity == cash + Σ unrealized` (to float tolerance)
- `Σ closed-trade P&L == equity delta − Σ unrealized`
- `position == Σ signed fills`
- no open position without sufficient margin at the time it was opened
- every position size is an exact multiple of `lot.step`, within `[lot.min, lot.max]`
- **no fill price lies outside the bid/ask of the tick it filled on**, ± configured slippage
- no fill occurs at a timestamp with no tick
- an OCA sibling is cancelled or reduced on fill, never both filled in full
- accepted SL/TP distances all satisfy `stops_level`

### 10.2 Cross-checking against MT5 — on demand, not as a gate

The owner has decided **not** to gate the build on reconciling against an MT5 backtest. That
is a real cost and it should be stated rather than buried: without it, this engine has no
external ground truth. §10.1's invariants prove the engine is *internally consistent*; they
cannot prove it fills the way a broker does. If a systematic bias exists — a spread applied on
the wrong side, a stop evaluated one tick late — every invariant still passes.

If the owner wants the check, it is a manual one: reimplement a single strategy as an MQL5 EA
by hand, run it in MT5's Strategy Tester over the same span, and reconcile. Worth doing once,
on one strategy, before trusting a number intended for real money — not as a build gate. The
reconciliation to run:

- entry and exit timestamps, and prices: exact
- per-trade P&L: within commission rounding
- final balance, max drawdown, trade count: within rounding

A divergence is a bug in this engine, in the instrument spec, or in the server-timezone
handling (§4.3).

### 10.3 These invariants already work

Not a hypothetical. The first run of `test_closed_pnl_reconciles_with_the_equity_curve`
failed: entry commission was charged to the cash ledger but only exit commission was recorded
in each trade's P&L, so the trade list and the equity curve disagreed by one commission per
trade. Small, entirely plausible, and invisible in any individual number — every trade looked
right, the equity curve looked right, and they were inconsistent with each other.

That is the whole argument for §10.1 in one example. With MT5 reconciliation no longer a build
gate (§10.2), these invariants are the primary defence, and they earn it.

### 10.4 Differential testing — and the reference we got back

`real_ticks` vs `m1_ohlc` vs `open_prices` on the same strategy should differ in a bounded,
explainable direction (§6). Assert the direction; investigate anything outside it.

**VectorBT is the independent cross-check (§3.1).** On a simple strategy — percentage stops,
wide relative to bar range — our engine and VectorBT are independent implementations by
different authors and should agree closely. Divergence means one has a bug.

It is weaker than MT5 reconciliation: agreement proves consistency, not realism. But an
independent implementation is considerably better than an internal-consistency argument alone,
and it costs one test rather than a subsystem.

### 10.5 Reproducibility

Same store version + same spec version + same policy + same seed ⇒ byte-identical trade list.
Asserted in CI, not assumed.

---

## 11. Optimization

Search space is declared, not discovered from a script. A `Params` dataclass with typed,
bounded fields gives the optimizer bounds, types and step for free — and unlike rev 3's
`getInputsMeta()`, it has no stale-cache trap (§2).

```python
@dataclass
class Params:
    ema_len:   Int(20,  min=5,  max=200, step=1)
    atr_mult:  Float(2.0, min=0.5, max=5.0, step=0.1)
    session:   Enum("london", ["london", "ny", "asia"])
```

The UI renders each parameter with start / stop / step (MT5-style) and enable/disable. Signal
timeframe and instrument may themselves be swept — subject to §4.5's duration requirement.

| Mode | Use |
|---|---|
| **Exhaustive grid** | Small spaces, full factorial |
| **Random search** | Higher dimensions; beats grid per unit compute |
| **Genetic** | Large spaces, cheap evals |
| **Bayesian / TPE** | Expensive evals, moderate dimensionality (Optuna) |
| **Coordinate descent** | Sensitivity analysis, not a winner-finder |
| **Multi-objective (NSGA-II)** | Pareto front. No single "best" |

**Objectives** (pluggable): net profit · profit factor · expectancy (R) · Sharpe · Sortino ·
Calmar/CAR-MDD · recovery factor · SQN · custom expression.

**Default is not net profit.** Optimizing raw return reliably surfaces the highest-variance
survivor — it wins the backtest and blows up live. A drawdown-aware objective
(`return_over_maxdd`) is the default; the owner sets it in §17.

**Two guards every objective needs, learned from building it:**

- **A minimum-trade floor.** Without it a configuration that took two lucky trades outranks
  one that took four hundred. The commonest way a sweep produces a nonsense winner, and it is
  silent — the results table looks perfectly reasonable.
- **Non-finite scores must sort last, not first.** `profit_factor` is `+inf` when a
  configuration had no losing trades, which is the global maximum of an unguarded search, so a
  three-trade fluke wins every sweep. Both are enforced in `objective_value`.

**The comparison count is cumulative, not per-sweep.** §12's deflated Sharpe adjusts for the
number of configurations tried, and that is the total across the whole research programme for
a family — three sweeps of 1,000 is 3,000 trials. A `SearchResult` counts its own run; the
library (§16) accumulates across runs and the family budget checks the total. Reporting one
sweep's count inflates DSR.

### Cost model

Measured on the working slice (`bench/demo_loop.py`), not extrapolated: a full evaluation —
signal block, backtest, metrics — over 40,000 M15 bars costs **2.2 ms** `[measured]`.

| | per eval | 10,000 evals | × 10 walk-forward folds |
|---|---|---|---|
| 40k bars (≈ 1 yr M15) | 2.2 ms | 22 s | ~4 min |
| ~370k bars (≈ 1 yr M1) | ~20 ms | ~3.5 min | ~35 min |
| `real_ticks` revalidation | ~0.2 s | — | top-N only |

For comparison, rev 3's PineTS path was ~3.3 s/eval — 9.7 hours for that same 10k grid, four
days with folds. The whole §12 validation matrix went from arguably unaffordable to
interactive.

**Where the time goes decides what to optimize.** Per §8.1 the fill engine is ~1% of an
evaluation and the indicator layer is nearly all of it. Do not micro-optimize the backtest
loop; keep the indicator fast path compiled and cached.

**Parallelism:** across evaluations, never inside one (§7). On Linux `fork` gives workers
copy-on-write access to the loaded bar arrays, so the store need not be copied per worker, and
`cache=True` on every `njit` stops them re-compiling.

So: **sweep in `open_prices`, re-validate survivors in `real_ticks`.** Parallelize across
evaluations with `joblib`/`multiprocessing` — never inside an evaluation (§7, determinism).

**Discipline:** coarse grids over fine (looking for plateaus, not peaks — a sharp optimum is a
fitting artifact) · optimize on train folds only · multi-symbol is a robustness test, not a
search space (fit on one, require survival on others *unchanged*) · **log the comparison
count**, because §12's deflated Sharpe needs it.

---

## 12. Validation

Independently toggleable. A run reports all enabled verdicts, not one pass/fail.

| Mode | What it tests |
|---|---|
| **In-sample only** | Nothing. Must be labelled as such in every report |
| **IS/OOS split** | Simple forward test |
| **Walk-forward, rolling** | Adaptation over time, sliding fixed-width train window |
| **Walk-forward, anchored** | Train window grows from a fixed start |
| **Purged k-fold + embargo** | Proper time-series CV. Purge overlapping labels, embargo around test folds |
| **MC — trade shuffle** | Whether results depend on lucky ordering |
| **MC — block bootstrap** | Ruin and drawdown distribution (prop-firm-sim) |
| **MC — cost perturbation** | Randomized spread/slippage within observed bounds |
| **Cross-symbol** | Same parameters, other instruments, unchanged |
| **Cross-timeframe** | Neighbouring signal timeframes (requires durations, §4.5) |
| **Cross-mode** | `open_prices` vs `real_ticks` (§6) |
| **Parameter plateau** | Neighbourhood surface, 2D/3D in the UI. A peak is a red flag |
| **Randomized-entry benchmark** | Same exits and sizing, random entries. If the strategy doesn't beat it, the entry logic contributes nothing |
| **Holdout** | Final. Touched once — §13 |

### Three rules the fold layer must enforce

Each was found while building `engine/validate/`, and each silently inflates out-of-sample
results if missed.

**A trade belongs to the fold containing its ENTRY.** Trades straddle boundaries, and both
obvious alternatives are wrong: counting a trade in both folds double-counts it, and counting
by exit lets a position opened on in-sample information score out-of-sample. One rule, applied
everywhere (`attribute_trades`).

**The purge window must be at least the longest trade duration.** Purged k-fold exists because
a label computed over a window overlapping the test set leaks. In a backtest the label *is* a
trade's outcome, so its span is the trade's duration. A purge of "10 bars, feels about right"
against trades that routinely last 40 is not a purge — the leak is simply unmeasured.
`purge_bars_for` derives it from observed durations at a high quantile.

**Walk-forward nests a full search inside every train fold.** The cost is `folds x
evaluations`, not `evaluations`, and each fold's winner is scored out-of-sample exactly once.
The `walk_forward` signature enforces this structurally: the search callback receives only
train indices, so it cannot read the test window even by mistake. Re-tuning after seeing a
test fold turns the whole exercise into an expensive way to overfit (§13).

**Walk-forward efficiency** — mean out-of-sample score over mean in-sample — is the headline
number. Near 1.0 means the optimization generalized. Far below means the train folds were
fitted. Well above ~1.2 usually means the test windows were easier, not that the strategy
improved; check before celebrating.

**Overfit detectors, reported alongside:** PBO (probability of backtest overfitting) via
combinatorially symmetric CV · deflated Sharpe ratio (adjusts for trial count, which is why §11
logs comparisons) · minimum backtest length implied by observed Sharpe.

**Python advantage worth using:** PBO, DSR and purged k-fold have published reference
implementations in the `mlfinlab` lineage. Check ours against them. In TypeScript these would
have been written blind.

**Every report states its modeling mode and its validation mode.** A `real_ticks` walk-forward
result and an `open_prices` in-sample result are not the same kind of object.

---

## 13. Research integrity mode

**Configurable, default on. The owner may disable it — but should understand the trade.**

The risk: if the agent reads out-of-sample results, reasons about why a candidate failed, and
proposes a variant, that is gradient descent on the test set with the agent's reasoning as the
optimizer. Each step looks honest. After N iterations the survivor is fitted to the OOS folds as
thoroughly as if they had been optimized on directly — and every metric says it's clean, because
there is no parameter sweep in the log, only a sequence of sensible decisions.

### Enforce by isolation, not by denylist

Rev 3 proposed denying `cat`, `grep` and `jq` against the OOS path. **That does not hold.** The
agent has Bash; you cannot enumerate the ways to read a file — `python -c`, `node -e`, `sed`,
`awk`, `tail`, `find -newer`, or a throwaway test script that reads it and asserts on the
contents. A denylist on a shell-capable agent is theatre.

Correct design: **OOS artifacts must not exist anywhere the agent can reach.**

- The OOS evaluation runs as a **separate process** with its own working directory, outside the
  agent's cwd, and returns only `runs/<id>/verdict.json`.
- `verdict.json` contains pass/fail per family and nothing else — no metrics, no equity, no
  trade counts.
- Raw OOS artifacts are written to a path the agent's process cannot read (separate directory
  ownership, or a container boundary if the owner wants the stronger guarantee).
- This is *cheaper* to build than the denylist, and it actually works.

| Data | Agent access (strict) |
|---|---|
| In-sample — equity, trades, per-fold IS metrics | Full. Reason and propose freely |
| Out-of-sample | Pass/fail per family only |
| Family that failed OOS | Closed. No variants |
| Family budget | Declared before the run; agent cannot request more |

**Open mode:** agent sees everything. Faster iteration, and the OOS number stops meaning what it
claims. **The report must label which mode produced it.**

**Holdout, both modes:** touched once, result hashed, re-runs against the same family refused in
code. If the system permits a re-run, it is not a holdout.

**Leakage through the instrument spec:** costs and swap (§5) are values you will tune. Tuning
them against the holdout window leaks just as surely as tuning parameters. Spec versions are
pinned per run and stamped into the artifact.

---

## 14. The agent — MCP-first, no API keys, one script

The agent is **Claude Code running locally on the owner's own subscription**.

> **No API keys. Ever.** No `ANTHROPIC_API_KEY`, no billing integration, no raw API client.
> Everything the agent does beyond reading and writing files, it does through **MCP**.

That is the design. The rest of this section is its consequences.

### 14.1 The MCP surface

| Server | Transport | What the agent uses it for |
|---|---|---|
| **LuxAlgo Library** | `https://mcp.luxalgo.com/mcp`, keyless (or `npx @luxalgo/mcp`) | **Understand a concept.** Read real Pine source to learn what an order block or an FVG actually *is*, then implement it in Python |
| **Edge Stats** | stdio | Conditional frequencies over M1 bars, N and a Wilson CI on every answer. "Does this setup even occur, and how often" — *before* writing a strategy for it |
| **Prop Firm Sim** | stdio | Block-bootstrap Monte Carlo over the R-multiple series a backtest produced |
| **QUANTOR engine** | stdio, **ours** | Run backtests, sweeps, validation; read results; render on the chart (§14.2) |

The Library is used two ways, and both emit Python only — no Pine is generated, stored or run:

- **As reference.** Read Pine to learn what a concept *is*, then implement it.
- **As a strategy source** (§14.9). Port published strategies wholesale, backtest them on the
  owner's data, and screen variants. Efficient, and statistically dangerous in a specific way
  that §14.9 designs for.

### 14.2 The engine is an MCP server

Without it the agent has Bash and a Python package, so every backtest is an ad-hoc script it
writes, runs, and parses its own stdout from. Brittle, unauditable, and far more latitude than
the task needs.

With it, the agent calls typed tools and the engine owns the semantics:

```
data_list()                          -> loaded sources, resolution, GMT offset, quality
strategy_save(name, description,
              signal_block, params)  -> new version, lineage recorded (§16.3)
strategy_get(id, version)            -> the signal block + its parameter spec
backtest_run(id, version, params,
             symbol, timeframe, span,
             modeling_mode)          -> metrics, trade list, equity curve ref
optimize_run(id, param_spec, budget) -> results table, comparison count
validate_run(id, params, modes)      -> per-mode verdicts (§12)
chart_apply(id, version)             -> render signals + trades on the chart
runs_query(filter)                   -> history from the library (§16)
```

Three things this buys:

- **Research integrity becomes enforceable** (§13). In strict mode the OOS tools return a
  verdict and there is no filesystem path to read around it, because the agent was never handed
  one. That is the process-isolation fix, for free.
- **Every action is a logged tool call**, so "what did it actually do" is answerable from the
  transcript rather than reconstructed from shell history.
- **The agent cannot invent its own backtest.** It cannot quietly change the fill model, the
  cost assumptions or the fold geometry, because those live behind the tool.

With the engine behind MCP the agent barely needs Bash, which tightens the permission surface
considerably.

### 14.3 One emission: Python

A strategy is **a Python signal block plus its parameter spec** (§9, §11). That is the whole
artifact. There is no second representation, no spec-to-code generation step, and nothing to
keep in sync.

Pine Script emission was considered and dropped, and the reasoning is worth keeping because the
idea recurs: two independently emitted scripts of "the same" strategy drift — a condition on a
different bar, an indicator seeded differently — and then neither is authoritative and you need
a comparison harness to find out which lied. One emission has no such failure mode.

A short human-readable **description** travels with each version, but it is metadata for the
library and the conversation (§15.2), not a source the code is generated from. The code is the
strategy.

**What dropping Pine also bought:** the chart now renders **the actual backtest's** signals and
trades (§14.4), not a parallel implementation's. Chart verification went from "these two things
should agree" to "this is what happened".

### 14.4 Verification by looking at the chart

The owner applies the strategy, looks at where it entered and exited, and judges. **First-class
step, not a consolation prize.** It catches what no metric does:

- entries clustering at one time of day, or only in one regime
- stops visibly inside normal noise
- signals firing somewhere the owner would never have taken the trade
- a strategy that is technically profitable and obviously untradeable

Metrics say whether it made money. The chart says whether it is the strategy you meant. The
loop in §15.1 runs on both, and because the chart draws the engine's own trade list, what you
see is exactly what was measured.

### 14.5 Why indicator parity still matters

§8 pins our `ema`/`rma`/`rsi`/`atr` to a Pine oracle by golden fixtures. That was previously
justified by "the two emissions must agree". With one emission the reason is different and
arguably stronger:

**The agent reads Pine from the Library and implements it in Python.** If our `ta.ema` seeds
differently from Pine's, the agent faithfully implements a Library concept and gets a strategy
that is not the concept. The fixtures are what make "implement this Library idea in Python"
mean something precise rather than approximately.

### 14.6 What "no API keys" costs, and what it does not

One real consequence, stated rather than discovered: **`--bare` requires `ANTHROPIC_API_KEY`
and ignores subscription login.** So the reproducibility recipe earlier revisions assumed is
unavailable.

What still works:

- `--mcp-config <file>` — pin the exact server set per run, and check `system/init` for
  `mcp_servers` / `mcp_server_errors` to fail fast when one does not load.
- `--settings <file>` and `--append-system-prompt-file` — pin instructions explicitly.
- `--allowedTools` scoped to the MCP tools plus `Read`/`Edit`.
- `--permission-mode`, `--max-turns` (§17).

What does not: we cannot guarantee a stray user-level hook, skill or personal MCP server did
not influence a run. So **record what actually resolved** — session id, model, runtime version,
the `system/init` server list — into the library (§16) and treat it as *provenance*, not a
reproducibility guarantee. The engine stays deterministic (§7); the agent layer is logged, not
reproduced.

Spend is also not enforceable client-side without the API path, so §17's `agent_spend` ceiling
becomes advisory. `--max-turns` and wall-clock still bite.

### 14.7 Runtime

**Claude Code** — the supported runtime.

- `claude -p "<prompt>"` non-interactive; `--output-format stream-json` with
  `--include-partial-messages` for streaming into the sidebar
- `--resume <session-id>` / `--continue` — what makes §15.1's conversation continue across
  restarts; transcripts are `.jsonl` and persist as run artifacts
- `--json-schema` to enforce structured envelopes
- SIGINT ends a turn cleanly (§15.4's cancel); SIGTERM leaves it unfinished
- Piped stdin capped at 10 MB — attachments go by path, never piped
- The Python Agent SDK is preferred for the interactive sidebar: the engine is Python, so there
  is no language boundary

**Codex — deferred, not designed around.** MCP calls under `codex exec` have been reported
auto-cancelled with non-interactive approval, and this design is MCP-first, so Codex support is
gated on that being fixed or worked around. Not a priority. The provider adapter (§15) should
keep the runtime swappable so adding it later is cheap, but nothing in the plan should wait on
it.

### 14.9 Screening the Library as a strategy source

The Library holds hundreds of published strategies. Porting them to Python,
backtesting them on the owner's data, and generating variants is a far more
efficient way to find candidates than inventing from scratch — **and it is also
the fastest way to fool yourself**, so the statistics need designing, not just
the workflow.

**Why it is a good idea.** A published Library strategy was selected on
historical performance *somewhere else* — someone else's instruments, someone
else's period. Running it on the owner's XAUUSD data is therefore a genuine
out-of-sample test of that author's hypothesis. That is a much stronger starting
point than an idea with no track record at all.

**Why it needs care.** Generating variants and keeping the winners re-overfits
on the owner's data, and at a scale hand-crafted strategies never reach.

#### Screening mode

This is a **batch** operation, not a conversation. It runs unattended and hands
the conversation a ranked shortlist:

```
screen_library(concepts, data, symbols, budget_per_family)
  for each concept:
      read its Pine from the Library (MCP) -> implement in Python
      backtest at the author's defaults        <- the honest first number
      sweep a SMALL neighbourhood of variants
      record everything, survivors and failures alike (§16.4)
  -> ranked table + the null comparison below
```

The default backtest comes first and is reported separately. It is the only
number in the whole exercise that has not been selected on, and it is the one
worth trusting most.

#### Four rules that make the screen honest

1. **One Library concept = one family** (§13). Variants within a family share a
   hypothesis, so the family budget and the closed-on-OOS-failure rule apply
   exactly as designed. This is the use case §13 was built for; screening makes
   it load-bearing rather than theoretical.

2. **Comparison counts are cumulative and now enormous** (§11). Two hundred
   concepts at fifty variants each is ten thousand trials, and the deflated
   Sharpe ratio adjusts for precisely that number. A strategy that looks good
   out of ten thousand is *far* less impressive than one that looks good out of
   ten. If the count is not tracked across the whole screen, every DSR reported
   afterwards is wrong in the flattering direction.

3. **Compare the best result against a null, not against zero.** This check only
   becomes available *because* of the volume, and it is the most valuable thing
   screening gives you. Run the same screen against randomized-entry strategies
   with matched trade counts and holding times (§12), and against shuffled
   returns. **If the best real strategy is inside the distribution the null
   produces, the screen found nothing** — however good that one equity curve
   looks. Report the best result's percentile against the null, every time.

4. **Cross-symbol survival is the strongest filter, and screening makes it
   cheap.** A concept that survives *unchanged* on five instruments is worth
   more than one tuned to fit one. Rank on cross-symbol consistency before
   ranking on any single-symbol metric.

#### What screening must not become

Screening produces *candidates*, never conclusions. A survivor enters the normal
loop (§15.1) — conversation, refinement, the chart, then full validation (§12)
on a holdout that the screen never touched. The holdout must be reserved
**before** the screen runs, or it has been used for selection like everything
else.

The failure side is as valuable as the survivors: a concept family that fails on
the owner's data is recorded with its reason (§16.4), which is what stops the
agent re-proposing it in three weeks.

### 14.8 Project memory

```
CLAUDE.md                 harness contract, §4.5 rule, forbidden calls, repo map
.claude/skills/
  signal-blocks/          the slot contract, worked examples, the static validator's rules
  indicators/             our semantics, parity fixtures, accumulated gotchas
  exemplars/              validated signal blocks, tagged by concept family
.mcp.json                 luxalgo, edge-stats, prop-firm-sim, quantor-engine
```

Start the exemplar corpus with ~10 hand-written blocks; append every candidate that validates
and produces sane trade counts. The failure taxonomy (§9) is a query over the rejected ones.

## 15. The application — the conversation is the product

The chart, the results panel and the optimizer are all instruments. **The thing the owner
actually uses is a conversation with an agent that knows their data, their strategies and
everything already tried.**

### 15.1 The conversation

A chat is not a prompt box. It is a long-lived working session that can span weeks, produce
several strategies, and be picked up where it was left.

**It carries more than one strategy.** A session might start as "what do you think about
London-open breakouts", branch into three variants, park two, and keep refining the third.
Each strategy has its own version history (§16.3); the conversation is the thread that links
them. A message can reference any of them — "go back to the second one and try a wider stop"
must work.

**It continues.** Closing the window is not ending the work. Reopening a conversation restores
the agent's context: the strategies it created, their current versions, which runs have been
executed and what they returned, what was already rejected and why. Backed by `--resume` on
the persisted transcript (§14) plus a re-read of the strategy state from the library.

**Refinement is the normal mode, not a special case.** Most turns are not "write me a
strategy" — they are "the drawdown is too deep, try a trailing stop", "what if entries only
fire in the London session", "why did it lose money in March". Each such turn that changes the
strategy produces a **new version** with its parent recorded. Nothing is edited in place, so
"go back to how it was on Tuesday" is always answerable.

**Turns do different amounts of work, and the UI says which.** A reply can be discussion
(free), a code change (cheap), a backtest (seconds), or an optimization (minutes and real
agent spend). The owner should never be surprised by a bill or a wait, so the mode is explicit
before the turn runs:

| Mode | Does | Costs |
|---|---|---|
| **Brainstorm** | Discusses, pulls Library concepts over MCP, checks whether a setup even occurs via Edge Stats. No code | Agent tokens only |
| **Build** | Writes the spec, emits **both scripts** (§14.3), runs a fast backtest, applies the Pine to the chart | Tokens + a fast backtest |
| **Analyze** | Reads results and explains them — subject to §13 in strict mode | Tokens |
| **Optimize** | Configures and launches a sweep or validation run | Tokens + real compute |

### 15.1.1 The loop, concretely

This is the cycle the whole system exists to serve:

1. **Brainstorm.** "What about order blocks on the London open?" The agent reads the concept
   and its real Pine source from the Library over MCP to learn what it *is*, and — before any
   code — asks Edge Stats whether the setup actually occurs often enough to matter. A concept
   that fires eleven times a year is worth knowing now, not after an afternoon of work.
2. **Build.** The agent writes the Python signal block and saves a version.
3. **Test.** `backtest_run`. Metrics come back into the conversation.
4. **Look.** `chart_apply` draws **that run's own** entries, exits and stops on the chart. The
   owner sees where it actually traded (§14.4) — no second implementation, so nothing to
   disagree.
5. **Refine.** "The stops are too tight in the Asian session." A new version, parent recorded.
6. **Repeat**, then validate the survivors properly (§12) rather than the first thing that
   looked good.

Steps 3 and 4 answer different questions — *did it make money* and *is it the strategy I
meant* — and a candidate needs both before it is worth optimizing.

### 15.2 What the agent is given each turn

An open question worth probing early (§19, Probe 7), because it decides both cost and quality.
The full transcript grows without bound and eventually dominates the context; too little and
the agent re-proposes something rejected three days ago.

The likely shape: a **compact strategy state** regenerated from the library each turn —
current strategies, their spec, parameters, latest metrics, and a one-line reason for each
rejected variant — plus the last few turns verbatim, plus retrieval into the older transcript
when a message refers back to something specific.

**The signal block itself usually should not be in context.** The description plus the
parameter spec is enough to reason about a strategy; the agent fetches the code with
`strategy_get` when it actually needs to edit it. Carrying every strategy's source in every
turn is the fastest route to an unaffordable conversation. Measure before committing.

### 15.3 Layout

**Left: chart.** Vela. Symbol and timeframe pickers, drawing tools, indicator menu. Strategies
render here — plotted levels, entry/exit markers, SL/TP lines, equity in a lower pane. The
Python backend computes series and pushes them to Vela as plain data over the API.

**Right: the conversation.** An Agent SDK session (§14). Accepts text, **file attachments**
(existing strategy source, CSVs, PDFs) and **images** (chart screenshots, hand-drawn setups).
Attachments are written to a scratch dir and referenced by path — never piped, given the 10 MB
stdin cap.

**Bottom: results panel.** Tabbed, MT5-style — Report · Trade list · Equity · Optimization
results table (sortable) · Optimization surface (2D/3D) · Validation verdicts. Modeling mode
and validation mode shown on every tab.

**Library tabs** — Strategies, History, Compare, Conversations, Data (§16.5). How the owner
navigates their own work when they are not in a conversation.

### 15.4 Runs

**Lifecycle (MT5-like):** Configure → Start → progress with cancel → results. Checkpointed and
resumable. Every run is a row in the library (§16) carrying its policy, store, spec and
strategy versions, its seed, its artifacts and its agent transcript — so any past run reopens
exactly as it was. **Cancel maps to SIGINT** (ends the turn cleanly) rather than SIGTERM.

A run launched from a conversation is linked to the turn that launched it, so "what was I
trying when I ran this" is answerable from either direction.

**Ceilings, any one of which stops a run** (§17): wall clock · evaluations · agent turns.
Spend is not enforceable without the API path (§14.6), so it is advisory only.

### 15.5 Shell

Local web — FastAPI + uvicorn serving a browser frontend. Resolves rev 3's Electron/Tauri
question, and with no AGPL in the stack there is no network-service concern.

---

## 16. The library — nothing generated is thrown away

**This is a platform, not a script runner.** Everything the system produces — strategies,
versions, runs, optimization tables, validation verdicts, agent conversations, imported data
sources — is catalogued, linked to what produced it, and browsable later.
Work that vanishes when a window closes is work paid for twice.

This section exists because the alternative is the default. Research tools drift into writing
results to a directory, printing a summary, and moving on; three weeks later nobody can say
which parameters produced the good equity curve, or whether a candidate was already tried and
rejected.

### 16.1 What is persisted

| Entity | Holds | Why it must survive |
|---|---|---|
| **Data source** | file hash, symbol, kind, resolution, GMT offset, row count, span | A run is meaningless without knowing which bytes it read |
| **Strategy** | name, tags, notes, archived flag | The unit the owner thinks in |
| **Strategy version** | signal-block source, params spec, parent version, origin | Every edit is a new version. Nothing is overwritten |
| **Run** | kind, symbol, timeframe, modeling mode, span, seed, all four stamped versions, status, artifact path | The reproducibility record |
| **Optimization** | mode, objective, budget, **comparison count**, full result table | §12's deflated Sharpe needs the trial count; §18's ceilings need the spend |
| **Validation** | per-mode verdicts, overfit detectors, holdout-touched flag | §13 depends on the holdout being provably touched once |
| **Agent session** | transcript, model, runtime version, resolved MCP servers, cost | §15's determinism record, and the raw material for the failure taxonomy |

### 16.2 Catalogue and artifacts

**SQLite for the catalogue, content-addressed files for the artifacts.** The database holds
rows small enough to query — identifiers, parameters, metrics, statuses, lineage. Equity
curves, trade lists, optimization surfaces and transcripts live on disk under their content
hash, so identical results deduplicate and nothing is ever overwritten in place.

```
library.db                     the catalogue
artifacts/<sha256[:2]>/<sha256>    equity curves, trade lists, surfaces, transcripts
```

SQLite because it is a single file the owner can copy, back up, and open with any tool — and
because this is a local application with one writer. Nothing here needs a server.

### 16.3 Lineage is the point

Every run records the **exact** inputs that produced it: strategy version, policy version,
store version (file hashes), instrument spec version, modeling mode, and seed. Given a run id,
the system can state precisely what produced its numbers, and re-running it must reproduce
them byte for byte (§7).

Strategy versions form a tree, not a list. A version records its parent and its origin — which
agent session, which prompt, or hand-written. So "where did this come from" is always
answerable, and a promising candidate three weeks old can be branched from rather than
reconstructed from memory.

### 16.4 Rejected work stays in the library

Rev 3 had a `graveyard/` directory. That was the right instinct with the wrong storage: a
rejected candidate is not rubbish, it is **evidence**. Archived strategies stay in the
catalogue, fully queryable, with the reason they were archived.

Two things in this plan depend on it directly:

- **§9's failure taxonomy** — "the highest-value output of month one" — is a query over
  rejected generations. If they are deleted, or dumped somewhere unqueryable, it cannot be
  built at all.
- **§15's exemplar corpus** grows by promoting validated blocks. Knowing what has already
  failed, and how, is what keeps it from re-proposing the same dead ends.

Archiving hides something from the default view. Nothing deletes it. Deletion is an explicit,
confirmed, single-item action.

### 16.5 In the UI

- **Strategies** — list and version tree, filter by tag, symbol, status; diff two versions.
- **History** — every run, newest first, filterable by strategy, symbol, timeframe, modeling
  mode, verdict. Open any past run and its chart, trade list and report render exactly as they
  did on the day it ran, because the artifacts are still there.
- **Compare** — two or more runs side by side. The system knows their inputs differ and says
  *how*, rather than leaving the owner to remember.
- **Conversations** — past agent sessions, resumable via `--resume` (§15), each linked to the
  strategy versions it produced.
- **Data** — imported sources with their quality reports (§4.4), GMT offsets, and which runs
  used them.

### 16.6 Cost of getting this wrong

Persisting too little is silent and expensive: it surfaces weeks later as work redone, or as a
number nobody can trace. Persisting too much is cheap and obvious — artifacts are small
relative to the tick store, and a year of heavy use will not approach the size of one year of
ticks (§4.1).

So the default is **keep it**. If storage ever becomes a real constraint, prune artifacts for
archived strategies while keeping their catalogue rows — the metrics and lineage are what the
taxonomy needs, not the equity curves.

---

## 17. Policy file — the owner sets every value

**No threshold appears anywhere else as a literal.** The engine computes metrics and applies
whatever comparisons the policy names. Diff two policy versions to see what changed between
research runs.

```yaml
version: <str>

run:
  symbols: [...]
  broker: <str>                       # selects the instrument spec (§5)
  signal_timeframe: <M1|M5|M15|M30|H1|H4|D1|W1>
  modeling_mode: <real_ticks|m1_ohlc|open_prices>   # §6; must be supported by the source
  span: {start: <date>, end: <date>}
  account_currency: <str>
  initial_capital: <num>

data:                                 # per imported file — §4.2
  sources:
    - id: <str>                       # library reference (§16), not a path
      utc_offset_hours: <num>         # the clock the file is written in — §4.3
      tick_size: <num|null>           # null = read the decimals out of the file
      timeframe: <str|null>           # null = infer from spacing; ignored for ticks

execution:
  slippage_model: <fixed|spread_prop|gap_based>
  slippage_params: {...}
  sizing: {mode: <risk_pct|fixed_lots|notional>, value: <num>}
  enforce_stops_level: <bool>         # default true

optimization:
  mode: <grid|random|genetic|tpe|coordinate|nsga2>
  sweep_modeling_mode: open_prices    # §6 — cheap mode for the sweep
  revalidate_top_n: <int>             # re-run survivors in real_ticks
  objective: <...>
  budget: {evaluations: <int>, grid_values_per_param: <int>}
  parameters: {}                      # declared, owner-overridable per param

validation:
  enabled_modes: [...]
  folds: {train: <duration>, test: <duration>, step: <duration>, anchored: <bool>}
  embargo: <duration>
  monte_carlo: {paths: <int>, seed: <int>}
  holdout: {window: <duration>, reserved_from: <date>}
  gates:
    - {metric: <...>, op: <...>, value: <...>}

agent_runtime:
  provider: claude_code               # §14.7 — Codex deferred, no API path
  model: <str>
  bare_mode: <bool>
  permission_mode: <...>
  allowed_tools: [...]
  mcp_servers: [...]
  max_turns: <int>

integrity:
  strict_mode: <bool>
  candidates_per_family: <int>

ceilings:
  wall_clock: <duration>
  evaluations_total: <int>
  agent_max_turns: <int>              # enforceable; spend is not (§14.6)

generation:
  repair_retries: <int>

library:                              # §16
  artifact_root: <path>
  keep_rejected: true                 # §9's failure taxonomy is a query over these

reproducibility:
  seed: <int>
  numba_fastmath: false               # do not change; see §7
```

Note what is *gone* from rev 3's policy: `execution_resolution` (no 1s layer),
`fill_on_gapped_seconds` (a gap is always a gap), and `instrumentation.ambiguity_tolerance`
(real ticks have no ambiguity).

---

## 18. Repo layout

```
app/
  api/           FastAPI — run control, series feed, agent event stream
  ui/            Vela chart host, agent sidebar, results panel
  agent/         Claude Code session host, MCP config, attachments   [§14]
  quantor_mcp/   the QUANTOR engine MCP server — EXISTS              [§14.2]
engine/
  store/         tick CSV ingest, Parquet store, bar construction, quality checks
                 ingest.py exists — reads the owner's format, verifies the time unit
  instruments/   spec registry (§5), FX conversion pairs
  indicators/    numpy/numba ta.* + parity fixtures          [§8]
  signal/        static validator — EXISTS; harness + helpers pending [PASS 1]
  backtest/      bar-mode fill engine + metrics — EXISTS       [PASS 2 sweep]
  execution/     NautilusTrader adapter: intents -> orders,    [PASS 2 validate]
                 instrument spec -> Nautilus instrument, results back
  optimize/      param specs + objective guards — EXISTS; Optuna backend pending
  validate/      fold geometry — EXISTS; MC modes, PBO/DSR, verdicts pending
  library/       SQLite catalogue + content-addressed artifact store      [§16]
tests/
  golden/        indicator fixtures (oracle) + fast-path parity   [§8]
  store/         ingest: ticks, bars, detection, resolution rule  [§4]
  backtest/      hand-computed cases + ledger invariants          [§10.1]
  validate/      objective guards, fold geometry, leakage guards  [§11, §12]
tools/
  pinets_oracle/ dev-only fixture generator (AGPL, never shipped)
policy/          policy.yaml — ALL thresholds live here
runs/
  <id>/          artifacts, seeds, agent transcript, checkpoints
  <id>/verdict.json
  # NOTE: OOS artifacts live OUTSIDE this tree — see §13
library.db       catalogue: strategies, versions, runs, sessions          [§16]
artifacts/       content-addressed equity curves, trade lists, transcripts
strategies/      signal-block sources (the catalogue holds their lineage)
CLAUDE.md / AGENTS.md
.claude/skills/  signal-blocks, indicators, exemplars
.mcp.json
```

---

## 19. Probes

Rev 3 had four. Two are dead, one is measured, one survives. Two are new.

**~~Probe 1 — bar-count ceiling.~~ Dead.** PineTS is gone. For reference, the 5000-candle limit
never applied to custom arrays anyway (it is a provider `limit` concern); 4M bars ran fine
`[measured]`.

**~~Probe 3 — intent extraction.~~ Dead.** Intents are Python dataclasses.

**Probe 2 — throughput. Measured** (§4.1, §6, §11). Re-measure on the owner's real tick files:
synthetic ticks are uniformly distributed, and real ones cluster hard around news, which
changes bar-construction and fill-loop cache behaviour.

**~~Probe 4 — agent runtime.~~ Folded into Probe 9**, which tests the same things against the
MCP-first design (§14).

**Probe 5 — real tick CSV ingest. Partly done.** `engine/store/ingest.py` reads the owner's
format and `tests/store/` covers it. What still needs the owner's actual files, not a sample:

- **The same-ms cluster histogram.** If a meaningful fraction of ticks share a millisecond,
  fill ordering matters more than §4.4's source-order convention assumes, and may deserve a
  sub-ms tiebreak field if the vendor provides one.
- **The UTC offset**, via `infer_utc_offset_hours()` on a span containing several weekends,
  confirmed with the owner (§4.3).
- **Real tick density and its distribution.** The sample implies ~20 ticks/s, but real streams
  cluster hard around news; the peak-month partition size is what sizes the chunking.
- Spread distribution per session, and the Parquet round-trip being lossless.

**~~Probe 6 — MT5 reconciliation.~~ Dropped at the owner's request.** It is available on demand
It remains available as a manual cross-check (§10.2). The cost of dropping it is stated there
and should be read before trading a number this engine produced.

**Probe 8 — VectorBT cross-check (new).** Run the same simple strategy — percentage stops,
wide relative to bar range, no intrabar ambiguity — through our engine and through VectorBT,
and compare trade-by-trade (§10.4). Agreement is the independent reference; divergence means
one of them has a bug and finding out which is the point. Cheap: one test, not a subsystem.

**Probe 9 — the MCP surface. Partly done.**

**Built and verified** (`quantor_mcp/server.py`): the engine runs as an MCP server on the
current SDK (`mcp` 2.x, where `FastMCP` became `MCPServer`), exposing `data_list`, `data_load`,
`strategy_save`, `strategy_list`, `strategy_get`, `backtest_run`, `optimize_run`,
`validate_run` and `chart_apply`. Every tool was driven end to end against a 40,000-bar source:
data loaded with its quality report, a strategy saved and versioned, a backtest returning
metrics with the ledger reconciling, a grid sweep reporting its comparison count, walk-forward
returning per-fold results and efficiency, and `chart_apply` returning that run's own markers.

**What it found.** The first design executed signal blocks in a restricted namespace with a
trimmed `__builtins__`, and it failed immediately: `KeyError: '__import__'`, because numba's
dispatchers import at call time. That is not a detail to patch around — it is the evidence that
namespace restriction is not enforcement. The AST validator (§9) is, and it now runs at
`strategy_save`, before a block is ever stored.

**Still to do, and it needs a Claude Code session rather than a test harness:**

1. `--mcp-config` with LuxAlgo, Edge Stats and prop-firm-sim — confirm all three load, checking
   `system/init` for `mcp_server_errors` rather than assuming. **LuxAlgo's endpoint is
   unreachable from this environment** (egress-blocked), so its tool surface is unverified —
   run this locally.
2. **Does the agent actually use the tools, or work around them?** The question Probe 9 exists
   for. If it reaches for Bash, the tool design is wrong and better prompting will not fix it.
3. Confirm `--allowedTools` scoped to MCP plus `Read`/`Edit` blocks what it should.
4. Confirm `--resume` restores a conversation with its servers intact — §15.1's premise is that
   the conversation survives restarts.

Codex is out of scope (§14.7).

**Probe 7 — conversational strategy loop.** The centrepiece feature (§15.1) is a conversation
that produces and refines strategies over many turns, so probe the loop rather than a single
generation. In one session: brainstorm a concept against the Library and Edge Stats, have the
agent write a signal block, backtest it, view it on the chart, then ask for three successive
refinements.

What this measures:

1. Does each turn produce a **new catalogued version** with correct lineage (§16.3), rather
   than overwriting?
2. Does the agent still have the thread after a `--resume` across a restart — does it know
   what was already tried and rejected?
3. How much context does a refinement turn actually need — the full transcript, or a compact
   strategy state plus the last few turns? This sizes §15.2's context design and is the
   question most likely to change the UI.

---

## 20. Build order

**Discuss this ordering before following it** — see §0.

0. **Thin vertical slice. Done** — `bench/demo_loop.py` runs signal block → backtest →
   grid search → walk-forward end to end on synthetic bars, with `engine/backtest/`,
   `engine/optimize/` and `engine/validate/` behind it and 138 tests passing. What it still
   needs: real data through `engine/store/`, and the tick-mode fill engine (§6 `real_ticks`).
1. **Probe 5** (§19) on the owner's real files — density, duplicate timestamps, GMT offset.
   These settle §4.3 and size the store; everything downstream inherits them.
2. **Instrument registry** (§5) — small, everything needs it.
3. **Store** (§4). Nothing matters until the data is trustworthy. Ingest exists
   (`engine/store/ingest.py`) and reads ticks or bars; bar construction, the resolution rule
   (§4.6) at run setup, partitioning and the Parquet layer do not.
4. **Library skeleton** (§16) — catalogue schema and artifact store, *before* the first real
   run. It is a few tables, and retrofitting lineage onto runs that were never recorded is
   guesswork. Everything built after this point writes to it.
5. **Indicator layer + golden fixtures** (§8). Cheap, and the foundation for every signal block.
6. **Harness + signal-block contract.** The static validator exists
   (`engine/signal/validator.py`) and the engine MCP server exposes it at `strategy_save`
   (§14.2); what remains is the bar-loading and session helpers the blocks call.
7. **Tick-mode fills + the VectorBT cross-check** (§6, §10.4) — Probe 8. The bar engine and
   its §10.1 invariants already exist; this adds the `real_ticks` path on the same engine and
   the independent comparison that stands in for an external reference.
8. **Early agent-hypothesis test.** Before the UI, before the optimizer: with the slice from
   step 0, have the agent generate ~20 signal blocks and read the failure modes. §9 says the
   failure taxonomy is month one's highest-value output — this is where that becomes true. Rev 3
   put the agent last and so could not have it before month six.
9. **Optimization + validation** (§11, §12) — put Optuna behind the existing `ParamSpec` and
   `QuantStats` behind `metrics.py` (§3.1) rather than extending the hand-rolled versions.
10. **Chart + results panel + library UI** (§15, §16).
11. **The conversation** (§15.1) — Probe 7, then the chat itself. Last to build, because it is
    the easiest part and the most dangerous to trust, and because a chat that proposes
    strategies is only useful once the machinery that evaluates them is sound. It is the
    headline feature; it is still built last.

Steps 3–5 are largely independent and can proceed in parallel, using synthetic data for the
harness work until the store is real.

---

## 21. Open questions

- **pandas or polars** for ingest and reporting. Polars is faster and has better memory
  behaviour on wide tick data; pandas has broader ecosystem support. Pick one and stay there.
  numpy/numba do the hot work either way, so this is a convenience choice, not a performance one.
- **Does the owner have tick data for the whole span they care about?** If there are gaps,
  `m1_ohlc` mode (§6) covers them, but the report must mark which spans used which mode.
- **Conversation context strategy** (§15.2). Full transcript, rolling window, or a compact
  strategy state the agent re-reads each turn? Probe 7 should answer it.
- **Duplicate-timestamp density** on the owner's real files (Probe 5). May promote §4.4 from a
  convention to a real sub-millisecond ordering problem.
- **The files' GMT offset** (§4.3). Inferable from weekend gaps, but must be confirmed.
- **Library retention.** Keep-everything is the default (§16.6). Is there any category the
  owner wants pruned automatically, or is unbounded growth genuinely fine?
- **Which MQL5 order-execution style** the generated EA should use — market orders with
  attached SL/TP, or pending orders — and whether it should mirror the owner's existing EA's
  conventions so the two are comparable.
- **Does the owner's MQL5 strategy use bar-count indicators**, or purely levels and session
  windows? Determines whether §4.5's duration machinery is needed on day one at all.
- **Agent SDK package vs CLI subprocess** for the interactive sidebar. The Python SDK gives tool
  approval callbacks and native message objects, and there is no language boundary now.
- **Not yet designed at all:** live-forward (paper) testing, multi-strategy portfolios, regime
  detection, correlation between concurrent strategies, how the exemplar corpus is curated as it
  grows. Propose designs.

---

## 22. Execution boundary

**Output is parameters, not orders.** The owner has a working MQL5 execution path. Broker SDK
has no MT4/MT5 adapter and its write layer is experimental/sandbox-only. Do not rebuild a
working execution path, and do not connect this system to a broker.

What crosses the boundary is a validated strategy definition and its parameters, which the
owner implements on their own execution path. This is also why MT5 is the reference platform
(§0): the numbers this system produces are only useful if they predict what the owner's EA
actually does.

**The system emits Python and nothing else** (§14.3). MQL5 export, and later Pine Script
alongside Python, were both considered and cut. The reasoning is worth keeping because the idea
recurs:

Any second emission of "the same" strategy is a *claim*, not a translation. A deterministic
transpiler fails reproducibly; an LLM writing a second version fails **plausibly** — code that
reads correctly, runs, and computes something subtly different. Making that safe needs a parity
harness comparing the two bar-by-bar plus a generate-check-repair loop for when they disagree:
a substantial subsystem whose entire purpose is making another subsystem trustworthy.

One emission has none of that. It also made chart verification stronger rather than weaker
(§14.4) — the chart draws the engine's own trade list, so what you see is what was measured.

Revisit only if running strategies somewhere else becomes a goal, and only with the parity
harness built first.

Trade Journal reconciles live results against backtest. **The gap between them is the real
slippage model** — feed it back into §17's `execution.slippage_params` so the backtest stops
drifting from reality.
