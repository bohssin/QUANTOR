# Local AI Trading Agent — Design Plan (rev 4)

**Status:** design handoff. Nothing built yet.

**Shape:** a local application. Chart on the left, AI agent sidebar on the right. The agent
writes strategy logic in Python, applies it to the chart, backtests it against the owner's own
tick data, optimizes and validates it, across any instrument.

**Reference platform: MetaTrader 5.** One reference, everywhere. The data model is MT5's
(tick history → bars at a chosen timeframe), the fidelity dial is MT5's (modeling modes), the
UI is MT5's (Strategy Tester layout), the ground truth for verification is the owner's own MT5
backtests, and the live execution path is the owner's existing MQL5 EA. Nothing in this system
targets TradingView.

**Engine:** Python — numpy + numba. No Pine Script transpiler, no 1-second intermediate bars.

**Agent runtime:** Claude Code or Codex CLI (primary). Raw API is a fallback, not the design
centre — see §14.

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
- **Question the sequencing.** §19's build order is a guess at dependency structure. If a
  different order gets to a working loop faster, propose it.

What is *not* up for renegotiation, because these are correctness properties rather than
preferences:

- **§5's instrument specs drive all sizing and P&L.** No hardcoded contract sizes, ever.
- **§4.3's server-timezone rule.** Bar boundaries are defined in broker server time. Get this
  wrong and every session- and daily-level signal is wrong, silently.
- **§4.4's tick ordering rule.** Identical-millisecond ticks must have a deterministic order,
  or results are not reproducible.
- **§8's golden-fixture rule.** No indicator ships without a frozen numerical fixture.
- **§10.4's engine invariants.** The fill engine is verified by property tests and MT5
  reconciliation, not by inspection.
- **§13's holdout-touched-once and comparison-count logging.**
- **§16's policy file: no threshold anywhere else in the codebase as a literal.**

Everything else is a proposal. Argue with it.

---

## 1. How to read this

- **§2** is what changed from rev 3 and why. Read it if you have seen the earlier plan;
  skip it if you have not.
- **§3–4** are the stack and the data layer. §4 is the foundation — nothing above it matters
  until the data is trustworthy.
- **§5–13** are design decisions.
- **§14–15** are the agent runtime and the application.
- **§16** is the policy file.
- **§17–21** are repo layout, probes, build order, open questions and the execution boundary.

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
  Rev 3 responded with a whole subsystem — §7.3's ambiguity instrumentation, a tolerance
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
discipline (§4.5), research integrity mode (§13), the policy file (§16), output-is-parameters
(§21), and §0's "argue with this" posture. Those were the good ideas and they survive intact.

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

| Component | Choice | Role |
|---|---|---|
| Arrays / math | **numpy** | all series computation |
| Hot loops | **numba** (`njit`) | bar construction, fill engine |
| Columnar store | **pyarrow** + Parquet | tick and bar persistence |
| Tabular / joins | **pandas** or **polars** | ingest, reporting (pick one, §20) |
| Optimization | **Optuna** | TPE, NSGA-II, grid, random |
| Property tests | **hypothesis** | engine invariants (§10.4) |
| API | **FastAPI** + uvicorn | serves the UI and run control |

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

**Measured, 10M ticks (~320 MB CSV):** `[measured]`

| Stage | Time | Note |
|---|---|---|
| `pandas.read_csv` | 3.7–5.2 s | one-time, per source file |
| Parquet write (zstd) | 1.5 s | 320 MB → **58 MB** |
| Parquet read | 1.2–1.4 s | the steady-state path |
| Ticks → M1 bars (numba) | 92 ms | 64,915 bars |
| Ticks → H1 bars (numba) | 51 ms | 1,082 bars |

In-memory cost is ~24 bytes/tick (int64 ms + 2× float64). 10M ticks = 240 MB, so one year of
a liquid instrument (~60M ticks) is ~1.4 GB resident and ~8 s to load from Parquet. Partition
by `symbol/year/month` and load only the run's span.

### 4.2 Ingest

The owner's CSVs are the universal data path, and tick CSV formats vary by vendor. Ingest is
**tolerant and explicit**: a per-source YAML mapping, never format sniffing.

```yaml
source: dukascopy_xauusd
file_glob: "raw/XAUUSD/*.csv"
columns:
  timestamp: {field: 0, format: "epoch_ms"}   # or strftime pattern
  bid:       {field: 1}
  ask:       {field: 2}
  bid_volume: {field: 3, optional: true}
timezone: UTC                # timezone the timestamps are IN
target_server_tz: "EET"      # broker server tz to convert to (§4.3)
```

Support at minimum: epoch ms/s, `YYYY-MM-DD HH:MM:SS.fff`, and MT5's exported tick format
(`<DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>`).

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
tick_size: 0.01
tick_value: 1.00            # quote currency, per 1.00 lot
digits: 2
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

| Mode | Signals evaluated | Fills resolved against | Cost (1 yr) | Use |
|---|---|---|---|---|
| **`real_ticks`** | bar close on signal TF | **real tick stream** | ~0.2 s `[measured]` | **System of record.** Every reported result |
| **`m1_ohlc`** | bar close on signal TF | M1 OHLC, assumed intrabar path | ~0.05 s | Spans with missing tick data |
| **`open_prices`** | bar open on signal TF | bar open | ~7 ms `[measured]` | **Optimizer sweeps** |

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
PASS 1 — SIGNAL                        PASS 2 — EXECUTION
numpy/numba on signal-TF bars     →    numba fill engine on real ticks
emits INTENTS with timestamps          resolves fills, P&L, accounting
```

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

### 8.1 Implementation

Vectorized numpy, with numba for the recursive families. A reference implementation of the
Wilder family already exists and is validated (see `engine/indicators/`):

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
- **Static validator** — an AST walk before execution, rejecting: imports, global/nonlocal
  writes, attribute access outside the allowed namespaces, and any negative-index or forward
  slice on `bars` (the lookahead check).
- **Signal blocks are unit-testable.** Call the function on a fixture and assert on its output.
  Rev 3 could only test through a CLI subprocess.

**Repair loop:** generate → validate → run on a small fixture → on failure the agent reads the
real Python traceback and retries, up to the owner's limit → on exhaustion, mark dead and log
the failure pattern. Tracebacks are considerably more useful than transpiler errors.

**The failure taxonomy is the highest-value output of month one** — it tells you what to put in
the skills. See §19, which is sequenced so this is actually possible in month one.

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

### 10.2 MT5 golden replay

The decisive test. Export the owner's existing MQL5 EA backtest from MT5 (report + full deal
list), replay the same strategy through this engine on the same tick data, and reconcile
**trade by trade**:

- entry and exit timestamps: exact
- entry and exit prices: exact
- per-trade P&L: within commission rounding
- final balance, max drawdown, trade count: within rounding

A divergence here is a bug in this engine, in the instrument spec, or in the server-timezone
handling — and you want to find out which before building anything on top. **This is the single
most important test in the project.**

### 10.3 Differential testing across modes

`real_ticks` vs `m1_ohlc` vs `open_prices` on the same strategy should differ in a bounded,
explainable direction (§6). Assert the direction; investigate anything outside it.

### 10.4 Reproducibility

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
survivor — it wins the backtest and blows up live. A drawdown-aware objective is the sane
default; the owner sets it in §16.

### Cost model

This is now cheap enough to change how you work. `[measured]`

| | per eval | 10,000 evals | × 10 folds |
|---|---|---|---|
| `open_prices` (sweep mode) | ~7 ms | ~70 s | ~12 min |
| `real_ticks` (validation) | ~0.2 s | ~35 min | — |

For comparison, rev 3's PineTS path was ~3.3 s/eval — 9.7 hours for that same 10k grid, four
days with folds. The whole §12 validation matrix went from arguably unaffordable to
interactive.

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

## 14. Agent runtime — Claude Code / Codex

**Primary: an agentic CLI running locally in the project directory. Not an API client.**

The agent has filesystem access, bash and native MCP, so much of what would be orchestration
code is the agent's own tool use:

| Not code | Instead |
|---|---|
| MCP client layer | Native MCP config — LuxAlgo, Edge Stats, prop-firm-sim plug straight in |
| Context-retrieval layer | `CLAUDE.md` / `AGENTS.md` + skills |
| Repair-loop orchestration | The agent runs pytest itself and reads the traceback |
| Result-parsing glue | The agent reads `runs/*.json` directly |

What stays as code: the store, the execution engine, the optimizer, the validation suite, the
policy gates. **The agent orchestrates; it does not compute.**

### Provider adapter

Normalize behind one interface so the runtime is swappable:

```
run_agent(prompt, *, cwd, mode, allowed_tools, mcp_config,
          schema=None, session=None) -> stream of events
```

**Claude Code** (primary):
- `claude -p "<prompt>"` non-interactive; exit 0 on success, non-zero on failure
- `--output-format text | json | stream-json`; `--include-partial-messages` + `--verbose` for
  token streaming into the sidebar
- `--json-schema '<schema>'` → result in `structured_output`. Use this to enforce structured
  output envelopes
- `--allowedTools "Bash(pytest *),Read,Edit"` using permission-rule syntax (note the space
  before `*` for prefix matching)
- `--permission-mode auto|dontAsk|acceptEdits`; `--permission-prompts none` for unattended runs
- `--mcp-config <file-or-json>`; check `system/init` for `mcp_servers` / `mcp_server_errors` to
  fail fast when a server doesn't load
- `--resume <session-id>` / `--continue`; transcripts are `.jsonl`
- `--bare` skips auto-discovery of hooks, skills, commands, subagents, plugins, MCP and
  `CLAUDE.md` — **use it for reproducible autonomous runs**, passing context explicitly via
  `--append-system-prompt-file`, `--settings`, `--mcp-config`. Bare mode needs
  `ANTHROPIC_API_KEY` and ignores subscription login
- `--output-format json` returns `total_cost_usd` plus a per-model breakdown → **this is how the
  spend ceiling in §16 is enforced.** Client-side estimate
- `--max-turns` caps work per invocation
- SIGTERM → exit 143, turn left unfinished; SIGINT ends the turn cleanly. Resume continues it
- Piped stdin capped at 10 MB — pass large data as file paths, never piped
- Python and TypeScript Agent SDK packages exist for full programmatic control. **Prefer the
  Python SDK for the interactive sidebar** — the engine is Python, so there is no longer a
  language boundary to cross — and the CLI for batch runs

**Codex CLI** (secondary):
- `codex exec "<prompt>"`; `--json` → JSONL event stream; `--output-schema <file.json>`
- `--sandbox read-only | workspace-write | danger-full-access`; `codex exec resume --last`
- `AGENTS.md` for project instructions; `.codex/config.toml` for MCP
- `--ignore-user-config` / `--ignore-rules` for reproducible runs
- **Known risk:** MCP tool calls in `codex exec` have been reported as auto-cancelled under
  non-interactive approval, with workarounds that disable sandboxing. Verify before relying on it

**Raw API** (tertiary): fallback only. Do not design around it.

### Project memory layout

```
CLAUDE.md / AGENTS.md     harness contract, §4.5 rule, forbidden calls, repo map
.claude/skills/
  signal-blocks/          slot spec, typed contract, worked examples
  indicators/             our indicator semantics + parity fixtures + gotchas
  exemplars/              validated signal blocks, tagged by concept family
.mcp.json                 luxalgo, edge-stats, prop-firm-sim
```

Skills work in `-p` mode — include `/skill-name` in the prompt. Start the exemplar corpus with
~10 hand-written blocks; append every candidate that validates and produces sane trade counts.

### Determinism under an agentic runtime

- Persist the session `.jsonl` transcript as a run artifact
- Record session ID, model, runtime version, and the resolved MCP server list from `system/init`
- Use `--bare` / `--ignore-user-config` for autonomous runs so a stray hook or personal MCP
  server on one machine can't change results
- Seed everything numeric in the engine (§7)
- Stamp every artifact with **policy, harness, store, instrument-spec and corpus versions**

**Accept that the agent layer is not bit-reproducible. The engine must be.** That is the line:
anything affecting a number is code and seeded; anything affecting a decision is the agent and
logged.

---

## 15. The application

**Left: chart.** Vela. Symbol and timeframe pickers, drawing tools, indicator menu. Strategies
render here — plotted levels, entry/exit markers, SL/TP lines, equity in a lower pane. The
Python backend computes series and pushes them to Vela as plain data over the API; Vela's Pine
plugin is not used.

**Right: agent sidebar.** Hosts an Agent SDK session (§14). Accepts text, **file attachments**
(existing `.pine`, MQL5 source, CSVs, PDFs) and **images** (chart screenshots, hand-drawn
setups). Attachments are written to a scratch dir and referenced by path — never piped, given
the 10 MB stdin cap.

**Bottom: results panel.** Tabbed, MT5-style — Report · Trade list · Equity · Optimization
results table (sortable) · Optimization surface (2D/3D) · Validation verdicts. Modeling mode and
validation mode shown on every tab.

**Agent modes**, explicit in the UI so the owner always knows whether a reply spends compute:
**Brainstorm** (no code, discusses ideas, pulls Library concepts) · **Build** (produces a signal
block, validates, applies to the chart at current symbol/TF) · **Analyze** (reads results,
subject to §13) · **Optimize** (configures and launches a run).

**Run lifecycle (MT5-like):** Configure → Start → progress with cancel → results. Checkpointed
and resumable. Every run is a SQLite row with policy version, artifacts, seeds and the agent
session transcript. **Cancel maps to SIGINT** (ends the turn cleanly) rather than SIGTERM.

**Shell:** local web — FastAPI + uvicorn serving a browser frontend. Resolves rev 3's
Electron/Tauri question, and with no AGPL in the stack there is no network-service concern.

**Ceilings, any one of which stops a run** (§16): wall clock · agent spend (from
`total_cost_usd`) · evaluations.

---

## 16. Policy file — the owner sets every value

**No threshold appears anywhere else as a literal.** The engine computes metrics and applies
whatever comparisons the policy names. Diff two policy versions to see what changed between
research runs.

```yaml
version: <str>

run:
  symbols: [...]
  broker: <str>                       # selects the instrument spec (§5)
  signal_timeframe: <M1|M5|M15|M30|H1|H4|D1|W1|MN>
  modeling_mode: <real_ticks|m1_ohlc|open_prices>   # §6
  span: {start: <date>, end: <date>}
  account_currency: <str>
  initial_capital: <num>

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
  provider: <claude_code|codex|api>
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
  agent_spend: <amount>
  evaluations_total: <int>

generation:
  repair_retries: <int>

reproducibility:
  seed: <int>
  numba_fastmath: false               # do not change; see §7
```

Note what is *gone* from rev 3's policy: `execution_resolution` (no 1s layer),
`fill_on_gapped_seconds` (a gap is always a gap), and `instrumentation.ambiguity_tolerance`
(real ticks have no ambiguity).

---

## 17. Repo layout

```
app/
  api/           FastAPI — run control, series feed, agent event stream
  ui/            Vela chart host, agent sidebar, results panel
  agent/         provider adapter (claude code / codex / api), attachments
engine/
  store/         tick CSV ingest, Parquet store, bar construction, quality checks
  instruments/   spec registry (§5), FX conversion pairs
  indicators/    numpy/numba ta.* + parity fixtures          [§8]
  signal/        harness, signal-block runner, static validator [PASS 1]
  execution/     fill engine, sizing, margin, swap, OCA       [PASS 2]
  optimize/      search modes, objectives, param specs
  validate/      fold geometry, MC modes, PBO/DSR, verdicts
tests/
  golden/        indicator fixtures (PineTS oracle + MT5)
  invariants/    hypothesis property tests for the fill engine [§10.1]
  mt5_replay/    the owner's MQL5 backtests + reconciliation   [§10.2]
tools/
  pinets_oracle/ dev-only fixture generator (AGPL, never shipped)
policy/          policy.yaml — ALL thresholds live here
runs/
  <id>/          artifacts, seeds, agent transcript, checkpoints
  <id>/verdict.json
  # NOTE: OOS artifacts live OUTSIDE this tree — see §13
strategies/      versioned signal blocks + result manifests
graveyard/       rejected candidates + reasons
CLAUDE.md / AGENTS.md
.claude/skills/  signal-blocks, indicators, exemplars
.mcp.json
```

---

## 18. Probes

Rev 3 had four. Two are dead, one is measured, one survives. Two are new.

**~~Probe 1 — bar-count ceiling.~~ Dead.** PineTS is gone. For reference, the 5000-candle limit
never applied to custom arrays anyway (it is a provider `limit` concern); 4M bars ran fine
`[measured]`.

**~~Probe 3 — intent extraction.~~ Dead.** Intents are Python dataclasses.

**Probe 2 — throughput. Measured** (§4.1, §6, §11). Re-measure on the owner's real tick files:
synthetic ticks are uniformly distributed, and real ones cluster hard around news, which
changes bar-construction and fill-loop cache behaviour.

**Probe 4 — agent runtime. Survives.** One `claude -p` round trip with `--mcp-config` pointing
at LuxAlgo MCP, `--json-schema` enforcing a trivial envelope, and `--allowedTools` scoped to
`Bash(pytest *)`. Confirm: schema enforcement works, MCP loads (check `system/init`),
`total_cost_usd` is present, and permission scoping actually blocks what it should. Repeat for
`codex exec --json --output-schema`, watching for the MCP-under-exec issue.

**Probe 5 — real tick CSV ingest (new).** Take the owner's actual tick files. Confirm: the
column mapping handles them, timestamps are monotonic after the stable sort, same-millisecond
cluster sizes are sane, spread distribution matches expectation per session, and the Parquet
round-trip is lossless. **Report the same-ms cluster histogram** — if a meaningful fraction of
ticks share a millisecond, fill ordering matters more than §4.4 assumes and may deserve a
sub-ms tiebreak field if the vendor provides one.

**Probe 6 — MT5 reconciliation (new, and the important one).** Before building anything on the
fill engine, take the owner's existing MQL5 EA, its MT5 backtest report and deal list, and
reconcile trade-by-trade per §10.2. Everything downstream depends on this matching.

---

## 19. Build order

**Discuss this ordering before following it** — see §0.

0. **Thin vertical slice, first.** One instrument, one month of ticks, one hardcoded strategy,
   `open_prices` mode, no optimizer, no agent, no UI — and get a trade list and an equity curve
   out. Days, not months, now that both passes are a few hundred lines of numpy/numba. Rev 3
   built five correct subsystems before anything ran end to end; do not repeat that.
1. **Probes 5 and 6** (§18). Real data, and MT5 reconciliation on a strategy the owner already
   understands. If the numbers disagree, find out why *now*.
2. **Instrument registry** (§5) — small, everything needs it.
3. **Store** (§4). Nothing matters until the data is trustworthy.
4. **Indicator layer + golden fixtures** (§8). Cheap, and it is the foundation for every signal
   block.
5. **Harness + signal-block contract + static validator** (§9).
6. **Execution engine** (§7, §10), with §10.1 invariants and §10.2 replay from the first commit,
   not bolted on afterwards.
7. **Early agent-hypothesis test.** Before the UI, before the optimizer: with the slice from
   step 0, have the agent generate ~20 signal blocks and read the failure modes. §9 says the
   failure taxonomy is month one's highest-value output — this is where that becomes true. Rev 3
   put the agent in step 8 and so could not have it before month six.
8. **Optimization + validation** (§11, §12).
9. **Chart + results panel** (§15).
10. **Agent sidebar + generation UI** (§14, §15) last. Easiest to build, most dangerous to
    trust, worth enabling only once everything downstream is sound.

Steps 3 and 4–5 are largely independent and can proceed in parallel, using synthetic ticks for
the harness work until the store is real.

---

## 20. Open questions

- **pandas or polars** for ingest and reporting. Polars is faster and has better memory
  behaviour on wide tick data; pandas has broader ecosystem support. Pick one and stay there.
  numpy/numba do the hot work either way, so this is a convenience choice, not a performance one.
- **Does the owner have tick data for the whole span they care about?** If there are gaps,
  `m1_ohlc` mode (§6) covers them, but the report must mark which spans used which mode.
- **Does the owner have TradingView scripts they want to run?** If yes, that path is now closed
  and we should discuss what it is worth. If no — and §21 suggests no — nothing is lost.
- **Same-millisecond tick density** on the owner's real files (Probe 5). May promote §4.4 from a
  convention to a real sub-ms ordering problem.
- **Does the owner's MQL5 strategy use bar-count indicators**, or purely levels and session
  windows? Determines whether §4.5's duration machinery is needed on day one at all.
- **Agent SDK package vs CLI subprocess** for the interactive sidebar. The Python SDK gives tool
  approval callbacks and native message objects, and there is no language boundary now.
- **Not yet designed at all:** live-forward (paper) testing, multi-strategy portfolios, regime
  detection, correlation between concurrent strategies, how the exemplar corpus is curated as it
  grows. Propose designs.

---

## 21. Execution boundary

**Output is parameters, not orders.** The owner has a working MQL5 execution path. Broker SDK
has no MT4/MT5 adapter and its write layer is experimental/sandbox-only. Do not rebuild a
working execution path.

This is also why MT5 is the reference platform throughout (§0). The numbers this system
produces are only useful if they predict what the owner's EA actually does. Every design choice
that trades TradingView fidelity for MT5 fidelity is therefore the correct trade.

Trade Journal reconciles live against backtest. **The gap between them is the real slippage
model** — feed it back into §16's `execution.slippage_params` so the backtest stops drifting
from reality.
