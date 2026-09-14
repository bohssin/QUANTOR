# What the first real pass through QUANTOR found

A record of building one strategy end to end — hypothesis, rejection, revision,
validation — because the point of the library is that this is readable later
rather than remembered. Everything here is reproducible from the repo.

**The data is synthetic and nothing here says anything about gold.** The real
11 GB archive is on a Windows machine this build cannot reach. What is being
demonstrated is the pipeline; the numbers are about a generator I wrote.

---

## The source

`scripts/make_sample_data.py` produces a XAUUSD-shaped tick CSV with the same
shape as the owner's export: GMT+3 timestamps, `timestamp,bidPrice,askPrice`,
three decimals, weekend gaps, session-varying spread and tick rate.

```
7,353,471 ticks · 730 days · 295 MB · 2022-01-02 .. 2024-01-02
  -> 860,979 M1 bars (cached)  ->  59,264 M15  ->  14,816 H1
  streamed in ~24 s, GMT+3 applied at ingest
  spread p50 0.168  p99 0.471 · 0 out-of-order · 647 duplicate timestamps
```

Two structures are deliberately built into the generator so the optimizer has
something real to find: an Ornstein-Uhlenbeck pull toward a slow anchor
(strongest when the book is thin) and trending regimes on a slow Markov chain.

---

## v1 — the hypothesis, and why it was wrong

**Claim.** Gold ranges overnight. Between the New York close and the London
open there is little directional flow, so a push away from a short rolling mean
is more often inventory noise than information. Fade it, only in that window,
fixed reward-to-risk.

**Result: every one of 256 configurations lost money.**

```
default params, M15:  293 trades · net -5,690 · win 34.8% · PF 0.57 · expectancy -0.285 R
best of 256:          165 trades · net -2,978 · return/maxDD -0.881
```

The plateau report said `unscored — the best configuration is not profitable,
so there is no peak to be robust around`, which is the correct thing to say.

**Why.** Two measurements, both taken with the platform rather than guessed:

Lag-1 autocorrelation of bar returns — negative means mean reversion:

| timeframe | all | quiet 22-05 UTC | active |
|---|---|---|---|
| M1 | −0.0041 | −0.0029 | −0.0041 |
| M5 | −0.0219 | −0.0200 | −0.0221 |
| M15 | −0.0667 | −0.0413 | −0.0665 |
| M30 | −0.1249 | −0.0563 | −0.1238 |
| **H1** | **−0.2382** | −0.0532 | **−0.2383** |

The session premise is backwards here: reversion is *weaker* overnight and
strongest at H1 in the active session.

And the cost floor, at 1% risk of $10,000 on M15:

```
ATR median 0.822 · stop 1.316 -> 0.760 lots
round-trip cost $19.07 against $100 risked = 19.1% of R
```

A fixed 1.2 reward-to-risk needed a win rate near 50% to clear that. It got
34.8%.

---

## v2 — what the evidence said to build instead

Branched from v1 (the version tree keeps both). Three changes, each answering
one of the measurements:

1. **Drop the session premise.** The window stays a parameter so a sweep can
   re-test it; `from_hour == to_hour` means no filter.
2. **Run at H1**, where the reversion actually is.
3. **Exit into the mean, not at a fixed ratio.** The target is the distance
   back to the basis times a `capture` fraction, so reward-to-risk follows how
   stretched price was rather than a number chosen in advance. That is the
   shape mean reversion actually has: many small wins, rare larger losses.

```
H1, default params:  722 trades · win 96.8% · PF 22.7 · expectancy +0.563 R
                     Sharpe 11.6 · max drawdown 2.4% · largest loss -$4,075
```

**These numbers are absurd, and saying so is part of the result.** Net profit
reads $553,010 from $10,000 because sizing is fixed-fractional — profit
compounds, so the headline mostly measures how many trades there were.
Expectancy per trade (+0.563 R) is the sizing-invariant number and the one to
read.

---

## Is it real? The control test

A 96.8% win rate demands a look-ahead check before anything else. Re-run the
identical strategy on **shuffled bar-to-bar returns**: same distribution, same
volatility, same costs, same bar count, the order destroyed. Order is the only
thing any of these strategies can read.

| run | expectancy R | profit factor | trades |
|---|---|---|---|
| **real data** | **+0.563** | 22.67 | 722 |
| shuffled #1 | −0.055 | 0.87 | 1326 |
| shuffled #2 | −0.007 | 0.97 | 1367 |
| shuffled #3 | −0.040 | 0.91 | 1271 |
| shuffled #4 | −0.013 | 0.96 | 1309 |
| shuffled #5 | −0.033 | 0.92 | 1330 |

The edge collapses to roughly zero — slightly negative, which is exactly what
costs alone should produce. **There is no look-ahead.** The edge is entirely
the serial correlation in the data, which in this case I put there on purpose.

This test is now a button in the UI and an MCP tool (`control_test`). It cost
twenty lines and it is the first thing to run before believing any number.

---

## Walk-forward

A full 256-point search inside each train fold, each winner scored once
out-of-sample on bars the search never saw. H1, train 2,000 / test 500.

```
25 folds · 6,400 comparisons · 5.2 s
efficiency 0.285 · mean OOS score 17.9 · 25 of 25 folds profitable
verdict: consistent but decayed — every fold profitable out-of-sample,
         at a fraction of the in-sample size; expect the smaller number
```

Efficiency of 0.285 sounds bad and mostly is not. It is **biased low by
construction**: the in-sample number is the best of 256 configurations, the
out-of-sample number is that one configuration's single draw. Picking the max
of 256 noisy scores inflates the ratio's denominator. The share of profitable
folds is the un-inflated half of the answer, and it was 25 out of 25.

Writing this verdict function is what exposed the original one: it called this
"weak — most of the edge was in-sample" purely on the ratio, throwing away the
more trustworthy of the two measurements.

---

## What the agent did with the same tools

A live `claude -p` session against the MCP server, 12 turns, 72 seconds, **zero
tool errors**. It listed the library, backtested v2, ran the control test,
then designed and saved its own variant (`xau-dualtf-mr` — fade the fast-lookback
stretch only while a slower basis is still mild, a regime filter instead of a
session filter) and compared them:

| | v2 | agent's variant |
|---|---|---|
| trades | 722 | 46 |
| expectancy R | 0.563 | 0.498 |
| Sharpe | 11.6 | 2.7 |
| control passed | yes | yes |

Its own verdict: *"my variant is worse"* — the regime filter cut trade count by
over 90% mostly through lack of opportunity. It also flagged, unprompted, that
the headline numbers are "unusually high for a real trading edge and worth
treating with some skepticism about cost/fill assumptions".

Both strategies, all their versions and all 22 runs are in the same library the
UI reads.

---

## Bugs this pass found, each by something actually running

| Found by | Bug |
|---|---|
| MCP smoke test | Runs defaulted to the M1 **cache** timeframe instead of the one the source was loaded at — an M1 Sharpe reported for an M15 strategy, the same trades scaled by √15, silently and flatteringly |
| The v1 sweep | `plateau.robustness` inverted below zero: peak −0.91 with neighbours averaging −0.96 scored **1.06** and read as the flattest possible plateau |
| Playwright driving the UI | A float indicator period (which the UI's own generated grid produced) surfaced as a forty-line numba `TypingError` |
| Playwright driving the UI | The chart legend rendered a literal `null`; per-marker labels covered the candles they annotated |
| The memory benchmark | 64 MB read blocks made peak RSS grow 2.14× with file size — Arrow's CSV readahead, not our state. 4 MB with readahead off is flat **and** faster |

---

## What is still not true

- **The owner's real 11 GB archive has never been read.** It is a Windows path;
  this build runs on Linux. The streaming path is proven constant-memory on
  generated files up to 1.6 GB, which is the property that generalises, and the
  command to run against the real file is in the README.
- **`real_ticks` fills do not exist.** Every number above is bar-mode, which
  assumes the worse side when a bar touches both stop and target. Ambiguity was
  0.00–0.68% here, so it changed little; on a strategy with tight stops it
  would.
- **LuxAlgo `library_*` is unreachable from this sandbox** — no sign-in on this
  machine and the hosts are egress-blocked. Its client maps every 403 to "your
  plan does not include this", which is not what is happening. On a machine
  signed in with `npx -y @luxalgo/mcp login` these tools should work; that has
  not been verified from here.
- **No deflated Sharpe is computed yet.** The cumulative comparison count it
  needs is now recorded per family (560 for `session-mean-reversion` at the time
  of writing); the statistic itself is still to be wired in.
