"""Session-filtered mean reversion on XAUUSD. A §9 signal block.

**The idea.** Gold ranges when the book is thin. Overnight, between the New York
close and the London open, there is little directional flow, so a push away from
a short rolling mean is more often inventory noise than information — and it
tends to come back. In the London and New York sessions the same push is as
likely to be the start of a move, so the filter is not cosmetic: it is most of
the strategy.

**The rules.** Measure how far price sits from a rolling mean in units of ATR,
so the threshold means the same thing in a quiet week and a violent one. When
that stretch exceeds `entry_z` *and* the clock is inside the quiet window, take
the other side. Stop at `stop_mult` ATR, target `rr` times the stop.

**What is parameterised and why.** Everything with a number in it, because the
right lookback for gold is not something to assert. The search picks them per
fold and walk-forward decides whether the choice generalised — which is the
whole point of not hand-tuning it here.

Hours are UTC. The ingest converts the file's clock to UTC (a GMT+3 export is
shifted at load), so the window is stated in UTC and stays correct no matter
which vendor's file it runs on.
"""

SIGNAL_BLOCK = '''
def signal(bars, p):
    n = len(bars.close)

    basis = sma_fast(bars.close, p["lookback"])
    rng = atr_fast(bars.high, bars.low, bars.close, p["atr"])

    # ATR is the yardstick, so a zero or missing one makes the stretch
    # meaningless rather than large. Mark it and drop those bars.
    bad = np.isnan(rng) | (rng <= 0) | np.isnan(basis)
    safe = np.where(bad, 1.0, rng)
    stretch = (bars.close - basis) / safe

    # Hour of day in UTC. Bars carry epoch milliseconds; the ingest already
    # moved the source clock onto UTC, so this is the real hour everywhere.
    hour = (bars.ms // 3600000) % 24
    start = p["from_hour"]
    end = p["to_hour"]
    if start == end:
        window = np.ones(n, np.bool_)               # no session filter at all
    elif start > end:
        window = (hour >= start) | (hour < end)     # the window crosses midnight
    else:
        window = (hour >= start) & (hour < end)

    z = p["entry_z"]
    long_entry = (stretch < -z) & window & ~bad
    short_entry = (stretch > z) & window & ~bad

    stop = safe * p["stop_mult"]
    # Mean reversion aims at the mean, not at a fixed multiple of the stop. The
    # target is the distance back to the basis, scaled by how much of the
    # reversion to try to keep — which makes the reward-to-risk a consequence of
    # how stretched price was, rather than a number chosen in advance.
    target = np.abs(bars.close - basis) * p["capture"]
    floor = safe * p["min_target"]
    target = np.where(target < floor, floor, target)

    return {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "stop_distance": stop,
        "target_distance": target,
    }
'''

DEFAULT_PARAMS = {
    "lookback": 24,      # bars in the rolling mean
    "atr": 24,           # ATR period, the yardstick for "stretched"
    "entry_z": 1.5,      # ATRs away from the mean before taking the other side
    "stop_mult": 2.0,    # stop distance in ATRs
    "capture": 0.7,      # fraction of the distance back to the mean to aim for
    "min_target": 0.5,   # never aim closer than this many ATRs
    "from_hour": 0,      # from == to means "every hour" — v1's session filter
    "to_hour": 0,        # was tested and did not survive (see the module notes)
}

#: What the optimizer is allowed to move, and how far. Deliberately coarse: a
#: fine grid over seven parameters is how a sweep finds noise (§11).
SEARCH_GRID = {
    "lookback": [12, 48, 12],
    "entry_z": [1.0, 2.5, 0.5],
    "stop_mult": [1.5, 3.0, 0.5],
    "capture": [0.4, 1.0, 0.2],
}

DESCRIPTION = (
    "Fade ATR-normalised stretch from a rolling mean and take profit on the way "
    "back to it, so reward-to-risk follows how stretched price was rather than a "
    "number picked in advance."
)

#: What v1 claimed, and what the data said. Kept because a rejected hypothesis
#: is a result — §16.4 — and because the next person to have this idea should
#: find the measurement rather than repeat the sweep.
#:
#:   v1: fade stretch ONLY in the quiet overnight window (22:00-05:00 UTC),
#:       fixed reward-to-risk. Every one of 256 configurations lost money.
#:
#:   Measured lag-1 autocorrelation of bar returns on the same source:
#:
#:       tf    all       quiet 22-05   active
#:       M1   -0.0041     -0.0029     -0.0041
#:       M5   -0.0219     -0.0200     -0.0221
#:       M15  -0.0667     -0.0413     -0.0665
#:       M30  -0.1249     -0.0563     -0.1238
#:       H1   -0.2382     -0.0532     -0.2383
#:
#:   The session premise is backwards on this data: reversion is *weaker*
#:   overnight and strongest at H1 in the active session. Round-trip costs are
#:   19.1% of a 1%-risk R at M15, so a fixed 1.2 reward-to-risk needed a win
#:   rate near 50% and got 34.8%.
#:
#: v2 therefore drops the session premise (the window stays a parameter so the
#: sweep can re-test it), moves to H1, and exits into the mean.
V1_FINDING = "session filter rejected; reversion is strongest at H1, not overnight"
