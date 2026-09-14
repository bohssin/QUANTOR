"""Performance metrics over a backtest result. Plan §11 objectives, §12 reporting.

Every metric is computed from the trade list and the per-bar equity curve, and
none of them decides anything on its own — §17's policy names the comparisons.

Two choices worth stating, because both are places implementations quietly
differ:

**Sharpe is annualized from the per-bar equity curve using the bar timeframe**,
not from trade returns. Trade-level Sharpe depends on trade frequency in a way
that makes two strategies on different timeframes incomparable.

**Expectancy is in R**, not currency. R is P&L divided by the amount risked at
entry, so it is invariant to account size and position sizing — which is what
makes it comparable across runs and the right input to §12's Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from engine.store.ingest import TIMEFRAMES

__all__ = ["Metrics", "compute_metrics", "BARS_PER_YEAR", "bars_per_year"]

#: Trading hours in a year for FX/metals: 258 sessions of 24 hours.
TRADING_MS_PER_YEAR = 258 * 24 * 3_600_000

#: Bars per year, by timeframe — **derived**, not tabulated.
#:
#: It used to be a table of eight entries read with `.get(tf, 6_192.0)`, and
#: that default is the landmine: every timeframe not in the table — M2, M10,
#: S1, S30 — silently annualized as if it were H1. On S1 data that is a Sharpe
#: wrong by a factor of 60, in whichever direction flatters. Deriving it from
#: the timeframe's own length cannot be wrong for a timeframe nobody thought of,
#: and `compute_metrics` now raises on a name it does not know rather than
#: guessing.
def bars_per_year(timeframe: str) -> float:
    tf = timeframe.strip().upper()
    if tf == "W1":
        return 52.0            # a weekly bar spans the weekend; length lies
    ms = TIMEFRAMES.get(tf)
    if not ms:
        raise ValueError(
            f"cannot annualize metrics for timeframe {timeframe!r}; known: "
            + ", ".join(k for k in TIMEFRAMES if k != "TICK")
        )
    return TRADING_MS_PER_YEAR / ms


#: Kept for callers that want the classic table; every value is derived.
BARS_PER_YEAR: dict[str, float] = {
    tf: bars_per_year(tf) for tf in TIMEFRAMES if tf != "TICK"
}


@dataclass(frozen=True)
class Metrics:
    n_trades: int
    net_profit: float
    profit_factor: float
    expectancy_r: float
    win_rate: float
    max_drawdown: float          # account currency
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    sqn: float
    avg_r: float
    std_r: float
    largest_loss: float
    max_consecutive_losses: int
    exposure: float              # fraction of bars holding a position

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(result, timeframe: str = "M15",
                    initial_capital: float = 10_000.0) -> Metrics:
    pnl = result.pnl
    r = result.r_multiple
    equity = result.equity
    n = int(pnl.shape[0])

    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0, 0.0)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(dd.max()) if dd.size else 0.0
    max_dd_pct = float((dd / np.maximum(peak, 1e-9)).max()) if dd.size else 0.0

    # Per-bar returns off the equity curve, annualized by the bar timeframe.
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    per_year = bars_per_year(timeframe)
    ann = math.sqrt(per_year)
    mean_ret = float(rets.mean()) if rets.size else 0.0
    std_ret = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    downside = rets[rets < 0]
    down_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0

    sharpe = (mean_ret / std_ret * ann) if std_ret > 0 else 0.0
    sortino = (mean_ret / down_std * ann) if down_std > 0 else 0.0

    net = float(pnl.sum())
    years = max(len(equity) / per_year, 1e-9)
    cagr = net / initial_capital / years
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    sqn = (mean_r / std_r * math.sqrt(n)) if std_r > 0 else 0.0

    held = int((result.exit_i - result.entry_i).sum())

    return Metrics(
        n_trades=n,
        net_profit=net,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else math.inf,
        expectancy_r=mean_r,
        win_rate=float((pnl > 0).mean()),
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        sqn=sqn,
        avg_r=mean_r,
        std_r=std_r,
        largest_loss=float(losses.min()) if losses.size else 0.0,
        max_consecutive_losses=_max_run(pnl < 0),
        exposure=held / len(equity) if len(equity) else 0.0,
    )


def _max_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best
