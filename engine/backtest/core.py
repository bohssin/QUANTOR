"""Bar-mode backtest core. Plan §6 (`open_prices` / `m1_ohlc`), §7 pass 2.

Deliberately the *cheap* mode — the one an optimizer sweep runs thousands of
times. Entries fill at the next bar's open; exits are checked against each
subsequent bar's high/low. A tick-mode engine (§6 `real_ticks`) resolves the
same intents against the tick stream and is the mode results are reported in.

Two things this module is careful about, because both silently corrupt results:

**Stop-versus-target ambiguity is counted, not hidden.** When a bar's range
contains both the stop and the target, bar data cannot say which was touched
first. The engine takes the stop (the conservative reading) and **counts the
occurrence**. A run whose ambiguity rate is high is a run whose bar resolution
is too coarse for its stop distance, and the report has to say so rather than
quietly reporting the pessimistic number as fact. Tick sources do not have this
problem (§4.6) — which is exactly why they are the reporting mode.

**No same-bar entry and exit.** A position opened at bar N's open is not tested
against bar N's own high/low, because the intrabar path is unknown and testing
it manufactures fills that depend on an ordering the data never recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

__all__ = ["Bars", "Instrument", "Signals", "BacktestResult", "run_backtest"]

# exit_reason codes
EXIT_STOP = 1
EXIT_TARGET = 2
EXIT_SIGNAL = 3
EXIT_EOD = 4      # end of data, position force-closed


@dataclass(frozen=True)
class Bars:
    """OHLC bars at the signal timeframe. Prices are floats in quote currency."""

    ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray | None = None     # per-bar, in price units

    def __len__(self) -> int:
        return int(self.ms.shape[0])

    def slice(self, start: int, stop: int) -> "Bars":
        return Bars(
            ms=self.ms[start:stop], open=self.open[start:stop],
            high=self.high[start:stop], low=self.low[start:stop],
            close=self.close[start:stop],
            spread=None if self.spread is None else self.spread[start:stop],
        )


@dataclass(frozen=True)
class Instrument:
    """Sizing and cost parameters. Plan §5 — nothing here may be hardcoded."""

    contract_size: float = 100.0         # units per 1.00 lot
    tick_size: float = 0.001
    tick_value: float = 0.10             # quote ccy per tick_size per 1.00 lot
    lot_min: float = 0.01
    lot_step: float = 0.01
    lot_max: float = 100.0
    commission_per_lot_per_side: float = 3.5
    default_spread: float = 0.30         # used when bars carry no spread series
    stops_level: float = 0.0             # min SL/TP distance, price units

    def value_per_price_unit(self) -> float:
        """Account-currency P&L for a 1.0 price move on 1.00 lot."""
        return self.tick_value / self.tick_size


@dataclass(frozen=True)
class Signals:
    """What a signal block produces. Plan §9 — levels and arm conditions only.

    Arrays are per-bar and aligned to `Bars`. A signal on bar i is acted on at
    bar i+1's open: the block only ever sees closed bars, so there is no
    look-ahead by construction.
    """

    long_entry: np.ndarray       # bool
    short_entry: np.ndarray      # bool
    stop_distance: np.ndarray    # price units, > 0
    target_distance: np.ndarray  # price units, > 0


@dataclass
class BacktestResult:
    entry_i: np.ndarray
    exit_i: np.ndarray
    direction: np.ndarray        # +1 long, -1 short
    entry_px: np.ndarray
    exit_px: np.ndarray
    lots: np.ndarray
    pnl: np.ndarray              # account currency, net of costs
    r_multiple: np.ndarray       # pnl / risked amount — the sizing-free view
    exit_reason: np.ndarray
    equity: np.ndarray           # per bar
    ambiguous_exits: int
    rejected_by_stops_level: int
    rejected_zero_lots: int

    @property
    def n_trades(self) -> int:
        return int(self.pnl.shape[0])

    def ambiguity_rate(self) -> float:
        return self.ambiguous_exits / self.n_trades if self.n_trades else 0.0


@njit(cache=True, fastmath=False)
def _run(
    b_open, b_high, b_low, b_close, b_spread,
    long_entry, short_entry, stop_dist, target_dist,
    initial_capital, risk_pct,
    value_per_unit, lot_min, lot_step, lot_max,
    commission, stops_level,
):
    n = b_open.shape[0]
    max_trades = n

    entry_i = np.empty(max_trades, np.int64)
    exit_i = np.empty(max_trades, np.int64)
    direction = np.empty(max_trades, np.int64)
    entry_px = np.empty(max_trades, np.float64)
    exit_px = np.empty(max_trades, np.float64)
    lots_a = np.empty(max_trades, np.float64)
    pnl_a = np.empty(max_trades, np.float64)
    r_a = np.empty(max_trades, np.float64)
    reason = np.empty(max_trades, np.int64)
    equity = np.empty(n, np.float64)

    cash = initial_capital
    nt = 0
    ambiguous = 0
    rej_stops = 0
    rej_zero = 0

    pos = 0
    e_px = 0.0
    sl = 0.0
    tp = 0.0
    lots = 0.0
    risked = 0.0
    e_idx = 0

    for i in range(n):
        # --- exits: never on the entry bar itself ---
        if pos != 0 and i > e_idx:
            hit_sl = False
            hit_tp = False
            if pos > 0:
                hit_sl = b_low[i] <= sl
                hit_tp = b_high[i] >= tp
            else:
                hit_sl = b_high[i] >= sl
                hit_tp = b_low[i] <= tp

            if hit_sl and hit_tp:
                ambiguous += 1
                px = sl                     # conservative: assume the stop
                rsn = EXIT_STOP
            elif hit_sl:
                px = sl
                rsn = EXIT_STOP
            elif hit_tp:
                px = tp
                rsn = EXIT_TARGET
            else:
                px = 0.0
                rsn = 0

            if rsn != 0:
                gross = (px - e_px) * pos * value_per_unit * lots
                cash += gross - commission * lots        # exit side; entry was
                                                         # already charged
                net = gross - commission * lots * 2.0    # round-trip, so the
                                                         # trade list reconciles
                                                         # with the equity curve
                entry_i[nt] = e_idx
                exit_i[nt] = i
                direction[nt] = pos
                entry_px[nt] = e_px
                exit_px[nt] = px
                lots_a[nt] = lots
                pnl_a[nt] = net
                r_a[nt] = net / risked if risked > 0.0 else 0.0
                reason[nt] = rsn
                nt += 1
                pos = 0

        # --- entries: signal on bar i-1 fills at bar i's open ---
        if pos == 0 and i > 0:
            want = 0
            if long_entry[i - 1]:
                want = 1
            elif short_entry[i - 1]:
                want = -1

            if want != 0:
                sd = stop_dist[i - 1]
                td = target_dist[i - 1]
                if sd <= 0.0 or td <= 0.0:
                    want = 0
                elif sd < stops_level or td < stops_level:
                    rej_stops += 1
                    want = 0

            if want != 0:
                spread = b_spread[i]
                fill = b_open[i] + spread if want > 0 else b_open[i]
                sd = stop_dist[i - 1]
                td = target_dist[i - 1]

                risk_cash = cash * risk_pct
                per_lot_risk = sd * value_per_unit + commission * 2.0
                raw_lots = risk_cash / per_lot_risk if per_lot_risk > 0.0 else 0.0
                steps = np.floor(raw_lots / lot_step)
                sized = steps * lot_step
                if sized > lot_max:
                    sized = lot_max

                if sized < lot_min:
                    rej_zero += 1
                else:
                    pos = want
                    lots = sized
                    e_px = fill
                    e_idx = i
                    sl = fill - sd * want
                    tp = fill + td * want
                    risked = per_lot_risk * sized
                    cash -= commission * lots          # entry side

        # --- mark to market ---
        if pos == 0:
            equity[i] = cash
        else:
            mtm = (b_close[i] - e_px) * pos * value_per_unit * lots
            equity[i] = cash + mtm

    # force-close anything still open at the end of data
    if pos != 0:
        i = n - 1
        px = b_close[i]
        gross = (px - e_px) * pos * value_per_unit * lots
        cash += gross - commission * lots
        net = gross - commission * lots * 2.0
        entry_i[nt] = e_idx
        exit_i[nt] = i
        direction[nt] = pos
        entry_px[nt] = e_px
        exit_px[nt] = px
        lots_a[nt] = lots
        pnl_a[nt] = net
        r_a[nt] = net / risked if risked > 0.0 else 0.0
        reason[nt] = EXIT_EOD
        nt += 1
        equity[i] = cash

    return (entry_i[:nt], exit_i[:nt], direction[:nt], entry_px[:nt], exit_px[:nt],
            lots_a[:nt], pnl_a[:nt], r_a[:nt], reason[:nt], equity,
            ambiguous, rej_stops, rej_zero)


def run_backtest(
    bars: Bars,
    signals: Signals,
    instrument: Instrument | None = None,
    *,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,
) -> BacktestResult:
    """Run one backtest. Deterministic: same inputs give the same trade list."""
    inst = instrument or Instrument()
    n = len(bars)
    if not (len(signals.long_entry) == len(signals.short_entry) == n):
        raise ValueError("signal arrays must align with bars")

    spread = (bars.spread if bars.spread is not None
              else np.full(n, inst.default_spread, dtype=np.float64))

    out = _run(
        np.ascontiguousarray(bars.open, dtype=np.float64),
        np.ascontiguousarray(bars.high, dtype=np.float64),
        np.ascontiguousarray(bars.low, dtype=np.float64),
        np.ascontiguousarray(bars.close, dtype=np.float64),
        np.ascontiguousarray(spread, dtype=np.float64),
        np.ascontiguousarray(signals.long_entry, dtype=np.bool_),
        np.ascontiguousarray(signals.short_entry, dtype=np.bool_),
        np.ascontiguousarray(signals.stop_distance, dtype=np.float64),
        np.ascontiguousarray(signals.target_distance, dtype=np.float64),
        float(initial_capital), float(risk_pct),
        inst.value_per_price_unit(),
        inst.lot_min, inst.lot_step, inst.lot_max,
        inst.commission_per_lot_per_side, inst.stops_level,
    )
    return BacktestResult(
        entry_i=out[0], exit_i=out[1], direction=out[2], entry_px=out[3],
        exit_px=out[4], lots=out[5], pnl=out[6], r_multiple=out[7],
        exit_reason=out[8], equity=out[9],
        ambiguous_exits=int(out[10]), rejected_by_stops_level=int(out[11]),
        rejected_zero_lots=int(out[12]),
    )
