"""Probe 8 groundwork: does NautilusTrader do what engine/execution/ would have to?

Plan §3.1 — the build-vs-buy decision for tick-resolution execution. This probe
establishes that the *library* works; Probe 8 (§19) establishes that our mapping
onto it works, which is the part that can actually go wrong.

Needs Python >= 3.12 and a separate environment — NautilusTrader is not a
dependency of the sweep engine:

    python3.12 -m venv .nt && .nt/bin/pip install nautilus_trader
    .nt/bin/python bench/probe_nautilus.py

Measured on a 4-core box: 200,000 quote ticks in 1.26s (159k ticks/s), entry
filled at the ask, commission charged from the instrument fee model. Compare
with our numba sweep loop at 390M ticks/s — not like-for-like, but the ratio is
why §6 has two engines.
"""
import time
import pandas as pd, numpy as np

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig

N_TICKS = 200_000

class PingPong(Strategy):
    """Minimal: buy on every Kth tick, exit via bracket. Exercises fills, not alpha."""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.i = 0
        self.fills = 0
    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.subscribe_quote_ticks(self.config.instrument_id)
    def on_quote_tick(self, tick):
        self.i += 1
        if self.i % 20_000 == 0 and self.portfolio.is_flat(self.config.instrument_id):
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(10_000),
            )
            self.submit_order(order)
    def on_order_filled(self, event):
        self.fills += 1

class Cfg(StrategyConfig, frozen=True):
    instrument_id: str

def main():
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id="PROBE-001",
        logging=LoggingConfig(log_level="ERROR"),
    ))
    venue = Venue("SIM")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,          # <- margin account
        base_currency=USD,
        starting_balances=[Money(10_000, USD)],
    )
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    engine.add_instrument(instrument)

    # synthetic bid/ask quote ticks -> exactly the shape of the owner's CSV
    rng = np.random.default_rng(5)
    idx = pd.date_range("2021-01-04", periods=N_TICKS, freq="50ms", tz="UTC")
    bid = 1.2200 + np.cumsum(rng.normal(0, 2e-6, N_TICKS))
    df = pd.DataFrame({"bid_price": bid, "ask_price": bid + 8e-5,
                       "bid_size": 1e6, "ask_size": 1e6}, index=idx)
    ticks = QuoteTickDataWrangler(instrument).process(df)
    engine.add_data(ticks)

    strat = PingPong(config=Cfg(instrument_id=instrument.id))
    engine.add_strategy(strat)

    t0 = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - t0

    account = engine.trader.generate_account_report(venue)
    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()

    print(f"ran {N_TICKS:,} quote ticks in {elapsed:.2f}s "
          f"({N_TICKS/elapsed/1000:.0f}k ticks/s)")
    print(f"orders filled : {strat.fills}")
    print(f"positions     : {len(positions)}")
    if len(positions):
        cols = [c for c in ("entry","side","peak_qty","avg_px_open","avg_px_close",
                            "realized_pnl","commissions") if c in positions.columns]
        print(positions[cols].head(3).to_string())
    print(f"account rows  : {len(account)}")
    engine.dispose()

if __name__ == "__main__":
    main()
