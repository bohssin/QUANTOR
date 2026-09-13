"""Indicator layer. Plan §8 — no indicator ships without a golden fixture.

Two implementations of the same recurrences:

- `wilder` — readable numpy reference, pinned to the PineTS oracle by
  `tests/golden/`. Use it when clarity matters.
- `fast` — numba, for the optimizer's inner loop. Pinned to `wilder` by
  `tests/golden/test_fast_parity.py`, so it inherits the oracle guarantee.
"""
from .fast import atr_fast, ema_fast, rma_fast, rsi_fast, sma_fast, true_range_fast
from .wilder import atr, ema, rma, rsi, sma, true_range

__all__ = [
    "sma", "ema", "rma", "rsi", "true_range", "atr",
    "sma_fast", "ema_fast", "rma_fast", "rsi_fast", "true_range_fast", "atr_fast",
]
