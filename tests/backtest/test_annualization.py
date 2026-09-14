"""Annualization is derived, not tabulated. Plan §12.

It used to be a table of eight timeframes read with `.get(tf, 6_192.0)`. That
default is the bug: every timeframe not in the table — M2, M10, S1, S30 —
annualized as if it were H1. On S1 that is a Sharpe wrong by a factor of 60,
silently, and in whichever direction happens to flatter.
"""

from __future__ import annotations

import math

import pytest

from engine.backtest.metrics import BARS_PER_YEAR, bars_per_year
from engine.store import TIMEFRAMES

#: What the hand-written table said, before it was derived. Every one of these
#: must still come out identical — the fix corrects the gaps, not the entries.
LEGACY = {"M1": 371_520, "M5": 74_304, "M15": 24_768, "M30": 12_384,
          "H1": 6_192, "H4": 1_548, "D1": 258, "W1": 52}


@pytest.mark.parametrize("timeframe,expected", sorted(LEGACY.items()))
def test_the_original_table_values_are_unchanged(timeframe, expected):
    assert bars_per_year(timeframe) == pytest.approx(expected)


@pytest.mark.parametrize("timeframe", [tf for tf in TIMEFRAMES if tf != "TICK"])
def test_every_known_timeframe_can_be_annualized(timeframe):
    value = bars_per_year(timeframe)
    assert value > 0 and math.isfinite(value)
    assert BARS_PER_YEAR[timeframe] == value


def test_a_finer_timeframe_always_has_more_bars_per_year():
    order = ["S1", "S5", "S30", "M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    values = [bars_per_year(tf) for tf in order]
    assert values == sorted(values, reverse=True)


def test_seconds_scale_exactly_against_minutes():
    """S1 is sixty M1 bars; the annualization factor must say so."""
    assert bars_per_year("S1") == pytest.approx(bars_per_year("M1") * 60)
    assert bars_per_year("S30") == pytest.approx(bars_per_year("M30") * 60)


def test_an_unknown_timeframe_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="cannot annualize"):
        bars_per_year("M7")


def test_the_annualization_factor_is_the_square_root_of_bars_per_year():
    """Sharpe scales with sqrt(periods); a 60x finer bar is a 7.75x factor."""
    assert math.sqrt(bars_per_year("S1") / bars_per_year("M1")) == pytest.approx(
        math.sqrt(60))
