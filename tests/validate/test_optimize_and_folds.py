"""Optimizer and validation correctness. Plan §11, §12.

The bugs worth catching here are not crashes — they are results that look fine
and are not. Each test names the specific way a search or a fold layout lies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from engine.optimize import (
    FloatParam,
    IntParam,
    ParamSpec,
    grid_search,
    objective_value,
    random_search,
)
from engine.validate import (
    anchored_folds,
    attribute_trades,
    purge_bars_for,
    purged_kfold,
    rolling_folds,
    train_mask,
    walk_forward,
)


@dataclass
class FakeMetrics:
    n_trades: int = 100
    net_profit: float = 0.0
    profit_factor: float = 1.0
    expectancy_r: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    sqn: float = 0.0
    max_drawdown: float = 100.0

    def as_dict(self):
        return {"n_trades": self.n_trades, "net_profit": self.net_profit,
                "max_drawdown": self.max_drawdown}


# --- objectives -------------------------------------------------------------

def test_default_objective_is_drawdown_aware():
    """Raw return picks the highest-variance survivor. A 2x profit bought with
    10x the drawdown must not win. Plan §11."""
    steady = FakeMetrics(net_profit=1_000.0, max_drawdown=200.0)   # 5.0
    wild = FakeMetrics(net_profit=2_000.0, max_drawdown=2_000.0)   # 1.0
    assert objective_value(wild, "net_profit") > objective_value(steady, "net_profit")
    assert objective_value(steady) > objective_value(wild)


def test_too_few_trades_is_disqualifying_not_a_high_score():
    """Two lucky trades must not outrank four hundred — the commonest way an
    optimizer produces a nonsense winner."""
    lucky = FakeMetrics(n_trades=2, net_profit=5_000.0, max_drawdown=10.0)
    real = FakeMetrics(n_trades=400, net_profit=1_000.0, max_drawdown=200.0)
    assert objective_value(lucky, min_trades=30) == -math.inf
    assert objective_value(real, min_trades=30) > -math.inf


def test_non_finite_scores_sort_last_rather_than_winning():
    """profit_factor is inf when there are no losses. Left unguarded that is
    the global maximum, so a 3-trade fluke wins every sweep."""
    flawless = FakeMetrics(n_trades=3, profit_factor=math.inf)
    assert objective_value(flawless, "profit_factor") == -math.inf


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        objective_value(FakeMetrics(), "sharpe_ratio_v2")


# --- search modes -----------------------------------------------------------

def _spec():
    return ParamSpec(params={"len": IntParam(10, 50, 10),
                             "mult": FloatParam(1.0, 3.0, 1.0)})


def test_grid_is_a_full_factorial_and_counts_every_comparison():
    spec = _spec()
    seen = []

    def ev(p):
        seen.append(p)
        return FakeMetrics(net_profit=float(p["len"]), max_drawdown=100.0)

    res = grid_search(spec, ev, max_values_per_param=10)
    assert len(seen) == 5 * 3                      # len 10..50 step 10, mult 1..3
    assert res.comparisons == len(seen)
    assert res.best.params["len"] == 50


def test_comparison_count_is_what_dsr_consumes():
    """§12's deflated Sharpe adjusts for trials. Under-reporting inflates it,
    so the count must equal evaluations actually performed."""
    res = random_search(_spec(), lambda p: FakeMetrics(), n_evaluations=37, seed=1)
    assert res.comparisons == 37 == len(res.evaluations)


def test_random_search_is_reproducible_under_seed():
    a = random_search(_spec(), lambda p: FakeMetrics(net_profit=p["len"]),
                      n_evaluations=25, seed=7)
    b = random_search(_spec(), lambda p: FakeMetrics(net_profit=p["len"]),
                      n_evaluations=25, seed=7)
    assert [e.params for e in a.evaluations] == [e.params for e in b.evaluations]
    c = random_search(_spec(), lambda p: FakeMetrics(net_profit=p["len"]),
                      n_evaluations=25, seed=8)
    assert [e.params for e in a.evaluations] != [e.params for e in c.evaluations]


def test_sampled_values_respect_bounds_and_step():
    spec = ParamSpec(params={"n": IntParam(4, 20, 4), "x": FloatParam(0.5, 2.5, 0.5)})
    res = random_search(spec, lambda p: FakeMetrics(), n_evaluations=200, seed=3)
    for e in res.evaluations:
        assert 4 <= e.params["n"] <= 20 and e.params["n"] % 4 == 0
        assert 0.5 <= e.params["x"] <= 2.5
        assert abs(round(e.params["x"] / 0.5) - e.params["x"] / 0.5) < 1e-9


def test_top_k_surfaces_the_neighbourhood_not_just_the_peak():
    """A sharp optimum is usually a fitting artifact, so the plateau around it
    is what should be inspected. Plan §11."""
    res = grid_search(_spec(), lambda p: FakeMetrics(net_profit=float(p["len"]),
                                                     max_drawdown=100.0),
                      max_values_per_param=10)
    top = res.top_k(5)
    assert len(top) == 5
    assert [e.score for e in top] == sorted([e.score for e in top], reverse=True)


def test_fixed_params_are_passed_through_but_not_swept():
    spec = ParamSpec(params={"len": IntParam(10, 30, 10)}, fixed={"session": "london"})
    res = grid_search(spec, lambda p: FakeMetrics(), max_values_per_param=10)
    assert len(res.evaluations) == 3
    assert all(e.params["session"] == "london" for e in res.evaluations)


# --- fold geometry ----------------------------------------------------------

def test_rolling_folds_never_let_train_touch_test():
    folds = rolling_folds(n_bars=1000, train=300, test=100, step=100)
    assert folds
    for f in folds:
        assert f.train_end <= f.test_start
        assert f.train_len == 300 and f.test_len == 100
        assert f.test_end <= 1000


def test_embargo_opens_a_real_gap():
    folds = rolling_folds(n_bars=1000, train=300, test=100, step=100, embargo=25)
    for f in folds:
        assert f.test_start - f.train_end == 25


def test_anchored_folds_grow_from_a_fixed_start():
    folds = anchored_folds(n_bars=1000, train=300, test=100, step=100)
    assert all(f.train_start == 0 for f in folds)
    lengths = [f.train_len for f in folds]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_purge_width_comes_from_trade_duration_not_a_guess():
    """Purging narrower than the label span is not purging. In a backtest the
    label span is how long a trade is open. Plan §12."""
    entry = np.array([0, 10, 20, 30, 40])
    exit_ = np.array([40, 51, 25, 33, 44])        # durations 40, 41, 5, 3, 4
    assert purge_bars_for(entry, exit_) >= 40


def test_purged_kfold_train_mask_excludes_the_purge_and_embargo_window():
    folds = purged_kfold(n_bars=1000, k=5, purge=30)
    f = folds[2]
    mask = train_mask(1000, f, purge=30, embargo=10)
    assert not mask[f.test_start:f.test_end].any()          # test excluded
    assert not mask[f.test_start - 30:f.test_start].any()    # purge excluded
    assert not mask[f.test_end:f.test_end + 10].any()        # embargo excluded
    assert mask[:f.test_start - 30].all()                    # the rest is usable


def test_trades_are_attributed_by_entry_bar():
    """A trade straddling the boundary belongs to the window where the decision
    was made. Attributing by exit lets in-sample information score
    out-of-sample."""
    entry = np.array([5, 95, 150, 290])
    idx = attribute_trades(entry, 100, 200)
    assert idx.tolist() == [2]           # only the trade entered at 150


def test_no_trade_is_counted_in_two_folds():
    entry = np.array([5, 95, 150, 290, 310])
    folds = [(0, 100), (100, 200), (200, 400)]
    counted = sorted(
        i for lo, hi in folds for i in attribute_trades(entry, lo, hi).tolist()
    )
    assert counted == sorted(set(counted)) == [0, 1, 2, 3, 4]


# --- walk-forward -----------------------------------------------------------

def test_walk_forward_search_never_receives_test_indices():
    """The leakage guard that matters. If `search` could see the test window it
    would be an expensive way to overfit (§13), so the signature makes it
    impossible rather than merely discouraged."""
    folds = rolling_folds(n_bars=1000, train=300, test=100, step=100)
    seen_spans = []

    def search(idx, tr_start, tr_end):
        seen_spans.append((tr_start, tr_end))
        return {"len": 20}, 1.5, 12

    def evaluate_test(params, te_start, te_end):
        return 1.2, {"n_trades": 30}

    res = walk_forward(folds, search, evaluate_test)

    for (tr_start, tr_end), fold in zip(seen_spans, folds):
        assert (tr_start, tr_end) == (fold.train_start, fold.train_end)
        assert tr_end <= fold.test_start
    assert res.total_comparisons == 12 * len(folds)


def test_walk_forward_evaluates_each_winner_once():
    folds = rolling_folds(n_bars=1000, train=300, test=100, step=100)
    calls = []

    def search(idx, a, b):
        return {"len": 10 * idx}, 2.0, 5

    def evaluate_test(params, a, b):
        calls.append((params["len"], a, b))
        return 1.0, {}

    walk_forward(folds, search, evaluate_test)
    assert len(calls) == len(folds)
    assert len({(a, b) for _, a, b in calls}) == len(folds)   # distinct windows


def test_efficiency_flags_a_fitted_search():
    """Out-of-sample far below in-sample is the signature of a fitted train
    fold. Near 1.0 means the optimization generalized."""
    folds = rolling_folds(n_bars=800, train=300, test=100, step=100)
    fitted = walk_forward(folds, lambda i, a, b: ({}, 10.0, 1),
                          lambda p, a, b: (1.0, {}))
    honest = walk_forward(folds, lambda i, a, b: ({}, 2.0, 1),
                          lambda p, a, b: (1.9, {}))
    assert fitted.efficiency() < 0.2
    assert honest.efficiency() > 0.9


def test_total_comparisons_accumulates_across_folds():
    """Walk-forward nests a full search per fold, so the trial count DSR needs
    is folds x evaluations — not one fold's worth."""
    folds = rolling_folds(n_bars=1000, train=300, test=100, step=100)
    res = walk_forward(folds, lambda i, a, b: ({}, 1.0, 250),
                       lambda p, a, b: (1.0, {}))
    assert res.total_comparisons == 250 * len(folds)
