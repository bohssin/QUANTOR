"""Plateau analysis. Plan §11.

Cross-symbol survival has been dropped as a robustness filter, so walk-forward,
the deflated Sharpe, and this carry the load. A sharp optimum surrounded by bad
neighbours is a hole in the noise, not an edge.
"""

from __future__ import annotations

import math

import pytest

from engine.optimize import Evaluation, analyze_plateau, rank_by_robustness


def ev(score, **params):
    return Evaluation(params=params, score=score, metrics={})


def grid(scores: dict):
    """scores maps (a, b) -> score."""
    return [ev(s, a=a, b=b) for (a, b), s in scores.items()]


def test_a_flat_plateau_scores_near_one():
    evals = grid({(a, b): 10.0 for a in (1, 2, 3) for b in (1, 2, 3)})
    r = analyze_plateau(evals, {"a": 2, "b": 2})
    assert r.n_neighbours == 4
    assert r.robustness == pytest.approx(1.0)
    assert r.is_plateau
    assert "plateau" in r.verdict()


def test_an_isolated_spike_is_caught():
    """The case the whole module exists for: one configuration scores 10, every
    neighbour scores 1. That is noise, not edge."""
    scores = {(a, b): 1.0 for a in (1, 2, 3) for b in (1, 2, 3)}
    scores[(2, 2)] = 10.0
    r = analyze_plateau(grid(scores), {"a": 2, "b": 2})
    assert r.score == 10.0
    assert r.robustness == pytest.approx(0.1)
    assert not r.is_plateau
    assert "spike" in r.verdict()


def test_a_slope_is_distinguished_from_both():
    scores = {(a, b): 1.0 for a in (1, 2, 3) for b in (1, 2, 3)}
    scores[(2, 2)] = 10.0
    for k in ((1, 2), (3, 2), (2, 1), (2, 3)):
        scores[k] = 5.0
    r = analyze_plateau(grid(scores), {"a": 2, "b": 2})
    assert r.robustness == pytest.approx(0.5)
    assert not r.is_plateau
    assert "slope" in r.verdict()


def test_neighbours_are_grid_adjacent_only():
    """Diagonals are not neighbours — one step per parameter."""
    evals = grid({(a, b): 1.0 for a in (1, 2, 3) for b in (1, 2, 3)})
    assert analyze_plateau(evals, {"a": 2, "b": 2}).n_neighbours == 4   # centre
    assert analyze_plateau(evals, {"a": 1, "b": 1}).n_neighbours == 2   # corner
    assert analyze_plateau(evals, {"a": 1, "b": 2}).n_neighbours == 3   # edge


def test_fixed_parameters_are_not_treated_as_axes():
    evals = [ev(1.0, a=a, session="london") for a in (1, 2, 3)]
    r = analyze_plateau(evals, {"a": 2, "session": "london"})
    assert r.n_neighbours == 2


def test_rank_by_robustness_prefers_a_plateau_over_a_higher_spike():
    """The point of the exercise: the top-scoring configuration is often not the
    one to take."""
    scores = {(a, b): 0.1 for a in range(1, 6) for b in range(1, 6)}
    scores[(1, 1)] = 9.0                                  # isolated spike
    for k in ((3, 3), (2, 3), (4, 3), (3, 2), (3, 4)):
        scores[k] = 5.0                                   # genuine plateau
    reports = rank_by_robustness(grid(scores), top_k=6)
    assert reports[0].params == {"a": 3, "b": 3}
    assert reports[0].score == 5.0
    spike = next(r for r in reports if r.params == {"a": 1, "b": 1})
    assert spike.score == 9.0 and not spike.is_plateau


def test_non_finite_scores_do_not_poison_the_neighbourhood():
    scores = {(a, b): 1.0 for a in (1, 2, 3) for b in (1, 2, 3)}
    scores[(2, 2)] = 2.0
    evals = grid(scores)
    evals.append(ev(-math.inf, a=1, b=2))
    r = analyze_plateau(evals, {"a": 2, "b": 2})
    assert math.isfinite(r.robustness)


def test_a_zero_score_peak_is_unscored_not_a_crash():
    scores = {(a, b): 1.0 for a in (1, 2, 3) for b in (1, 2, 3)}
    scores[(2, 2)] = 0.0
    r = analyze_plateau(grid(scores), {"a": 2, "b": 2})
    assert not math.isfinite(r.robustness)
    assert "unscored" in r.verdict()


def test_empty_evaluations_raise():
    with pytest.raises(ValueError, match="no evaluations"):
        analyze_plateau([], {"a": 1})
