"""Fold geometry and walk-forward. Plan §12.

Three subtleties here that the plan did not previously state, each of which
silently inflates out-of-sample results if missed:

**A trade is attributed to the fold containing its ENTRY.** Trades straddle
fold boundaries, and the obvious alternatives are both wrong: counting a trade
in both folds double-counts it, and counting by exit lets a position opened on
in-sample information score in the out-of-sample window.

**The purge window must be at least the longest trade duration.** Purged k-fold
exists because a label computed over a window overlapping the test set leaks.
In a backtest the "label" is a trade's outcome, so its span is the trade's
duration. Purging by a fixed number of bars chosen by feel is not purging — if
trades routinely last 40 bars and the purge is 10, the leak is still there.
`purge_bars_for` derives it from the observed trade durations.

**Walk-forward nests a full search inside every train fold.** The cost is
`folds x evaluations`, not `evaluations`, and each fold's winner is evaluated
out-of-sample exactly once. Re-tuning after seeing a test fold turns the whole
exercise into an expensive way to overfit (§13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["Fold", "rolling_folds", "anchored_folds", "purged_kfold",
           "purge_bars_for", "attribute_trades", "walk_forward", "WalkForwardResult"]


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int      # exclusive
    test_start: int
    test_end: int       # exclusive

    @property
    def train_len(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_len(self) -> int:
        return self.test_end - self.test_start


def rolling_folds(n_bars: int, train: int, test: int,
                  step: int | None = None, embargo: int = 0) -> list[Fold]:
    """Sliding fixed-width train window. Tests adaptation over time."""
    step = step or test
    folds: list[Fold] = []
    start = 0
    while True:
        tr_end = start + train
        te_start = tr_end + embargo
        te_end = te_start + test
        if te_end > n_bars:
            break
        folds.append(Fold(len(folds), start, tr_end, te_start, te_end))
        start += step
    return folds


def anchored_folds(n_bars: int, train: int, test: int,
                   step: int | None = None, embargo: int = 0) -> list[Fold]:
    """Train window grows from a fixed start. Tests whether more history helps."""
    step = step or test
    folds: list[Fold] = []
    tr_end = train
    while True:
        te_start = tr_end + embargo
        te_end = te_start + test
        if te_end > n_bars:
            break
        folds.append(Fold(len(folds), 0, tr_end, te_start, te_end))
        tr_end += step
    return folds


def purge_bars_for(entry_i: np.ndarray, exit_i: np.ndarray,
                   quantile: float = 0.99) -> int:
    """Purge width implied by observed trade durations.

    Uses a high quantile rather than the max so one pathological trade does not
    consume the whole dataset, and rounds up so the common case is fully
    covered. Plan §12: a purge narrower than the label span is not a purge.
    """
    if entry_i.size == 0:
        return 0
    durations = (exit_i - entry_i).astype(np.int64)
    return int(np.ceil(np.quantile(durations, quantile)))


def purged_kfold(n_bars: int, k: int, purge: int, embargo: int = 0) -> list[Fold]:
    """Time-series CV with purging around each test fold.

    Train is everything outside [test_start - purge, test_end + embargo). The
    returned Fold carries the test window; `train_mask` builds the train index
    because purged training data is not contiguous.
    """
    bounds = np.linspace(0, n_bars, k + 1).round().astype(int)
    folds: list[Fold] = []
    for i in range(k):
        te_start, te_end = int(bounds[i]), int(bounds[i + 1])
        folds.append(Fold(i, 0, n_bars, te_start, te_end))
    return folds


def train_mask(n_bars: int, fold: Fold, purge: int, embargo: int = 0) -> np.ndarray:
    """Boolean mask of bars usable for training given a purged test fold."""
    mask = np.ones(n_bars, dtype=bool)
    lo = max(0, fold.test_start - purge)
    hi = min(n_bars, fold.test_end + embargo)
    mask[lo:hi] = False
    return mask


def attribute_trades(entry_i: np.ndarray, start: int, end: int) -> np.ndarray:
    """Indices of trades belonging to [start, end) — by ENTRY bar.

    One rule, applied everywhere: a trade counts in the window where the
    decision was made. Counting by exit would let a position opened on
    in-sample information score out-of-sample.
    """
    return np.flatnonzero((entry_i >= start) & (entry_i < end))


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    chosen_params: list[dict]
    train_scores: list[float]
    test_scores: list[float]
    test_metrics: list[dict]
    total_comparisons: int

    @property
    def mean_test_score(self) -> float:
        finite = [s for s in self.test_scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else float("-inf")

    def efficiency(self) -> float:
        """Out-of-sample score as a fraction of in-sample.

        Near 1.0 means the optimization generalized. Far below it means the
        train folds were fitted. Above ~1.2 usually means the test windows were
        easier, not that the strategy improved — check before celebrating.
        """
        tr = [s for s in self.train_scores if np.isfinite(s) and s != 0]
        te = [s for s, t in zip(self.test_scores, self.train_scores)
              if np.isfinite(s) and np.isfinite(t) and t != 0]
        if not tr or not te:
            return float("nan")
        return float(np.mean(te) / np.mean(tr))


def walk_forward(
    folds: Sequence[Fold],
    search: Callable[[int, int, int], tuple[dict, float, int]],
    evaluate_test: Callable[[dict, int, int], tuple[float, dict]],
) -> WalkForwardResult:
    """Run a full search per train fold, then score its winner once on test.

    `search(fold_index, train_start, train_end) -> (params, train_score, comparisons)`
    `evaluate_test(params, test_start, test_end) -> (test_score, metrics)`

    The separation is the point: `search` never receives test indices, so it
    cannot read out-of-sample data even by accident.
    """
    chosen, train_scores, test_scores, test_metrics = [], [], [], []
    comparisons = 0

    for fold in folds:
        params, train_score, n_compared = search(
            fold.index, fold.train_start, fold.train_end)
        comparisons += n_compared
        score, metrics = evaluate_test(params, fold.test_start, fold.test_end)
        chosen.append(params)
        train_scores.append(train_score)
        test_scores.append(score)
        test_metrics.append(metrics)

    return WalkForwardResult(
        folds=list(folds), chosen_params=chosen, train_scores=train_scores,
        test_scores=test_scores, test_metrics=test_metrics,
        total_comparisons=comparisons,
    )
