"""Plateau analysis over a parameter sweep. Plan §11.

**A sharp optimum is usually a fitting artifact.** If a configuration scores well
and its immediate neighbours score badly, the search found a hole in the noise,
not an edge — and it will not survive contact with new data, because next year's
noise has its holes somewhere else. A configuration surrounded by *comparably
good* neighbours is describing something real about the market.

This matters more than it used to. Cross-symbol survival was the plan's strongest
robustness filter and has been dropped, so the defences that remain — walk-forward
efficiency, the deflated Sharpe over a cumulative trial count, and this — carry
the whole load.

The measure is deliberately simple, because a complicated robustness score that
nobody interprets is worse than a blunt one that gets read:

    robustness = mean(neighbour scores) / peak score

Near 1.0 means a plateau: the neighbours are as good as the peak. Near 0 or
negative means an isolated spike. The neighbourhood is the grid-adjacent
configurations — one step in each parameter — so it costs nothing extra when the
sweep was a grid, the points are already evaluated.

**The ratio is only defined for a positive peak.** Divide by a negative score and
the scale inverts: a peak of -0.91 with neighbours averaging -0.96 gives 1.06,
which reads as the flattest possible plateau while actually describing a
configuration that loses money and whose neighbours lose more. So a non-positive
peak is reported as unscored, with the neighbour statistics kept — there is
nothing to be robust about at a setting that does not make money in the first
place, and saying so is more useful than a number pointing the wrong way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PlateauReport", "analyze_plateau", "rank_by_robustness"]


@dataclass
class PlateauReport:
    params: dict
    score: float
    n_neighbours: int
    neighbour_mean: float
    neighbour_min: float
    neighbour_std: float
    #: mean(neighbours) / peak. Near 1.0 = plateau; near 0 = isolated spike.
    robustness: float

    @property
    def is_plateau(self) -> bool:
        """Neighbours hold most of the peak's score. The threshold is a default,
        not a law — §17 owns the real one."""
        return self.robustness >= 0.7

    def verdict(self) -> str:
        if math.isfinite(self.score) and self.score <= 0:
            return ("unscored — the best configuration is not profitable, so "
                    "there is no peak to be robust around")
        if not math.isfinite(self.robustness):
            return "unscored (peak score is zero or non-finite)"
        if self.n_neighbours == 0:
            return "no neighbours in the grid — cannot judge"
        if self.robustness >= 0.9:
            return "flat plateau — neighbours score as well as the peak"
        if self.robustness >= 0.7:
            return "plateau — neighbours hold up"
        if self.robustness >= 0.4:
            return "slope — neighbours are meaningfully worse, treat with caution"
        return "isolated spike — probably a fitting artifact"


def analyze_plateau(evaluations, target: dict, *,
                    swept: list[str] | None = None) -> PlateauReport:
    """Score one configuration against its grid-adjacent neighbours.

    `evaluations` is a `SearchResult.evaluations` list. `target` is the
    configuration to judge — usually `result.best.params`.
    """
    if not evaluations:
        raise ValueError("no evaluations to analyse")

    names = swept if swept is not None else _swept_names(evaluations)
    axes = {n: sorted({e.params[n] for e in evaluations if n in e.params}) for n in names}

    by_key = {}
    for e in evaluations:
        by_key[_key(e.params, names)] = e.score

    target_key = _key(target, names)
    score = by_key.get(target_key, float("nan"))

    neighbour_scores = [
        by_key[k] for k in _neighbour_keys(target_key, names, axes)
        if k in by_key and math.isfinite(by_key[k])
    ]

    if not neighbour_scores or not math.isfinite(score):
        return PlateauReport(target, score, len(neighbour_scores),
                             float("nan"), float("nan"), float("nan"), float("nan"))

    arr = np.asarray(neighbour_scores, dtype=float)
    return PlateauReport(
        params=target,
        score=float(score),
        n_neighbours=int(arr.size),
        neighbour_mean=float(arr.mean()),
        neighbour_min=float(arr.min()),
        neighbour_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        # Undefined at or below zero — see the module docstring. The neighbour
        # statistics above stay populated because they are still readable.
        robustness=float(arr.mean() / score) if score > 0 else float("nan"),
    )


def rank_by_robustness(evaluations, top_k: int = 10,
                       min_robustness: float = 0.7) -> list[PlateauReport]:
    """Re-rank the best configurations by how well their neighbourhoods hold up.

    The point of the exercise: the highest-scoring configuration is often not the
    one to take. This returns plateau candidates first, each still carrying its
    raw score, so the trade-off is visible rather than decided silently.
    """
    finite = [e for e in evaluations if math.isfinite(e.score)]
    if not finite:
        return []

    ranked = sorted(finite, key=lambda e: e.score, reverse=True)[:top_k]
    reports = [analyze_plateau(evaluations, e.params) for e in ranked]

    def sort_key(r: PlateauReport):
        robust = r.robustness if math.isfinite(r.robustness) else -math.inf
        return (robust >= min_robustness, r.score)

    return sorted(reports, key=sort_key, reverse=True)


# --- internals ---------------------------------------------------------------

def _swept_names(evaluations) -> list[str]:
    """Parameters that actually vary. A fixed parameter has no neighbours."""
    names = []
    for name in evaluations[0].params:
        values = {e.params.get(name) for e in evaluations}
        if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
            names.append(name)
    return names


def _key(params: dict, names: list[str]) -> tuple:
    return tuple(params.get(n) for n in names)


def _neighbour_keys(key: tuple, names: list[str], axes: dict) -> list[tuple]:
    """One grid step in each direction, in each swept parameter."""
    out = []
    for i, name in enumerate(names):
        values = axes[name]
        try:
            pos = values.index(key[i])
        except ValueError:
            continue
        for step in (-1, 1):
            j = pos + step
            if 0 <= j < len(values):
                neighbour = list(key)
                neighbour[i] = values[j]
                out.append(tuple(neighbour))
    return out
