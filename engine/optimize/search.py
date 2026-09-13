"""Parameter search. Plan §11.

Search space is **declared**, not discovered from a script: a `ParamSpec` gives
the optimizer types, bounds and step for free, and unlike rev 3's runtime
introspection it has no stale-cache failure mode (§2).

Three things here that are easy to get wrong and expensive to notice late:

**The comparison count is cumulative, not per-sweep.** §12's deflated Sharpe
adjusts for the number of configurations tried, and that number is the total
across the whole research programme for a strategy family — not the size of the
last sweep. Three sweeps of 1,000 is 3,000 trials, and reporting 1,000 inflates
DSR. `SearchResult.comparisons` counts this run; the library (§16) accumulates
it across runs, and the policy's family budget is checked against the total.

**Optimize on train folds only.** Nothing in this module reads a test or
holdout span. Walk-forward (`engine.validate`) nests a search inside each train
fold and evaluates the winner out-of-sample exactly once.

**Plateaus beat peaks.** A sharp optimum is usually a fitting artifact, so the
default reporting surfaces the neighbourhood, not just the argmax. `top_k` and
the full result table exist for that; `best` is a convenience, not the answer.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal, Sequence

import numpy as np

__all__ = ["IntParam", "FloatParam", "ChoiceParam", "ParamSpec",
           "Evaluation", "SearchResult", "grid_search", "random_search",
           "OBJECTIVES", "objective_value"]


@dataclass(frozen=True)
class IntParam:
    low: int
    high: int
    step: int = 1

    def grid(self, max_values: int) -> list[int]:
        values = list(range(self.low, self.high + 1, self.step))
        return _thin(values, max_values)

    def sample(self, rng: np.random.Generator) -> int:
        n = (self.high - self.low) // self.step
        return int(self.low + self.step * rng.integers(0, n + 1))


@dataclass(frozen=True)
class FloatParam:
    low: float
    high: float
    step: float | None = None

    def grid(self, max_values: int) -> list[float]:
        if self.step:
            n = int(round((self.high - self.low) / self.step)) + 1
            values = [self.low + i * self.step for i in range(n)]
        else:
            values = list(np.linspace(self.low, self.high, max_values))
        return _thin([round(v, 10) for v in values], max_values)

    def sample(self, rng: np.random.Generator) -> float:
        v = float(rng.uniform(self.low, self.high))
        if self.step:
            v = self.low + round((v - self.low) / self.step) * self.step
        return round(min(max(v, self.low), self.high), 10)


@dataclass(frozen=True)
class ChoiceParam:
    options: Sequence[object]

    def grid(self, max_values: int) -> list[object]:
        return list(self.options)

    def sample(self, rng: np.random.Generator) -> object:
        return self.options[int(rng.integers(0, len(self.options)))]


Param = IntParam | FloatParam | ChoiceParam


@dataclass(frozen=True)
class ParamSpec:
    """The declared search space. Fixed values are allowed and simply not swept."""

    params: dict[str, Param]
    fixed: dict[str, object] = field(default_factory=dict)

    def grid_size(self, max_values_per_param: int) -> int:
        total = 1
        for p in self.params.values():
            total *= len(p.grid(max_values_per_param))
        return total

    def grid(self, max_values_per_param: int) -> Iterator[dict]:
        names = list(self.params)
        axes = [self.params[n].grid(max_values_per_param) for n in names]
        for combo in itertools.product(*axes):
            yield {**self.fixed, **dict(zip(names, combo))}

    def sample(self, rng: np.random.Generator) -> dict:
        return {**self.fixed,
                **{n: p.sample(rng) for n, p in self.params.items()}}


@dataclass
class Evaluation:
    params: dict
    score: float
    metrics: dict


@dataclass
class SearchResult:
    evaluations: list[Evaluation]
    objective: str
    #: Configurations evaluated in THIS run. §12's DSR needs the cumulative
    #: total across the family — the library holds that, not this object.
    comparisons: int

    @property
    def best(self) -> Evaluation | None:
        return max(self.evaluations, key=lambda e: e.score, default=None)

    def top_k(self, k: int = 10) -> list[Evaluation]:
        return sorted(self.evaluations, key=lambda e: e.score, reverse=True)[:k]

    def table(self) -> list[dict]:
        return [{**e.params, "score": e.score, **e.metrics} for e in self.evaluations]


# --- objectives --------------------------------------------------------------

def _guard(value: float) -> float:
    """Unusable scores sort last rather than crashing or winning by accident."""
    return value if math.isfinite(value) else -math.inf


OBJECTIVES: dict[str, Callable[[object], float]] = {
    "net_profit": lambda m: _guard(m.net_profit),
    "profit_factor": lambda m: _guard(m.profit_factor),
    "expectancy_r": lambda m: _guard(m.expectancy_r),
    "sharpe": lambda m: _guard(m.sharpe),
    "sortino": lambda m: _guard(m.sortino),
    "calmar": lambda m: _guard(m.calmar),
    "sqn": lambda m: _guard(m.sqn),
    # Drawdown-aware default: raw return reliably surfaces the highest-variance
    # survivor, which wins the backtest and blows up live (§11).
    "return_over_maxdd": lambda m: _guard(
        m.net_profit / m.max_drawdown if m.max_drawdown > 0 else -math.inf
    ),
}

DEFAULT_OBJECTIVE = "return_over_maxdd"


def objective_value(metrics, objective: str = DEFAULT_OBJECTIVE,
                    *, min_trades: int = 0) -> float:
    """Score one result. Too few trades is disqualifying, not a high score.

    Without the trade-count floor, a configuration that happens to take two
    lucky trades outranks one that took four hundred — the single most common
    way an optimizer produces a nonsense winner.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; known: {sorted(OBJECTIVES)}")
    if metrics.n_trades < min_trades:
        return -math.inf
    return OBJECTIVES[objective](metrics)


# --- search modes ------------------------------------------------------------

def grid_search(spec: ParamSpec, evaluate: Callable[[dict], object], *,
                objective: str = DEFAULT_OBJECTIVE,
                max_values_per_param: int = 10,
                min_trades: int = 0,
                max_evaluations: int | None = None) -> SearchResult:
    """Full factorial over a coarse grid.

    Coarse over fine, deliberately: the aim is to find a plateau, and a fine
    grid costs exponentially more while making a sharp artifact easier to
    mistake for a peak (§11).
    """
    evaluations: list[Evaluation] = []
    for params in spec.grid(max_values_per_param):
        if max_evaluations is not None and len(evaluations) >= max_evaluations:
            break
        metrics = evaluate(params)
        evaluations.append(Evaluation(
            params=params,
            score=objective_value(metrics, objective, min_trades=min_trades),
            metrics=metrics.as_dict(),
        ))
    return SearchResult(evaluations, objective, len(evaluations))


def random_search(spec: ParamSpec, evaluate: Callable[[dict], object], *,
                  objective: str = DEFAULT_OBJECTIVE,
                  n_evaluations: int = 100,
                  min_trades: int = 0,
                  seed: int = 0) -> SearchResult:
    """Random sampling. Beats grid per unit compute above ~3 dimensions.

    Seeded, so a run is reproducible — the engine is deterministic even though
    the agent that configured it is not (§7).
    """
    rng = np.random.default_rng(seed)
    evaluations: list[Evaluation] = []
    for _ in range(n_evaluations):
        params = spec.sample(rng)
        metrics = evaluate(params)
        evaluations.append(Evaluation(
            params=params,
            score=objective_value(metrics, objective, min_trades=min_trades),
            metrics=metrics.as_dict(),
        ))
    return SearchResult(evaluations, objective, len(evaluations))


def _thin(values: list, max_values: int) -> list:
    if max_values <= 0 or len(values) <= max_values:
        return values
    idx = np.linspace(0, len(values) - 1, max_values).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(values[i])
    return out
