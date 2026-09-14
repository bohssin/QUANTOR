"""QUANTOR engine as an MCP server. Plan §14.2.

The agent reaches the engine only through these tools. It does not write
backtest scripts, does not parse stdout, and cannot change the fill model, the
cost assumptions or the fold geometry — those live behind the tool boundary.

Three things that buys (§14.2):

- **Research integrity becomes enforceable** (§13). In strict mode the
  out-of-sample tools return a verdict and there is no filesystem path to read
  around, because the agent was never handed one.
- **Every action is a logged tool call**, so "what did it actually do" is read
  from the transcript rather than reconstructed from shell history.
- **The agent cannot invent its own backtest.**

Run it:

    python3 -m quantor_mcp.server    # stdio

Register it with Claude Code via `.mcp.json`; see `quantor_mcp/README.md`.

Signal blocks are checked by a static AST validator before execution
(`engine.signal.validate_signal_block`) and then run in a restricted namespace.
Two layers with different jobs:

- the **validator** rejects imports, file access, `exec`/`eval`, dunder access
  and forward indexing — the §9 contract, enforced before anything runs;
- the **namespace** exposes only bars, params and the indicator set, so a block
  reaching for an order function gets a `NameError` rather than doing something
  silently wrong.

Neither is a security sandbox — Python cannot be sandboxed by withholding names,
and numba needs `__import__` at call time regardless. These are correctness
boundaries against an agent following the contract imperfectly. Never run a
signal block the owner did not initiate.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import Bars, Instrument, Signals, compute_metrics, run_backtest
from engine.indicators import (
    atr, atr_fast, ema, ema_fast, rma, rma_fast,
    rsi, rsi_fast, sma, sma_fast, true_range, true_range_fast,
)
from engine.optimize import FloatParam, IntParam, ParamSpec, grid_search
from engine.signal import validate_signal_block
from engine.store import build_bars, read_csv
from engine.validate import rolling_folds, walk_forward

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="quantor-engine",
    instructions=(
        "QUANTOR backtest engine. Use these tools to load data, save strategies, "
        "run backtests and sweeps, and render results on the chart. Do NOT write "
        "your own backtest scripts — the engine owns fill semantics, costs and "
        "fold geometry, and results produced outside these tools are not "
        "comparable with anything in the library."
    ),
)

# --- in-process state (the real build puts this in the library, §16) ---------

DATA: dict[str, dict[str, Any]] = {}
STRATEGIES: dict[str, list[dict[str, Any]]] = {}

#: Everything a signal block is allowed to see. Plan §9 — enforcement by
#: namespace, not by prompting.
STRATEGY_GLOBALS: dict[str, Any] = {
    "np": np,
    "sma": sma, "ema": ema, "rma": rma, "rsi": rsi, "atr": atr,
    "true_range": true_range,
    "sma_fast": sma_fast, "ema_fast": ema_fast, "rma_fast": rma_fast,
    "rsi_fast": rsi_fast, "atr_fast": atr_fast, "true_range_fast": true_range_fast,
    # numba dispatchers import at call time, so __import__ has to be present.
    # That is precisely why the validator (not this dict) is the enforcement.
    "__builtins__": {
        "abs": abs, "min": min, "max": max, "len": len, "range": range,
        "float": float, "int": int, "bool": bool, "round": round,
        "sum": sum, "enumerate": enumerate, "zip": zip, "print": print,
        "__import__": __import__, "isinstance": isinstance, "getattr": getattr,
        "hasattr": hasattr, "type": type, "tuple": tuple, "list": list,
        "dict": dict, "set": set, "str": str, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError, "Exception": Exception,
    },
}

#: Fallback only. Each loaded source carries its own Instrument built from the
#: file's detected tick size plus whatever the caller declared — §5 forbids
#: hardcoding anything that drives sizing or P&L, and a single shared default
#: silently prices every instrument as if it were the first one loaded.
DEFAULT_INSTRUMENT = Instrument()


@dataclass
class Version:
    version: int
    description: str
    signal_block: str
    params: dict[str, Any]
    parent: int | None
    created_at: float = field(default_factory=time.time)


# --- error surfacing ---------------------------------------------------------

def surfacing_errors(fn):
    """Return the error text to the agent instead of letting it be swallowed.

    Found by actually running an agent against this server (§19, Probe 9). The
    MCP SDK wraps any exception a tool raises as `UnexpectedToolError: Error
    executing tool <name>` and the real message never reaches the client. The
    agent therefore saw an identical opaque failure for every input — its own
    strategy, a dummy strategy, a nonexistent id — correctly concluded from that
    evidence that the tool was broken server-side, and stopped.

    **An agent cannot fix a mistake it cannot see.** A tool that hides why it
    failed converts a one-line correction into a dead end, so every tool here
    catches its own exceptions and returns the message as text. The type name is
    included because "KeyError: 'fast'" tells the agent exactly what to change,
    while "something went wrong" tells it to give up.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001 - deliberate
            return json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "tool": fn.__name__,
                "hint": _HINTS.get(type(exc).__name__, ""),
            }, indent=2)

    return wrapper


_HINTS = {
    "KeyError": "a parameter the signal block reads was not supplied — check the "
                "params passed to strategy_save or backtest_run",
    "ValueError": "check the arguments against the tool description",
    "IndexError": "an array in the signal block is not the same length as bars",
    "TypeError": "an argument has the wrong type; arrays must be numpy arrays",
}


# --- tools -------------------------------------------------------------------

@server.tool(
    description=(
        "List loaded market data sources with their resolution, GMT offset, row "
        "count and quality summary. Call this first — a strategy can only run on "
        "a timeframe its source is fine enough to serve."
    )
)
@surfacing_errors
def data_list() -> str:
    if not DATA:
        return "No data loaded. Use data_load(path) first."
    rows = []
    for name, d in DATA.items():
        inst = d.get("_instrument")
        rows.append(
            f"{name}: kind={d['kind']} timeframe={d.get('timeframe')} "
            f"rows={d.get('rows', 0):,} bars={d['bars']:,} "
            f"tick_size={d['tick_size']} "
            f"tick_value={getattr(inst, 'tick_value', '?')} "
            f"contract_size={getattr(inst, 'contract_size', '?')} span={d['span']}"
        )
    return "\n".join(rows)


@server.tool(
    description=(
        "Load a tick or bar CSV as a named data source. Columns and price precision "
        "are detected from the file. A TICK source requires `timeframe` (e.g. 'M15') "
        "to build bars from; a bar source infers it from the row spacing. "
        "contract_size / tick_value / commission drive sizing and P&L and cannot be "
        "detected — pass the real ones for the instrument or the money is wrong. "
        "Pass utc_offset_hours if the file's clock is not UTC. Returns the quality "
        "report — read it, do not assume the file is clean."
    )
)
@surfacing_errors
def data_load(
    name: str,
    path: str,
    utc_offset_hours: float = 0.0,
    timeframe: str = "",
    contract_size: float = 100.0,
    tick_value: float = 0.10,
    commission_per_lot_per_side: float = 3.5,
    default_spread: float = 0.30,
) -> str:
    from engine.store import SourceSpec

    md = read_csv(path, SourceSpec(utc_offset_hours=utc_offset_hours))
    if md.kind == "tick" and not timeframe:
        raise ValueError(
            "a tick source needs a timeframe to build bars from — pass e.g. "
            "timeframe='M15'"
        )
    tf = timeframe or _name_for_resolution(md.resolution_ms)
    built = build_bars(md, tf)

    bars = Bars(
        ms=built["ms"], open=built["open"], high=built["high"],
        low=built["low"], close=built["close"],
        spread=built["spread"] if built["spread"].any() else None,
    )

    instrument = Instrument(
        contract_size=contract_size,
        tick_size=md.tick_size,                 # detected from the file, §4.2
        tick_value=tick_value,
        commission_per_lot_per_side=commission_per_lot_per_side,
        default_spread=default_spread,
    )

    DATA[name] = {
        "rows": len(md), "bars": len(bars), "kind": md.kind,
        "timeframe": tf, "resolution_ms": md.resolution_ms,
        "tick_size": md.tick_size, "_bars": bars, "_instrument": instrument,
        "span": f"{md.quality.first_ms}..{md.quality.last_ms}",
    }
    return (
        f"Loaded {name}: {len(md):,} rows -> {len(bars):,} {tf} bars\n"
        f"{md.quality.summary()}\n"
        f"instrument: tick_size={md.tick_size} (detected) tick_value={tick_value} "
        f"contract_size={contract_size} commission={commission_per_lot_per_side}/lot/side"
    )


@server.tool(
    description=(
        "Save a strategy version. `signal_block` is Python source defining "
        "signal(bars, p) -> dict with keys long_entry, short_entry, "
        "stop_distance, target_distance (all numpy arrays aligned to bars). "
        "It may use np and the indicator functions; it must not place orders, "
        "hold position state, or index bars forward. Returns the new version "
        "number; nothing is ever overwritten."
    )
)
@surfacing_errors
def strategy_save(
    strategy_id: str,
    description: str,
    signal_block: str,
    params: dict[str, Any] | None = None,
) -> str:
    result = validate_signal_block(signal_block)          # §9, before anything runs
    result.raise_if_invalid()
    versions = STRATEGIES.setdefault(strategy_id, [])
    v = Version(
        version=len(versions) + 1,
        description=description,
        signal_block=signal_block,
        params=params or {},
        parent=len(versions) or None,
    )
    versions.append(asdict(v))
    msg = f"Saved {strategy_id} v{v.version} (parent: {v.parent})"
    if result.warnings:
        msg += "\nwarnings:\n  - " + "\n  - ".join(result.warnings)
    return msg


@server.tool(description="List strategies and their versions, newest first.")
@surfacing_errors
def strategy_list() -> str:
    if not STRATEGIES:
        return "No strategies saved."
    out = []
    for sid, versions in STRATEGIES.items():
        out.append(f"{sid}: {len(versions)} version(s)")
        for v in reversed(versions[-3:]):
            out.append(f"  v{v['version']}  {v['description'][:70]}")
    return "\n".join(out)


@server.tool(description="Fetch one strategy version's source and parameters.")
@surfacing_errors
def strategy_get(strategy_id: str, version: int = 0) -> str:
    v = _version(strategy_id, version)
    return json.dumps(
        {"version": v["version"], "description": v["description"],
         "params": v["params"], "signal_block": v["signal_block"]},
        indent=2,
    )


@server.tool(
    description=(
        "Run a backtest and return metrics plus a trade summary. This is the only "
        "sanctioned way to produce numbers — results from scripts written outside "
        "the engine are not comparable with anything in the library."
    )
)
@surfacing_errors
def backtest_run(
    strategy_id: str,
    data: str,
    version: int = 0,
    params: dict[str, Any] | None = None,
    timeframe: str = "",
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,
) -> str:
    bars = _bars(data)
    inst = _instrument(data)
    tf = _timeframe(data, timeframe)
    v = _version(strategy_id, version)
    merged = {**v["params"], **(params or {})}

    result = _evaluate(v["signal_block"], bars, merged, inst,
                       initial_capital, risk_pct)
    m = compute_metrics(result, tf, initial_capital)

    reconcile = abs(result.equity[-1] - initial_capital - float(result.pnl.sum()))
    return json.dumps({
        "strategy": f"{strategy_id} v{v['version']}",
        "timeframe": tf,
        "params": merged,
        "metrics": {k: (None if isinstance(val, float) and not np.isfinite(val) else val)
                    for k, val in m.as_dict().items()},
        "ambiguous_exits": result.ambiguous_exits,
        "ambiguity_rate": round(result.ambiguity_rate(), 4),
        "rejected_zero_lots": result.rejected_zero_lots,
        "ledger_reconciles": bool(reconcile < 1e-6),
    }, indent=2)


@server.tool(
    description=(
        "Grid-sweep parameters. `param_grid` maps name -> [low, high, step] for "
        "ints or floats. Returns the top results and the comparison count, which "
        "the deflated Sharpe ratio needs — it accumulates across every sweep of a "
        "strategy family, not per run."
    )
)
@surfacing_errors
def optimize_run(
    strategy_id: str,
    data: str,
    param_grid: dict[str, list[float]],
    version: int = 0,
    fixed: dict[str, Any] | None = None,
    objective: str = "return_over_maxdd",
    min_trades: int = 30,
    timeframe: str = "",
    top_k: int = 10,
) -> str:
    bars = _bars(data)
    inst = _instrument(data)
    tf = _timeframe(data, timeframe)
    v = _version(strategy_id, version)

    params: dict[str, Any] = {}
    for name, (lo, hi, step) in param_grid.items():
        if float(lo).is_integer() and float(hi).is_integer() and float(step).is_integer():
            params[name] = IntParam(int(lo), int(hi), int(step))
        else:
            params[name] = FloatParam(float(lo), float(hi), float(step))
    spec = ParamSpec(params=params, fixed={**v["params"], **(fixed or {})})

    t0 = time.perf_counter()
    res = grid_search(
        spec,
        lambda p: compute_metrics(_evaluate(v["signal_block"], bars, p, inst), tf),
        objective=objective, min_trades=min_trades,
    )
    elapsed = time.perf_counter() - t0

    return json.dumps({
        "strategy": f"{strategy_id} v{v['version']}",
        "timeframe": tf,
        "objective": objective,
        "comparisons": res.comparisons,
        "seconds": round(elapsed, 2),
        "note": "comparisons accumulate across the family for DSR (plan §11)",
        "top": [
            {"params": e.params, "score": None if not np.isfinite(e.score) else round(e.score, 4),
             "net_profit": round(e.metrics["net_profit"], 2),
             "max_drawdown": round(e.metrics["max_drawdown"], 2),
             "n_trades": e.metrics["n_trades"]}
            for e in res.top_k(top_k)
        ],
    }, indent=2)


@server.tool(
    description=(
        "Walk-forward validation: a full parameter search inside each train fold, "
        "and each winner scored once out-of-sample. Efficiency near 1.0 means the "
        "optimization generalized; far below means the train folds were fitted."
    )
)
@surfacing_errors
def validate_run(
    strategy_id: str,
    data: str,
    param_grid: dict[str, list[float]],
    version: int = 0,
    train: int = 12_000,
    test: int = 4_000,
    timeframe: str = "",
) -> str:
    bars = _bars(data)
    inst = _instrument(data)
    tf = _timeframe(data, timeframe)
    v = _version(strategy_id, version)
    folds = rolling_folds(len(bars), train=train, test=test, step=test)
    if not folds:
        return json.dumps({"error": f"not enough bars ({len(bars):,}) for "
                                    f"train={train} test={test}"})

    params: dict[str, Any] = {}
    for name, (lo, hi, step) in param_grid.items():
        if float(lo).is_integer() and float(hi).is_integer() and float(step).is_integer():
            params[name] = IntParam(int(lo), int(hi), int(step))
        else:
            params[name] = FloatParam(float(lo), float(hi), float(step))
    spec = ParamSpec(params=params, fixed=v["params"])

    def search(idx, a, b):
        w = bars.slice(a, b)
        r = grid_search(spec,
                        lambda p: compute_metrics(_evaluate(v["signal_block"], w, p, inst), tf),
                        min_trades=10)
        return r.best.params, r.best.score, r.comparisons

    def test_fold(p, a, b):
        w = bars.slice(a, b)
        m = compute_metrics(_evaluate(v["signal_block"], w, p, inst), tf)
        score = m.net_profit / m.max_drawdown if m.max_drawdown > 0 else float("-inf")
        return score, m.as_dict()

    wf = walk_forward(folds, search, test_fold)
    eff = wf.efficiency()
    return json.dumps({
        "strategy": f"{strategy_id} v{v['version']}",
        "timeframe": tf,
        "folds": len(folds),
        "total_comparisons": wf.total_comparisons,
        "mean_oos_score": None if not np.isfinite(wf.mean_test_score) else round(wf.mean_test_score, 4),
        "walk_forward_efficiency": None if not np.isfinite(eff) else round(eff, 3),
        "per_fold": [
            {"fold": i, "params": wf.chosen_params[i],
             "train": None if not np.isfinite(wf.train_scores[i]) else round(wf.train_scores[i], 3),
             "test": None if not np.isfinite(wf.test_scores[i]) else round(wf.test_scores[i], 3),
             "trades": wf.test_metrics[i].get("n_trades", 0)}
            for i in range(len(folds))
        ],
    }, indent=2)


@server.tool(
    description=(
        "Render a run's own entries, exits and stops for the chart. This draws the "
        "engine's actual trade list, so what is shown is exactly what was measured."
    )
)
@surfacing_errors
def chart_apply(
    strategy_id: str,
    data: str,
    version: int = 0,
    params: dict[str, Any] | None = None,
    max_markers: int = 500,
) -> str:
    bars = _bars(data)
    v = _version(strategy_id, version)
    r = _evaluate(v["signal_block"], bars, {**v["params"], **(params or {})},
                  _instrument(data))
    n = min(r.n_trades, max_markers)
    return json.dumps({
        "strategy": f"{strategy_id} v{v['version']}",
        "trades": r.n_trades,
        "shown": n,
        "markers": [
            {"entry_ms": int(bars.ms[r.entry_i[k]]), "exit_ms": int(bars.ms[r.exit_i[k]]),
             "side": "long" if r.direction[k] > 0 else "short",
             "entry": round(float(r.entry_px[k]), 5), "exit": round(float(r.exit_px[k]), 5),
             "pnl": round(float(r.pnl[k]), 2)}
            for k in range(n)
        ],
    }, indent=2)


# --- internals ---------------------------------------------------------------

def _bars(name: str) -> Bars:
    return _source(name)["_bars"]


def _source(name: str) -> dict[str, Any]:
    if name not in DATA:
        raise ValueError(f"unknown data source {name!r}; loaded: {list(DATA) or 'none'}")
    return DATA[name]


def _instrument(name: str) -> Instrument:
    """The source's own instrument, never a shared default (§5)."""
    return _source(name).get("_instrument", DEFAULT_INSTRUMENT)


def _timeframe(name: str, override: str) -> str:
    """The source knows its own timeframe; an override is for deliberate cases.

    This is not cosmetic. `compute_metrics` annualizes Sharpe and Sortino from
    bars-per-year, so a caller guessing M15 on H1 data reports a Sharpe wrong by
    a factor of two, silently and in the flattering direction as often as not.
    """
    return override or _source(name).get("timeframe") or "M15"


def _name_for_resolution(ms: int) -> str:
    from engine.store import TIMEFRAMES
    for name, value in TIMEFRAMES.items():
        if value == ms:
            return name
    return "M15"


def _version(strategy_id: str, version: int) -> dict[str, Any]:
    versions = STRATEGIES.get(strategy_id)
    if not versions:
        raise ValueError(f"unknown strategy {strategy_id!r}; saved: {list(STRATEGIES) or 'none'}")
    if version in (0, -1):
        return versions[-1]
    if not 1 <= version <= len(versions):
        raise ValueError(f"{strategy_id} has versions 1..{len(versions)}, asked for {version}")
    return versions[version - 1]


def _evaluate(signal_block: str, bars: Bars, params: dict[str, Any],
              instrument: Instrument | None = None,
              initial_capital: float = 10_000.0, risk_pct: float = 0.01):
    """Execute a signal block in the restricted namespace and backtest it."""
    ns: dict[str, Any] = dict(STRATEGY_GLOBALS)
    exec(compile(signal_block, "<signal_block>", "exec"), ns)
    fn = ns.get("signal")
    if fn is None:
        raise ValueError("signal block must define signal(bars, p)")

    out = fn(bars, params)
    try:
        sig = Signals(
            long_entry=np.asarray(out["long_entry"], dtype=bool),
            short_entry=np.asarray(out["short_entry"], dtype=bool),
            stop_distance=np.asarray(out["stop_distance"], dtype=np.float64),
            target_distance=np.asarray(out["target_distance"], dtype=np.float64),
        )
    except KeyError as exc:
        raise ValueError(
            f"signal() must return a dict with long_entry, short_entry, "
            f"stop_distance, target_distance — missing {exc}"
        ) from None

    return run_backtest(bars, sig, instrument or DEFAULT_INSTRUMENT,
                        initial_capital=initial_capital, risk_pct=risk_pct)


if __name__ == "__main__":
    server.run(transport="stdio")
