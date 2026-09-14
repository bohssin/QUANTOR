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

**This file is an adapter, not an implementation.** Every operation lives in
`engine/service.py`, which the HTTP API calls too. That is deliberate: when the
agent and the UI each had their own copy, the failure mode was a strategy
measuring Sharpe 1.4 in one and 0.9 in the other, with no error anywhere and no
way to tell which was real. One implementation, two adapters, and a test that
runs the same strategy through both and compares every number.

Results persist. Strategies, versions, runs, metrics and cached bars live in the
library (§16), so closing the session no longer discards the work — and the
cumulative comparison count the deflated Sharpe needs (§12) can finally exist,
since it is a claim about history.

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

import functools
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.service import Quantor

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="quantor-engine",
    instructions=(
        "QUANTOR backtest engine. Use these tools to load data, save strategies, "
        "run backtests and sweeps, and render results on the chart. Do NOT write "
        "your own backtest scripts — the engine owns fill semantics, costs and "
        "fold geometry, and results produced outside these tools are not "
        "comparable with anything in the library. Everything you save persists: "
        "strategies, versions and runs are still here in the next session."
    ),
)

#: One library, shared with the API server through the file. Both processes
#: write it, which is why the connection opens in WAL mode (see engine/library).
QUANTOR = Quantor(os.environ.get("QUANTOR_LIBRARY") or None)


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
    "FileNotFoundError": "the path is read on the machine running this server",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


# --- data --------------------------------------------------------------------

@server.tool(
    description=(
        "List loaded market data sources with their resolution, GMT offset, row "
        "count and quality summary. Call this first — a strategy can only run on "
        "a timeframe its source is fine enough to serve. Sources persist across "
        "sessions, so what is listed here may have been loaded days ago."
    )
)
@surfacing_errors
def data_list() -> str:
    sources = QUANTOR.list_data()
    if not sources:
        return "No data loaded. Use data_load(name, path, timeframe) first."
    lines = []
    for d in sources:
        lines.append(
            f"{d['name']}: kind={d['kind']} base={d['base_timeframe']} "
            f"rows={d['rows']:,} bars={d['bars']:,} tick_size={d['tick_size']} "
            f"gmt{d['utc_offset_hours']:+g} span={d['span']} ({d['span_days']}d) "
            f"tick_value={d['instrument'].get('tick_value')} "
            f"contract_size={d['instrument'].get('contract_size')}"
        )
        lines.append(f"  serves: {', '.join(d['serves'][:12])}")
    return "\n".join(lines)


@server.tool(
    description=(
        "Load a tick or bar CSV as a named data source. Columns and price "
        "precision are detected from the file. A TICK source requires `timeframe` "
        "(e.g. 'M15'); bars are cached at M1 so every coarser timeframe is then "
        "free — a large archive is read once, ever. Files above ~256 MB stream in "
        "bounded memory automatically. contract_size / tick_value / commission "
        "drive sizing and P&L and cannot be detected — pass the real ones for the "
        "instrument or the money is wrong. Pass utc_offset_hours if the file's "
        "clock is not UTC (e.g. 3.0 for a GMT+3 export). Returns the quality "
        "report — read it, do not assume the file is clean."
    )
)
@surfacing_errors
def data_load(
    name: str,
    path: str,
    timeframe: str = "",
    utc_offset_hours: float = 0.0,
    contract_size: float = 100.0,
    tick_value: float = 0.10,
    commission_per_lot_per_side: float = 3.5,
    default_spread: float = 0.30,
) -> str:
    out = QUANTOR.load_data(
        name=name, path=path, timeframe=timeframe,
        utc_offset_hours=utc_offset_hours, contract_size=contract_size,
        tick_value=tick_value,
        commission_per_lot_per_side=commission_per_lot_per_side,
        default_spread=default_spread,
    )
    return (
        f"Loaded {name}: {out['rows']:,} {out['kind']} rows -> "
        f"{out['base_bars']:,} {out['base_timeframe']} bars "
        f"({out['bars']:,} at {out['timeframe']}) in {out['seconds']}s "
        f"via {out['ingest']} ingest\n"
        f"{out['quality_summary']}\n"
        f"span: {out['span']}\n"
        f"instrument: tick_size={out['tick_size']} (detected) "
        f"tick_value={tick_value} contract_size={contract_size} "
        f"commission={commission_per_lot_per_side}/lot/side\n"
        f"Cached — any timeframe from {out['base_timeframe']} upward is now free."
    )


# --- strategies --------------------------------------------------------------

@server.tool(
    description=(
        "Save a strategy version. `signal_block` is Python source defining "
        "signal(bars, p) -> dict with keys long_entry, short_entry, "
        "stop_distance, target_distance (all numpy arrays aligned to bars). "
        "It may use np and the indicator functions; it must not place orders, "
        "hold position state, or index bars forward. Nothing is ever "
        "overwritten — versions form a tree, and `parent_version` branches from "
        "an older one. Set `family` to the same value across variants of one "
        "idea so their comparison counts accumulate for the deflated Sharpe. "
        "Set `source_url` when the idea came from the LuxAlgo Library — its "
        "licence is free WITH attribution."
    )
)
@surfacing_errors
def strategy_save(
    strategy_id: str,
    description: str,
    signal_block: str,
    params: dict[str, Any] | None = None,
    parent_version: int | None = None,
    family: str = "",
    source_url: str = "",
    origin: str = "agent",
) -> str:
    out = QUANTOR.save_strategy(
        strategy_id=strategy_id, description=description,
        signal_block=signal_block, params=params, parent_version=parent_version,
        family=family, source_url=source_url, origin=origin,
    )
    msg = (f"Saved {strategy_id} v{out['version']} "
           f"(parent: {out['parent_version']}, family: {out['family']})")
    if out["warnings"]:
        msg += "\nwarnings:\n  - " + "\n  - ".join(out["warnings"])
    return msg


@server.tool(
    description=(
        "List strategies, newest activity first, with version and run counts and "
        "the cumulative comparison count for each family. Add include_archived to "
        "see rejected ones — they are kept, never deleted, because 'we tried this "
        "and it failed walk-forward' is a result."
    )
)
@surfacing_errors
def strategy_list(include_archived: bool = False) -> str:
    rows = QUANTOR.list_strategies(include_archived=include_archived)
    if not rows:
        return "No strategies saved."
    out = []
    for s in rows:
        flag = "  [archived]" if s["archived"] else ""
        out.append(
            f"{s['strategy_id']}{flag}: {s['versions']} version(s), "
            f"{s['runs']} run(s), family={s['family']}, "
            f"{s['comparisons']:,} cumulative comparisons"
        )
        out.append(f"  v{s['latest']['version']}  {s['latest']['description'][:70]}")
        if s["archived"] and s["archived_reason"]:
            out.append(f"  reason: {s['archived_reason']}")
    return "\n".join(out)


@server.tool(
    description=("Fetch one strategy version's source, parameters and lineage. "
                 "version=0 means the latest.")
)
@surfacing_errors
def strategy_get(strategy_id: str, version: int = 0) -> str:
    return _json(QUANTOR.get_strategy(strategy_id, version))


@server.tool(
    description=(
        "Archive a strategy with a reason, or restore it with archived=false. "
        "Archiving hides it from the default list and deletes nothing: its runs, "
        "metrics and the reason it was rejected stay queryable. Record why — a "
        "rejection without a reason gets re-tried in three months."
    )
)
@surfacing_errors
def strategy_archive(strategy_id: str, reason: str = "", archived: bool = True) -> str:
    out = QUANTOR.archive_strategy(strategy_id, reason, archived)
    verb = "Archived" if archived else "Restored"
    return f"{verb} {strategy_id}" + (f": {reason}" if reason else "")


# --- runs --------------------------------------------------------------------

@server.tool(
    description=(
        "Run a backtest and return metrics plus a trade summary. This is the only "
        "sanctioned way to produce numbers — results from scripts written outside "
        "the engine are not comparable with anything in the library. The run is "
        "recorded with its parameters, data source and equity curve, and the "
        "run_id it returns can be reopened later."
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
    return _json(QUANTOR.backtest(
        strategy_id=strategy_id, data=data, version=version, params=params,
        timeframe=timeframe, initial_capital=initial_capital, risk_pct=risk_pct,
    ))


@server.tool(
    description=(
        "Grid-sweep parameters. `param_grid` maps name -> [low, high, step] for "
        "ints or floats. Returns the top results, a plateau verdict on the winner "
        "(a sharp optimum surrounded by bad neighbours is a fitting artifact, not "
        "an edge), and the comparison count — which accumulates across every "
        "sweep of a strategy family, because that total is what the deflated "
        "Sharpe ratio divides by."
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
    return _json(QUANTOR.optimize(
        strategy_id=strategy_id, data=data, param_grid=param_grid, version=version,
        fixed=fixed, objective=objective, min_trades=min_trades,
        timeframe=timeframe, top_k=top_k,
    ))


@server.tool(
    description=(
        "Walk-forward validation: a full parameter search inside each train fold, "
        "and each winner scored once out-of-sample. Efficiency near 1.0 means the "
        "optimization generalized; far below means the train folds were fitted. "
        "Returns a plain-words verdict alongside the number."
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
    min_trades: int = 10,
) -> str:
    return _json(QUANTOR.validate(
        strategy_id=strategy_id, data=data, param_grid=param_grid, version=version,
        train=train, test=test, timeframe=timeframe, min_trades=min_trades,
    ))


@server.tool(
    description=(
        "Re-run a strategy on SHUFFLED returns and compare. The cheapest test "
        "for whether an edge is real: shuffling bar-to-bar returns keeps the "
        "distribution and destroys the order, which is the only thing a "
        "strategy can read. An edge that survives the shuffle is look-ahead or "
        "a sizing artifact, not an edge. Run this before believing any result."
    )
)
@surfacing_errors
def control_test(
    strategy_id: str,
    data: str,
    version: int = 0,
    params: dict[str, Any] | None = None,
    timeframe: str = "",
    trials: int = 5,
    seed: int = 0,
) -> str:
    return _json(QUANTOR.control_test(
        strategy_id=strategy_id, data=data, version=version, params=params,
        timeframe=timeframe, trials=trials, seed=seed,
    ))


@server.tool(
    description=(
        "Render a run's own entries, exits and stops for the chart. This draws the "
        "engine's actual trade list, so what is shown is exactly what was "
        "measured. Returns bars, markers, the equity curve and the metrics for "
        "the same window; the UI reads the identical payload."
    )
)
@surfacing_errors
def chart_apply(
    data: str,
    strategy_id: str = "",
    version: int = 0,
    params: dict[str, Any] | None = None,
    timeframe: str = "",
    limit: int = 1_000,
    max_markers: int = 200,
) -> str:
    out = QUANTOR.chart(
        data=data, strategy_id=strategy_id, version=version, params=params,
        timeframe=timeframe, limit=limit, max_markers=max_markers,
    )
    # The full bar series is for the browser; the agent wants the summary and
    # the trades, and shipping 4,000 OHLC rows into a transcript helps nobody.
    out.pop("bars", None)
    out.pop("equity", None)
    out.pop("markers", None)
    out["trades"] = out.get("trades", [])[:max_markers]
    return _json(out)


@server.tool(
    description=(
        "Past runs, newest first — backtests, sweeps and validations with their "
        "parameters, metrics and status. Filter by strategy_id or kind. This is "
        "the record of what was actually tried, and it survives restarts."
    )
)
@surfacing_errors
def run_history(strategy_id: str = "", kind: str = "", limit: int = 25) -> str:
    rows = QUANTOR.run_history(strategy_id=strategy_id, kind=kind, limit=limit)
    if not rows:
        return "No runs recorded yet."
    return _json(rows)


@server.tool(
    description=(
        "Reopen one run by id: its parameters, metrics, and the full result table "
        "for a sweep or the per-fold detail for a validation."
    )
)
@surfacing_errors
def run_get(run_id: int) -> str:
    out = QUANTOR.get_run(run_id)
    if out is None:
        raise ValueError(f"no run {run_id}; use run_history to list them")
    return _json(out)


@server.tool(
    description=("What the library holds: data sources, strategies, versions, "
                 "runs and artifacts, with where it lives on disk.")
)
@surfacing_errors
def library_stats() -> str:
    return _json(QUANTOR.stats())


if __name__ == "__main__":
    server.run(transport="stdio")
