"""One implementation of every operation, for every surface. Plan §14, §15, §16.

The MCP server and the HTTP API expose the same nine operations. If each wrote
its own, they would drift — and the way that drift shows up is the worst
possible way: a strategy the agent measured at Sharpe 1.4 displaying 0.9 in the
UI, with no error anywhere and no way to tell which one is real.

So neither has a copy. `quantor_mcp/server.py` is an MCP adapter over this file
and `app/api/main.py` is an HTTP adapter over this file. Both return the same
dicts from the same calls, which `tests/integration/test_cross_surface.py`
asserts by running a strategy through both and comparing every number.

Everything here also **records to the library** (§16): a run that produced a
number is a row with its parameters, its data source, its metrics and an
artifact holding its equity curve and trade list. That is what makes "what did
we try, and what came of it" answerable after a restart.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engine.backtest import Bars, Instrument, Signals, compute_metrics, run_backtest
from engine.indicators import (
    atr, atr_fast, ema, ema_fast, rma, rma_fast,
    rsi, rsi_fast, sma, sma_fast, true_range, true_range_fast,
)
from engine.library import Library
from engine.optimize import FloatParam, IntParam, ParamSpec, grid_search
from engine.optimize.plateau import analyze_plateau
from engine.signal import validate_signal_block
from engine.store import SourceSpec, TIMEFRAMES, build_bars, read_csv, stream_bars
from engine.store.bars import aggregate_bars
from engine.store.ingest import timeframe_ms
from engine.validate import rolling_folds, walk_forward

__all__ = ["Quantor", "STRATEGY_GLOBALS", "evaluate"]

#: Above this many rows, read the file as a stream instead of materializing it.
#: A 400 MB tick CSV is ~10M rows and already costs more resident memory than
#: the bars it produces; an 11 GB one is not an option at all (§4.2).
STREAM_ABOVE_BYTES = 256 << 20

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


def evaluate(signal_block: str, bars: Bars, params: dict[str, Any],
             instrument: Instrument, initial_capital: float = 10_000.0,
             risk_pct: float = 0.01, subbars: Bars | None = None,
             subbar_timeframe: str = "", bar_ms: int = 0):
    """Run a signal block and backtest what it produced. Plan §9.

    The block is checked by the AST validator before it is saved, not here —
    this is the hot path and re-parsing on every one of 4,000 sweep evaluations
    would dominate the sweep.
    """
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
            "signal() must return a dict with long_entry, short_entry, "
            f"stop_distance, target_distance — missing {exc}"
        ) from None
    except TypeError:
        raise ValueError(
            f"signal() must return a dict, got {type(out).__name__}"
        ) from None

    return run_backtest(bars, sig, instrument, initial_capital=initial_capital,
                        risk_pct=risk_pct, subbars=subbars,
                        subbar_timeframe=subbar_timeframe, bar_ms=bar_ms)


@dataclass
class LoadedBars:
    bars: Bars
    instrument: Instrument
    timeframe: str
    name: str


class Quantor:
    """Every operation the platform performs, backed by the library."""

    def __init__(self, library: Library | str | Path | None = None) -> None:
        self.library = library if isinstance(library, Library) else Library(library)
        # Deriving H1 from 2.6M cached M1 bars takes ~20 ms; doing it on every
        # evaluation of a 4,000-point sweep would take 80 seconds of nothing.
        self._bars_cache: dict[tuple[str, str], LoadedBars] = {}

    # --- data -------------------------------------------------------------

    def load_data(
        self, *, name: str, path: str, timeframe: str = "",
        utc_offset_hours: float = 0.0, contract_size: float = 100.0,
        tick_value: float = 0.10, commission_per_lot_per_side: float = 3.5,
        default_spread: float = 0.30, stream: bool | None = None,
        base_timeframe: str = "", progress=None,
    ) -> dict[str, Any]:
        """Read a CSV, build bars, and register it in the library.

        Tick files are folded into **base bars** once and cached; every coarser
        timeframe is then derived from that cache in milliseconds, so a large
        archive is read once ever rather than once per timeframe anyone becomes
        curious about (§4.6).

        `base_timeframe` chooses how fine that cache is. M1 is the default and
        is small. **S1 is the one to pick for a tick archive**: this engine fills
        on bars, so S1 underneath an M1 strategy is what settles which of the
        stop and the target was hit first — the question a single M1 bar cannot
        answer and which the engine otherwise resolves pessimistically. It costs
        roughly sixty times the bars.
        """
        src = Path(path).expanduser()
        if not src.exists():
            raise FileNotFoundError(
                f"{src} does not exist. Paths are read on the machine running "
                "this server — a Windows path only works if the server runs on "
                "Windows."
            )
        size = src.stat().st_size
        use_stream = stream if stream is not None else size > STREAM_ABOVE_BYTES
        spec = SourceSpec(utc_offset_hours=utc_offset_hours)
        t0 = time.perf_counter()

        if use_stream:
            if not timeframe:
                timeframe = "M15"
            base = _base_for(timeframe, base_timeframe)
            out = stream_bars(src, base, spec, progress=progress)
            bars_dict, quality = out.bars, out.quality
            kind, tick_size, rows, ingest = out.kind, out.tick_size, quality.rows, "stream"
        else:
            md = read_csv(src, spec)
            if md.kind == "tick" and not timeframe:
                raise ValueError(
                    "a tick source needs a timeframe to build bars from — pass "
                    "e.g. timeframe='M15'"
                )
            base = (_base_for(timeframe, base_timeframe) if md.kind == "tick"
                    else _name_for_ms(md.resolution_ms))
            if md.kind == "bar" and timeframe and timeframe_ms(timeframe) < md.resolution_ms:
                raise ValueError(
                    f"source is {base} bars and cannot serve {timeframe.upper()} — "
                    "bars roll up, never split (§4.6)"
                )
            bars_dict = build_bars(md, base)
            quality = md.quality
            kind, tick_size, rows, ingest = md.kind, md.tick_size, len(md), "whole"
            if not timeframe:
                timeframe = base

        instrument = {
            "contract_size": contract_size,
            "tick_size": tick_size,              # detected from the file, §4.2
            "tick_value": tick_value,
            "commission_per_lot_per_side": commission_per_lot_per_side,
            "default_spread": default_spread,
        }
        requested = (timeframe or base).upper()
        derived_bars = len(aggregate_bars(bars_dict, requested, base)["ms"])
        source = self.library.put_data_source(
            name=name, path=str(src), digest=quality.file_sha256, kind=kind,
            base_timeframe=base, default_timeframe=requested,
            default_bars=derived_bars, utc_offset_hours=utc_offset_hours, rows=rows,
            first_ms=quality.first_ms, last_ms=quality.last_ms,
            tick_size=tick_size, instrument=instrument,
            quality=_quality_dict(quality), bars=bars_dict, ingest=ingest,
        )
        self._bars_cache = {k: v for k, v in self._bars_cache.items() if k[0] != name}

        return {
            "name": name,
            "kind": kind,
            "rows": rows,
            "base_timeframe": base,
            "base_bars": source.bars,
            "timeframe": requested,
            "bars": derived_bars,
            "tick_size": tick_size,
            "utc_offset_hours": utc_offset_hours,
            "ingest": ingest,
            "seconds": round(time.perf_counter() - t0, 2),
            "span": _span(quality.first_ms, quality.last_ms),
            "quality": _quality_dict(quality),
            "quality_summary": quality.summary(),
            "instrument": instrument,
        }

    def list_data(self) -> list[dict[str, Any]]:
        return [
            {
                "name": d.name, "kind": d.kind, "rows": d.rows,
                "base_timeframe": d.base_timeframe, "base_bars": d.bars,
                "timeframe": d.default_timeframe,
                "bars": d.default_bars or d.bars,
                "tick_size": d.tick_size, "utc_offset_hours": d.utc_offset_hours,
                "span": _span(d.first_ms, d.last_ms),
                "span_days": round(d.span_days(), 1),
                "first_ms": d.first_ms, "last_ms": d.last_ms,
                "instrument": d.instrument, "ingest": d.ingest,
                "path": d.path, "quality": d.quality,
                "serves": [tf for tf, ms in TIMEFRAMES.items()
                           if ms >= timeframe_ms(d.base_timeframe)],
            }
            for d in self.library.list_data_sources()
        ]

    def bars_for(self, name: str, timeframe: str = "") -> LoadedBars:
        """Bars at `timeframe`, derived from the cached base bars if need be."""
        source = self.library.get_data_source(name)
        if source is None:
            known = [d.name for d in self.library.list_data_sources()]
            raise ValueError(
                f"unknown data source {name!r}; loaded: {known or 'none'}"
            )
        # The source's own loaded timeframe, never the cache base. Defaulting
        # to the base would report an M1 Sharpe for an M15 strategy — the same
        # trades scaled by √15 — silently, and in the flattering direction.
        want = (timeframe or source.default_timeframe or source.base_timeframe).upper()
        if want not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe {want!r}; known: {', '.join(TIMEFRAMES)}")
        if timeframe_ms(want) < timeframe_ms(source.base_timeframe):
            raise ValueError(
                f"{name} holds {source.base_timeframe} bars and cannot serve "
                f"{want} — bars roll up, never split (§4.6). Reload the source "
                f"with timeframe='{want}'."
            )

        key = (name, want)
        hit = self._bars_cache.get(key)
        if hit is not None:
            return hit

        raw = self.library.load_bars(name)
        if want != source.base_timeframe:
            raw = aggregate_bars(raw, want, source.base_timeframe)
        bars = Bars(
            ms=raw["ms"], open=raw["open"], high=raw["high"], low=raw["low"],
            close=raw["close"],
            spread=raw["spread"] if raw["spread"].any() else None,
        )
        loaded = LoadedBars(bars=bars, instrument=Instrument(**source.instrument),
                            timeframe=want, name=name)
        self._bars_cache[key] = loaded
        return loaded

    def intrabar_for(self, name: str, signal_timeframe: str,
                     intrabar: str) -> LoadedBars | None:
        """The finer series used to settle fills, or None.

        Refuses the two combinations that cannot mean anything: a series coarser
        than the signal timeframe (it would resolve nothing), and one finer than
        what the source actually cached (it would have to be invented).
        """
        if not intrabar:
            return None
        want = intrabar.upper()
        if want not in TIMEFRAMES or TIMEFRAMES[want] <= 0:
            raise ValueError(f"unknown intrabar timeframe {intrabar!r}")
        if timeframe_ms(want) >= timeframe_ms(signal_timeframe):
            raise ValueError(
                f"intrabar {want} is not finer than the signal timeframe "
                f"{signal_timeframe} — it would resolve nothing"
            )
        source = self.library.get_data_source(name)
        assert source is not None
        if timeframe_ms(want) < timeframe_ms(source.base_timeframe):
            raise ValueError(
                f"{name} caches {source.base_timeframe} bars, so {want} would "
                f"have to be invented. Reload it with base_timeframe='{want}'."
            )
        return self.bars_for(name, want)

    # --- strategies -------------------------------------------------------

    def save_strategy(
        self, *, strategy_id: str, description: str, signal_block: str,
        params: dict[str, Any] | None = None, parent_version: int | None = None,
        source_url: str = "", origin: str = "agent", family: str = "",
        tags: str = "",
    ) -> dict[str, Any]:
        result = validate_signal_block(signal_block)      # §9, before anything runs
        result.raise_if_invalid()
        version = self.library.save_version(
            strategy_id=strategy_id, description=description,
            signal_block=signal_block, params=params or {},
            parent_version=parent_version, source_url=source_url, origin=origin,
            family=family, tags=tags, warnings=result.warnings,
        )
        return {
            "strategy_id": strategy_id, "version": version.version,
            "parent_version": version.parent_version,
            "family": self.library.family_of(strategy_id),
            "warnings": list(result.warnings),
            "source_url": source_url,
        }

    def list_strategies(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        out = []
        for s in self.library.list_strategies(include_archived=include_archived):
            latest = self.library.get_version(s.strategy_id)
            out.append({
                "strategy_id": s.strategy_id, "name": s.name, "family": s.family,
                "tags": s.tags, "archived": s.archived,
                "archived_reason": s.archived_reason, "versions": s.versions,
                "runs": s.runs, "created_at": s.created_at,
                "last_run_at": s.last_run_at,
                "latest": {"version": latest.version,
                           "description": latest.description,
                           "params": latest.params,
                           "origin": latest.origin,
                           "source_url": latest.source_url},
                "comparisons": self.library.family_comparisons(s.family),
            })
        return out

    def get_strategy(self, strategy_id: str, version: int = 0) -> dict[str, Any]:
        v = self.library.get_version(strategy_id, version)
        return {
            "strategy_id": strategy_id, "version": v.version,
            "parent_version": v.parent_version, "description": v.description,
            "params": v.params, "signal_block": v.signal_block,
            "signal_sha": v.signal_sha[:12], "origin": v.origin,
            "source_url": v.source_url, "warnings": v.warnings,
            "created_at": v.created_at,
            "versions": [
                {"version": x.version, "parent": x.parent_version,
                 "description": x.description, "created_at": x.created_at}
                for x in self.library.list_versions(strategy_id)
            ],
            "tree": self.library.version_tree(strategy_id),
            "family": self.library.family_of(strategy_id),
            "comparisons": self.library.family_comparisons(
                self.library.family_of(strategy_id)),
        }

    def archive_strategy(self, strategy_id: str, reason: str = "",
                         archived: bool = True) -> dict[str, Any]:
        self.library.archive(strategy_id, reason, archived)
        return {"strategy_id": strategy_id, "archived": archived, "reason": reason}

    # --- runs -------------------------------------------------------------

    def backtest(
        self, *, strategy_id: str, data: str, version: int = 0,
        params: dict[str, Any] | None = None, timeframe: str = "",
        initial_capital: float = 10_000.0, risk_pct: float = 0.01,
        store_artifact: bool = True, intrabar: str = "",
    ) -> dict[str, Any]:
        """Backtest one version. `intrabar` names a finer timeframe (e.g. "S1")
        whose bars settle stop-versus-target order inside each signal bar."""
        loaded = self.bars_for(data, timeframe)
        sub = self.intrabar_for(data, loaded.timeframe, intrabar)
        v = self.library.get_version(strategy_id, version)
        merged = {**v.params, **(params or {})}
        source = self.library.get_data_source(data)
        assert source is not None

        run_id = self.library.start_run(
            version_pk=v.id, data_pk=source.id, kind="backtest", params=merged,
            timeframe=loaded.timeframe,
        )
        try:
            result = evaluate(
                v.signal_block, loaded.bars, merged, loaded.instrument,
                initial_capital, risk_pct,
                subbars=sub.bars if sub else None,
                subbar_timeframe=sub.timeframe if sub else "",
                bar_ms=timeframe_ms(loaded.timeframe) if sub else 0,
            )
            m = compute_metrics(result, loaded.timeframe, initial_capital)
        except Exception as exc:
            self.library.finish_run(run_id, status="error",
                                    error=f"{type(exc).__name__}: {exc}")
            raise

        reconcile = abs(result.equity[-1] - initial_capital - float(result.pnl.sum()))
        metrics = _finite(m.as_dict())
        artifact = None
        if store_artifact:
            artifact = self.library.artifacts.put_arrays({
                "equity": result.equity.astype(np.float64),
                "entry_i": result.entry_i.astype(np.int64),
                "exit_i": result.exit_i.astype(np.int64),
                "entry_px": result.entry_px.astype(np.float64),
                "exit_px": result.exit_px.astype(np.float64),
                "direction": result.direction.astype(np.int64),
                "pnl": result.pnl.astype(np.float64),
                "exit_reason": result.exit_reason.astype(np.int64),
                "lots": result.lots.astype(np.float64),
            })
        self.library.finish_run(run_id, metrics=metrics, artifact_sha=artifact)

        return {
            "run_id": run_id,
            "strategy": f"{strategy_id} v{v.version}",
            "strategy_id": strategy_id, "version": v.version,
            "data": data, "timeframe": loaded.timeframe, "bars": len(loaded.bars),
            "params": merged,
            "metrics": metrics,
            "ambiguous_exits": result.ambiguous_exits,
            "ambiguity_rate": round(result.ambiguity_rate(), 4),
            "rejected_zero_lots": result.rejected_zero_lots,
            "ledger_reconciles": bool(reconcile < 1e-6),
            "intrabar": result.intrabar_timeframe,
            "resolved_intrabar": result.resolved_intrabar,
            "intrabar_resolution_rate": round(result.intrabar_resolution_rate(), 4),
            "artifact": artifact,
        }

    def optimize(
        self, *, strategy_id: str, data: str, param_grid: dict[str, list[float]],
        version: int = 0, fixed: dict[str, Any] | None = None,
        objective: str = "return_over_maxdd", min_trades: int = 30,
        timeframe: str = "", top_k: int = 10, initial_capital: float = 10_000.0,
    ) -> dict[str, Any]:
        loaded = self.bars_for(data, timeframe)
        v = self.library.get_version(strategy_id, version)
        source = self.library.get_data_source(data)
        assert source is not None
        spec = _param_spec(param_grid, {**v.params, **(fixed or {})})

        run_id = self.library.start_run(
            version_pk=v.id, data_pk=source.id, kind="optimize",
            params={"grid": param_grid, "fixed": fixed or {}},
            timeframe=loaded.timeframe,
        )
        t0 = time.perf_counter()
        try:
            res = grid_search(
                spec,
                lambda p: compute_metrics(
                    evaluate(v.signal_block, loaded.bars, p, loaded.instrument,
                             initial_capital),
                    loaded.timeframe, initial_capital),
                objective=objective, min_trades=min_trades,
            )
        except Exception as exc:
            self.library.finish_run(run_id, status="error",
                                    error=f"{type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - t0

        best = res.best
        if best is None:
            self.library.finish_run(run_id, status="error",
                                    error="no configuration was evaluated")
            raise ValueError("the sweep evaluated nothing — check param_grid")

        plateau = None
        try:
            # The peak alone is not the answer (§11): a sharp optimum surrounded
            # by bad neighbours is a fitting artifact, and that is exactly the
            # shape a grid search is best at finding.
            plateau = analyze_plateau(res.evaluations, best.params,
                                      swept=list(param_grid))
        except Exception:                                  # noqa: BLE001
            plateau = None                                 # advisory, never fatal

        rows = [
            {"params": e.params,
             "score": _num(e.score),
             "net_profit": _num(e.metrics.get("net_profit")),
             "max_drawdown": _num(e.metrics.get("max_drawdown")),
             "sharpe": _num(e.metrics.get("sharpe")),
             "profit_factor": _num(e.metrics.get("profit_factor")),
             "win_rate": _num(e.metrics.get("win_rate")),
             "n_trades": int(e.metrics.get("n_trades", 0))}
            for e in res.evaluations
        ]
        results_sha = self.library.artifacts.put_json(rows)
        family = self.library.family_of(strategy_id)

        self.library.finish_run(
            run_id,
            metrics={"best_score": _num(best.score), **_finite(best.metrics)},
            artifact_sha=results_sha,
        )
        self.library.record_optimization(
            run_id, mode="grid", objective=objective,
            budget={"grid": param_grid, "min_trades": min_trades},
            comparisons=res.comparisons, results_sha=results_sha,
            robustness=_num(plateau.robustness) if plateau else None,
            verdict=plateau.verdict() if plateau else "",
        )

        return {
            "run_id": run_id,
            "strategy": f"{strategy_id} v{v.version}",
            "strategy_id": strategy_id, "version": v.version,
            "data": data, "timeframe": loaded.timeframe,
            "objective": objective,
            "comparisons": res.comparisons,
            "family": family,
            "family_comparisons": self.library.family_comparisons(family),
            "seconds": round(elapsed, 2),
            "evaluated": len(rows),
            "note": "comparisons accumulate across the family for DSR (plan §11)",
            "best": {"params": best.params, "score": _num(best.score),
                     **{k: _num(v) for k, v in _finite(best.metrics).items()}},
            "plateau": None if plateau is None else {
                "robustness": _num(plateau.robustness),
                "verdict": plateau.verdict(),
                "peak": _num(plateau.score),
                "neighbour_mean": _num(plateau.neighbour_mean),
                "neighbour_min": _num(plateau.neighbour_min),
                "neighbours": int(plateau.n_neighbours),
            },
            "top": sorted(
                [r for r in rows if r["score"] is not None],
                key=lambda r: r["score"], reverse=True)[:top_k],
            "results_sha": results_sha,
        }

    def validate(
        self, *, strategy_id: str, data: str, param_grid: dict[str, list[float]],
        version: int = 0, train: int = 12_000, test: int = 4_000,
        timeframe: str = "", min_trades: int = 10,
        initial_capital: float = 10_000.0,
    ) -> dict[str, Any]:
        loaded = self.bars_for(data, timeframe)
        v = self.library.get_version(strategy_id, version)
        source = self.library.get_data_source(data)
        assert source is not None

        folds = rolling_folds(len(loaded.bars), train=train, test=test, step=test)
        if not folds:
            raise ValueError(
                f"not enough bars ({len(loaded.bars):,}) for train={train:,} "
                f"test={test:,} — needs at least {train + test:,}. Use a finer "
                "timeframe, a longer file, or smaller folds."
            )
        spec = _param_spec(param_grid, v.params)
        run_id = self.library.start_run(
            version_pk=v.id, data_pk=source.id, kind="validate",
            params={"grid": param_grid, "train": train, "test": test},
            timeframe=loaded.timeframe,
        )

        def search(idx, a, b):
            w = loaded.bars.slice(a, b)
            r = grid_search(
                spec,
                lambda p: compute_metrics(
                    evaluate(v.signal_block, w, p, loaded.instrument, initial_capital),
                    loaded.timeframe, initial_capital),
                min_trades=min_trades,
            )
            return r.best.params, r.best.score, r.comparisons

        def test_fold(p, a, b):
            w = loaded.bars.slice(a, b)
            m = compute_metrics(
                evaluate(v.signal_block, w, p, loaded.instrument, initial_capital),
                loaded.timeframe, initial_capital)
            score = m.net_profit / m.max_drawdown if m.max_drawdown > 0 else float("-inf")
            return score, m.as_dict()

        t0 = time.perf_counter()
        try:
            wf = walk_forward(folds, search, test_fold)
        except Exception as exc:
            self.library.finish_run(run_id, status="error",
                                    error=f"{type(exc).__name__}: {exc}")
            raise
        eff = wf.efficiency()

        per_fold = [
            {"fold": i, "params": wf.chosen_params[i],
             "train": _num(wf.train_scores[i]), "test": _num(wf.test_scores[i]),
             "trades": int(wf.test_metrics[i].get("n_trades", 0)),
             "net_profit": _num(wf.test_metrics[i].get("net_profit")),
             "max_drawdown": _num(wf.test_metrics[i].get("max_drawdown")),
             "bars": [int(folds[i].train_start), int(folds[i].train_end),
                      int(folds[i].test_start), int(folds[i].test_end)]}
            for i in range(len(folds))
        ]
        positive = sum(1 for f in per_fold if (f["test"] or 0) > 0)
        verdict = _walk_forward_verdict(eff, positive, len(folds))

        self.library.finish_run(
            run_id,
            metrics={"walk_forward_efficiency": _num(eff),
                     "mean_oos_score": _num(wf.mean_test_score),
                     "folds": len(folds), "positive_folds": positive},
            artifact_sha=self.library.artifacts.put_json(per_fold),
        )
        self.library.record_validation(
            run_id, mode="walk_forward", folds=len(folds), efficiency=eff,
            verdict=verdict, comparisons=wf.total_comparisons, per_fold=per_fold,
        )

        return {
            "run_id": run_id,
            "strategy": f"{strategy_id} v{v.version}",
            "strategy_id": strategy_id, "version": v.version,
            "data": data, "timeframe": loaded.timeframe,
            "folds": len(folds),
            "total_comparisons": wf.total_comparisons,
            "mean_oos_score": _num(wf.mean_test_score),
            "walk_forward_efficiency": _num(eff),
            "positive_folds": positive,
            "verdict": verdict,
            "seconds": round(time.perf_counter() - t0, 2),
            "per_fold": per_fold,
        }

    def control_test(
        self, *, strategy_id: str, data: str, version: int = 0,
        params: dict[str, Any] | None = None, timeframe: str = "",
        trials: int = 5, seed: int = 0, initial_capital: float = 10_000.0,
        risk_pct: float = 0.01,
    ) -> dict[str, Any]:
        """Re-run the strategy on shuffled returns. Plan §13.

        The single cheapest way to tell a real edge from a bug. Shuffling the
        bar-to-bar log returns keeps the distribution — same volatility, same
        fat tails, same costs, same number of bars — and destroys the **order**,
        which is the only thing any of these strategies can actually be reading.

        So a strategy that still makes money here is not reading the market. It
        is reading the future, or the engine is. Look-ahead through an
        off-by-one, an indicator that peeks, a fill that uses a price the trade
        could not have had: all of them survive the shuffle, and none of them
        survive being noticed.

        Expectancy in R is reported rather than net profit because sizing is
        fixed-fractional — profit compounds, so its headline is dominated by how
        many trades there were, while expectancy per trade is not.
        """
        loaded = self.bars_for(data, timeframe)
        v = self.library.get_version(strategy_id, version)
        merged = {**v.params, **(params or {})}
        b = loaded.bars

        real = compute_metrics(
            evaluate(v.signal_block, b, merged, loaded.instrument,
                     initial_capital, risk_pct),
            loaded.timeframe, initial_capital)

        rng = np.random.default_rng(seed)
        log_returns = np.diff(np.log(b.close))
        results = []
        for _ in range(max(1, int(trials))):
            close = b.close[0] * np.exp(
                np.concatenate(([0.0], np.cumsum(rng.permutation(log_returns)))))
            # Scale the whole bar so the wicks keep their proportions; the point
            # is to reorder the moves, not to invent a different bar shape.
            scale = close / b.close
            shuffled = Bars(ms=b.ms, open=b.open * scale, high=b.high * scale,
                            low=b.low * scale, close=close, spread=b.spread)
            m = compute_metrics(
                evaluate(v.signal_block, shuffled, merged, loaded.instrument,
                         initial_capital, risk_pct),
                loaded.timeframe, initial_capital)
            results.append({"expectancy_r": _num(m.expectancy_r),
                            "n_trades": int(m.n_trades),
                            "profit_factor": _num(m.profit_factor),
                            "win_rate": _num(m.win_rate)})

        controls = [r["expectancy_r"] for r in results if r["expectancy_r"] is not None]
        control_mean = float(np.mean(controls)) if controls else float("nan")
        control_max = float(np.max(controls)) if controls else float("nan")
        edge = real.expectancy_r
        passed = bool(np.isfinite(edge) and controls and edge > control_max)

        return {
            "strategy": f"{strategy_id} v{v.version}",
            "data": data, "timeframe": loaded.timeframe, "trials": len(results),
            "real": {"expectancy_r": _num(edge), "n_trades": int(real.n_trades),
                     "profit_factor": _num(real.profit_factor),
                     "win_rate": _num(real.win_rate)},
            "controls": results,
            "control_mean_expectancy_r": _num(control_mean),
            "control_max_expectancy_r": _num(control_max),
            "passed": passed,
            "verdict": (
                "the edge is in the ordering of the data — it does not survive "
                "shuffling, which is what a real edge should do"
                if passed else
                "WARNING: the strategy performs as well on shuffled returns as on "
                "real ones. Order carries no information here, so this is "
                "look-ahead, a sizing artifact, or no edge at all — not a result."
            ),
        }

    # --- chart ------------------------------------------------------------

    def chart(
        self, *, data: str, strategy_id: str = "", version: int = 0,
        params: dict[str, Any] | None = None, timeframe: str = "",
        limit: int = 4_000, offset: int = 0, max_markers: int = 800,
        initial_capital: float = 10_000.0, risk_pct: float = 0.01,
    ) -> dict[str, Any]:
        """Bars for the chart, plus a run's own trades drawn on them. §14.4.

        The markers come from the engine's actual trade list, not from
        re-deriving entries in the browser — so what is drawn is exactly what
        was measured, including the trades the fill model rejected for zero
        lots and the ones that exited on an ambiguous bar.
        """
        loaded = self.bars_for(data, timeframe)
        n = len(loaded.bars)
        end = n - offset
        start = max(0, end - limit)
        if end <= start:
            start, end = 0, min(n, limit)

        ms = loaded.bars.ms[start:end]
        payload: dict[str, Any] = {
            "data": data,
            "timeframe": loaded.timeframe,
            "total_bars": n,
            "offset": offset,
            "start": start,
            "end": end,
            "bars": {
                # Seconds: lightweight-charts takes UNIX seconds, and sending
                # milliseconds silently plots everything in the year 55000.
                "time": (ms // 1000).astype(np.int64).tolist(),
                "open": _round(loaded.bars.open[start:end]),
                "high": _round(loaded.bars.high[start:end]),
                "low": _round(loaded.bars.low[start:end]),
                "close": _round(loaded.bars.close[start:end]),
            },
            "markers": [], "trades": [], "equity": [], "metrics": None,
        }
        if not strategy_id:
            return payload

        v = self.library.get_version(strategy_id, version)
        merged = {**v.params, **(params or {})}
        result = evaluate(v.signal_block, loaded.bars, merged, loaded.instrument,
                          initial_capital, risk_pct)
        m = compute_metrics(result, loaded.timeframe, initial_capital)

        in_window = [
            k for k in range(result.n_trades)
            if start <= int(result.entry_i[k]) < end or start <= int(result.exit_i[k]) < end
        ]
        step = max(1, len(in_window) // max_markers) if in_window else 1
        shown = in_window[::step][:max_markers]

        trades = [
            {
                "i": k,
                "entry_i": int(result.entry_i[k]), "exit_i": int(result.exit_i[k]),
                "entry_ms": int(loaded.bars.ms[result.entry_i[k]]),
                "exit_ms": int(loaded.bars.ms[result.exit_i[k]]),
                "entry_time": int(loaded.bars.ms[result.entry_i[k]] // 1000),
                "exit_time": int(loaded.bars.ms[result.exit_i[k]] // 1000),
                "side": "long" if result.direction[k] > 0 else "short",
                "entry": round(float(result.entry_px[k]), 5),
                "exit": round(float(result.exit_px[k]), 5),
                "pnl": round(float(result.pnl[k]), 2),
                "lots": round(float(result.lots[k]), 4),
                "reason": _EXIT_NAMES.get(int(result.exit_reason[k]), "?"),
            }
            for k in shown
        ]
        equity_step = max(1, (end - start) // 2_000)
        payload.update({
            "strategy": f"{strategy_id} v{v.version}",
            "strategy_id": strategy_id, "version": v.version, "params": merged,
            "trades": trades,
            "shown_trades": len(trades),
            "total_trades": result.n_trades,
            "markers": _markers(trades),
            "equity": [
                {"time": int(loaded.bars.ms[i] // 1000),
                 "value": round(float(result.equity[i]), 2)}
                for i in range(start, end, equity_step)
            ],
            "metrics": _finite(m.as_dict()),
        })
        return payload

    # --- history ----------------------------------------------------------

    def run_history(self, *, strategy_id: str = "", kind: str = "",
                    limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "run_id": r.id, "strategy_id": r.strategy_id, "version": r.version,
                "kind": r.kind, "data": r.data_name, "timeframe": r.timeframe,
                "status": r.status, "params": r.params, "metrics": r.metrics,
                "error": r.error, "started_at": r.started_at,
                "seconds": round(r.seconds, 2) if r.seconds else None,
            }
            for r in self.library.list_runs(strategy_id=strategy_id or None,
                                            kind=kind or None, limit=limit)
        ]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        r = self.library.get_run(run_id)
        if r is None:
            return None
        out = {
            "run_id": r.id, "strategy_id": r.strategy_id, "version": r.version,
            "kind": r.kind, "data": r.data_name, "timeframe": r.timeframe,
            "status": r.status, "params": r.params, "metrics": r.metrics,
            "error": r.error, "started_at": r.started_at,
            "finished_at": r.finished_at,
            "seconds": round(r.seconds, 2) if r.seconds else None,
            **r.extra,
        }
        if r.artifact_sha and r.kind == "optimize":
            out["results"] = self.library.artifacts.get_json(r.artifact_sha)
        elif r.artifact_sha and r.kind == "validate":
            out["per_fold"] = self.library.artifacts.get_json(r.artifact_sha)
        return out

    def stats(self) -> dict[str, Any]:
        return self.library.stats()


# --- helpers -----------------------------------------------------------------

_EXIT_NAMES = {0: "stop", 1: "target", 2: "signal", 3: "eod"}


def _markers(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chart markers, one per entry and one per exit."""
    out = []
    for t in trades:
        out.append({
            "time": t["entry_time"], "position": "belowBar" if t["side"] == "long" else "aboveBar",
            "shape": "arrowUp" if t["side"] == "long" else "arrowDown",
            "color": "#4ade80" if t["side"] == "long" else "#f87171",
            "text": f"{t['side'][0].upper()} {t['entry']}", "kind": "entry", "trade": t["i"],
        })
        out.append({
            "time": t["exit_time"], "position": "aboveBar" if t["side"] == "long" else "belowBar",
            "shape": "circle",
            "color": "#22d3ee" if t["pnl"] >= 0 else "#fb923c",
            "text": f"{t['reason']} {t['pnl']:+.2f}", "kind": "exit", "trade": t["i"],
        })
    out.sort(key=lambda m: m["time"])
    return out


def _param_spec(param_grid: dict[str, list[float]],
                fixed: dict[str, Any]) -> ParamSpec:
    """`{"fast": [5, 30, 5]}` -> the sweep spec, int or float by how it is written."""
    params: dict[str, Any] = {}
    for name, bounds in param_grid.items():
        if len(bounds) != 3:
            raise ValueError(
                f"param_grid[{name!r}] must be [low, high, step], got {bounds!r}"
            )
        lo, hi, step = (float(x) for x in bounds)
        if step <= 0:
            raise ValueError(f"param_grid[{name!r}] step must be positive, got {step}")
        if hi < lo:
            raise ValueError(f"param_grid[{name!r}] high {hi} is below low {lo}")
        if lo.is_integer() and hi.is_integer() and step.is_integer():
            params[name] = IntParam(int(lo), int(hi), int(step))
        else:
            params[name] = FloatParam(lo, hi, step)
    if not params:
        raise ValueError("param_grid is empty — nothing to sweep")
    return ParamSpec(params=params, fixed=fixed)


def _base_for(timeframe: str, requested_base: str) -> str:
    """How fine to cache the bars (§4.6). Rolling up is free; splitting is not.

    An explicit base wins, as long as it is at least as fine as the timeframe
    that will be run on it — asking to cache H1 and then run M1 is the one
    combination that cannot work.
    """
    want = timeframe_ms(timeframe) if timeframe else TIMEFRAMES["M1"]
    if requested_base:
        base_ms = timeframe_ms(requested_base)
        if base_ms > want:
            raise ValueError(
                f"base_timeframe {requested_base.upper()} is coarser than "
                f"{timeframe.upper()} and cannot serve it — bars roll up, never "
                "split (§4.6)"
            )
        return requested_base.upper()
    return timeframe.upper() if want < TIMEFRAMES["M1"] else "M1"


def _name_for_ms(ms: int) -> str:
    for name, value in TIMEFRAMES.items():
        if value == ms:
            return name
    return "M1"


def _walk_forward_verdict(efficiency: float, positive: int, folds: int) -> str:
    """Plain words for two numbers that mean different things. §12.

    Efficiency (out-of-sample score over in-sample score) is **biased low by
    construction**: the in-sample number is the best of every configuration the
    search tried, while the out-of-sample number is that one configuration's
    single draw. Picking the max of 256 noisy scores inflates the numerator of
    the ratio's denominator, so even a genuine edge rarely shows 1.0.

    The share of folds that were profitable out-of-sample carries the other half
    of the answer, and it is the half that is not inflated. Twenty-five folds out
    of twenty-five in profit at efficiency 0.3 is a real, consistent, smaller-
    than-advertised edge. Calling that "weak" on the ratio alone would throw away
    the more trustworthy of the two measurements.
    """
    if not np.isfinite(efficiency):
        return "no verdict — folds produced no comparable score"
    share = positive / folds if folds else 0.0
    if efficiency >= 0.5 and share >= 0.8:
        return "generalized — out-of-sample held up in size and in sign"
    if share >= 0.8:
        return ("consistent but decayed — every fold profitable out-of-sample, "
                "at a fraction of the in-sample size; expect the smaller number")
    if efficiency >= 0.4 and share >= 0.6:
        return "partial — most folds held up, with real decay; worth another look"
    if efficiency <= 0.0 or share < 0.4:
        return "did not generalize — the train folds were fitted"
    return "weak — most of the edge was in-sample"


def _finite(metrics: dict[str, Any]) -> dict[str, Any]:
    """JSON has no NaN or Infinity. Send null and let the UI say 'n/a'."""
    return {k: _num(v) if isinstance(v, float) else v for k, v in metrics.items()}


def _num(value: Any) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return round(f, 6) if np.isfinite(f) else None


def _round(arr: np.ndarray, places: int = 5) -> list[float]:
    return np.round(arr, places).tolist()


def _span(first_ms: int, last_ms: int) -> str:
    import datetime as dt
    if not first_ms and not last_ms:
        return ""
    fmt = "%Y-%m-%d"
    a = dt.datetime.fromtimestamp(first_ms / 1000, dt.timezone.utc).strftime(fmt)
    b = dt.datetime.fromtimestamp(last_ms / 1000, dt.timezone.utc).strftime(fmt)
    return f"{a} .. {b}"


def _quality_dict(quality) -> dict[str, Any]:
    from dataclasses import asdict
    d = asdict(quality)
    d["summary"] = quality.summary()
    return d
