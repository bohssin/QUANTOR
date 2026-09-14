"""The library has to outlive the process. Plan §16.

Before this existed, everything lived in two dicts in the MCP server: close the
session and every version, run and rejection was gone. These tests are written
against that failure specifically — most of them close the library and open a
new one before asserting anything, because "it is still in memory" is not the
claim being made.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.library import Artifacts, Library

BLOCK = (
    'def signal(bars, p):\n'
    '    f = ema_fast(bars.close, p["fast"])\n'
    '    return {"long_entry": f > 0, "short_entry": f < 0,\n'
    '            "stop_distance": f, "target_distance": f}\n'
)


def _bars(n: int = 100) -> dict[str, np.ndarray]:
    ms = 1_609_722_000_000 + np.arange(n, dtype=np.int64) * 900_000
    close = 1900.0 + np.arange(n, dtype=np.float64)
    return {"ms": ms, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": np.full(n, 10, np.int64),
            "spread": np.full(n, 0.3), "spread_max": np.full(n, 0.4)}


def _source(lib: Library, name: str = "xau") -> int:
    bars = _bars()
    return lib.put_data_source(
        name=name, path="/tmp/xau.csv", digest="fp:abc", kind="tick",
        base_timeframe="M1", default_timeframe="M15", default_bars=7,
        utc_offset_hours=3.0,
        rows=1_000_000,
        first_ms=int(bars["ms"][0]), last_ms=int(bars["ms"][-1]), tick_size=0.001,
        instrument={"contract_size": 100.0, "tick_value": 0.10},
        quality={"rows": 1_000_000, "out_of_order": 0}, bars=bars, ingest="stream",
    ).id


# --- the actual point: survival ----------------------------------------------

def test_a_strategy_survives_closing_and_reopening_the_library(tmp_path):
    lib = Library(tmp_path)
    lib.save_version(strategy_id="ema-cross", description="first",
                     signal_block=BLOCK, params={"fast": 10})
    lib.close()

    reopened = Library(tmp_path)
    v = reopened.get_version("ema-cross")
    assert v.version == 1
    assert v.params == {"fast": 10}
    assert v.signal_block == BLOCK


def test_runs_and_metrics_survive_a_restart(tmp_path):
    lib = Library(tmp_path)
    data_pk = _source(lib)
    v = lib.save_version(strategy_id="ema-cross", description="first",
                         signal_block=BLOCK, params={"fast": 10})
    run_id = lib.start_run(version_pk=v.id, data_pk=data_pk, kind="backtest",
                           params={"fast": 10}, timeframe="M15")
    lib.finish_run(run_id, metrics={"net_profit": 1234.5, "n_trades": 42})
    lib.close()

    runs = Library(tmp_path).list_runs(strategy_id="ema-cross")
    assert len(runs) == 1
    assert runs[0].metrics["net_profit"] == 1234.5
    assert runs[0].status == "ok"
    assert runs[0].data_name == "xau"


def test_cached_bars_survive_so_a_huge_file_is_read_once(tmp_path):
    """The restart is only useful if it does not mean re-reading 11 GB."""
    lib = Library(tmp_path)
    _source(lib)
    lib.close()

    bars = Library(tmp_path).load_bars("xau")
    np.testing.assert_array_equal(bars["ms"], _bars()["ms"])
    np.testing.assert_allclose(bars["close"], _bars()["close"])


# --- versions form a tree, not a list ----------------------------------------

def test_versions_default_to_branching_from_the_latest(tmp_path):
    lib = Library(tmp_path)
    for i in range(3):
        lib.save_version(strategy_id="s", description=f"v{i}", signal_block=BLOCK)
    versions = lib.list_versions("s")
    assert [v.version for v in versions] == [1, 2, 3]
    assert [v.parent_version for v in versions] == [None, 1, 2]


def test_a_version_can_branch_from_an_older_one(tmp_path):
    """'Go back to Tuesday's and try again' — what a list cannot express."""
    lib = Library(tmp_path)
    for i in range(3):
        lib.save_version(strategy_id="s", description=f"v{i}", signal_block=BLOCK)
    branched = lib.save_version(strategy_id="s", description="from v1",
                                signal_block=BLOCK, parent_version=1)
    assert branched.version == 4
    assert branched.parent_version == 1

    roots = lib.version_tree("s")
    assert len(roots) == 1
    children = {c["version"] for c in roots[0]["children"]}
    assert children == {2, 4}, "v4 must hang off v1, beside v2"


def test_branching_from_a_version_that_does_not_exist_is_refused(tmp_path):
    lib = Library(tmp_path)
    lib.save_version(strategy_id="s", description="v1", signal_block=BLOCK)
    with pytest.raises(ValueError, match="cannot branch"):
        lib.save_version(strategy_id="s", description="bad", signal_block=BLOCK,
                         parent_version=9)


def test_saving_never_overwrites(tmp_path):
    lib = Library(tmp_path)
    lib.save_version(strategy_id="s", description="one", signal_block="A")
    lib.save_version(strategy_id="s", description="two", signal_block="B")
    assert lib.get_version("s", 1).signal_block == "A"
    assert lib.get_version("s", 2).signal_block == "B"
    assert lib.get_version("s", 0).signal_block == "B", "0 means latest"


# --- research integrity ------------------------------------------------------

def test_comparison_counts_accumulate_across_the_family(tmp_path):
    """§12: the deflated Sharpe divides by everything tried, not one sweep."""
    lib = Library(tmp_path)
    data_pk = _source(lib)
    for name in ("ema-cross-a", "ema-cross-b"):
        v = lib.save_version(strategy_id=name, description="x", signal_block=BLOCK,
                             family="ema-cross")
        run_id = lib.start_run(version_pk=v.id, data_pk=data_pk, kind="optimize",
                               params={}, timeframe="M15")
        lib.finish_run(run_id, metrics={})
        lib.record_optimization(run_id, mode="grid", objective="sharpe",
                                budget={}, comparisons=250)

    assert lib.family_comparisons("ema-cross") == 500
    lib.close()
    assert Library(tmp_path).family_comparisons("ema-cross") == 500, "must persist"


def test_an_unrelated_family_does_not_inflate_the_count(tmp_path):
    lib = Library(tmp_path)
    data_pk = _source(lib)
    for name, fam in (("a", "one"), ("b", "two")):
        v = lib.save_version(strategy_id=name, description="x", signal_block=BLOCK,
                             family=fam)
        rid = lib.start_run(version_pk=v.id, data_pk=data_pk, kind="optimize",
                            params={}, timeframe="M15")
        lib.finish_run(rid, metrics={})
        lib.record_optimization(rid, mode="grid", objective="sharpe", budget={},
                                comparisons=100)
    assert lib.family_comparisons("one") == 100
    assert lib.family_comparisons("two") == 100


def test_archiving_hides_a_strategy_without_losing_its_runs(tmp_path):
    """§16.4: rejections are the most informative rows in the library."""
    lib = Library(tmp_path)
    data_pk = _source(lib)
    v = lib.save_version(strategy_id="doomed", description="x", signal_block=BLOCK)
    rid = lib.start_run(version_pk=v.id, data_pk=data_pk, kind="validate",
                        params={}, timeframe="M15")
    lib.finish_run(rid, metrics={"walk_forward_efficiency": 0.1})
    lib.archive("doomed", reason="walk-forward efficiency 0.1")
    lib.close()

    reopened = Library(tmp_path)
    assert [s.strategy_id for s in reopened.list_strategies()] == []
    archived = reopened.list_strategies(include_archived=True)
    assert archived[0].archived_reason == "walk-forward efficiency 0.1"
    assert reopened.list_runs(strategy_id="doomed")[0].metrics[
        "walk_forward_efficiency"] == 0.1


def test_source_url_is_kept_per_version_for_attribution(tmp_path):
    """The LuxAlgo Library is free WITH attribution, and it attaches per work."""
    lib = Library(tmp_path)
    url = "https://www.luxalgo.com/library/concept/supertrend/"
    lib.save_version(strategy_id="st", description="from the library",
                     signal_block=BLOCK, source_url=url, origin="library")
    lib.close()
    v = Library(tmp_path).get_version("st")
    assert v.source_url == url
    assert v.origin == "library"


# --- errors say what to do ---------------------------------------------------

def test_an_unknown_strategy_names_the_ones_that_exist(tmp_path):
    lib = Library(tmp_path)
    lib.save_version(strategy_id="real", description="x", signal_block=BLOCK)
    with pytest.raises(ValueError, match="real"):
        lib.get_version("typo")


def test_an_unknown_data_source_names_the_ones_that_exist(tmp_path):
    lib = Library(tmp_path)
    _source(lib, "xau")
    with pytest.raises(ValueError, match="xau"):
        lib.load_bars("eur")


def test_asking_for_a_version_out_of_range_says_the_range(tmp_path):
    lib = Library(tmp_path)
    lib.save_version(strategy_id="s", description="x", signal_block=BLOCK)
    with pytest.raises(ValueError, match="versions 1..1"):
        lib.get_version("s", 7)


# --- artifacts ---------------------------------------------------------------

def test_identical_artifacts_are_stored_once(tmp_path):
    art = Artifacts(tmp_path / "a")
    one = art.put_json({"equity": [1, 2, 3]})
    two = art.put_json({"equity": [1, 2, 3]})
    assert one == two
    assert art.size()[0] == 1


def test_arrays_round_trip_exactly(tmp_path):
    art = Artifacts(tmp_path / "a")
    arrays = {"ms": np.arange(5, dtype=np.int64), "close": np.linspace(1, 2, 5)}
    got = art.get_arrays(art.put_arrays(arrays))
    np.testing.assert_array_equal(got["ms"], arrays["ms"])
    np.testing.assert_array_equal(got["close"], arrays["close"])


def test_the_same_arrays_always_produce_the_same_digest(tmp_path):
    """Content addressing is worthless if the bytes carry a timestamp."""
    art = Artifacts(tmp_path / "a")
    arrays = {"close": np.linspace(1, 2, 100)}
    assert art.put_arrays(arrays) == art.put_arrays(dict(arrays))


def test_a_missing_artifact_says_which_one(tmp_path):
    art = Artifacts(tmp_path / "a")
    with pytest.raises(FileNotFoundError, match="abcdef"):
        art.get_bytes("abcdef" + "0" * 58)


# --- concurrency -------------------------------------------------------------

def test_two_connections_can_write_at_once(tmp_path):
    """The MCP server and the API server both write this file."""
    a, b = Library(tmp_path), Library(tmp_path)
    a.save_version(strategy_id="from-mcp", description="x", signal_block=BLOCK)
    b.save_version(strategy_id="from-api", description="x", signal_block=BLOCK)
    names = {s.strategy_id for s in Library(tmp_path).list_strategies()}
    assert names == {"from-mcp", "from-api"}


def test_writes_from_many_threads_all_land(tmp_path):
    import threading
    lib = Library(tmp_path)
    errors: list[Exception] = []

    def save(i: int) -> None:
        try:
            lib.save_version(strategy_id=f"s{i}", description="x", signal_block=BLOCK)
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes failed: {errors[:3]}"
    assert len(lib.list_strategies()) == 12


def test_stats_counts_what_is_there(tmp_path):
    lib = Library(tmp_path)
    data_pk = _source(lib)
    v = lib.save_version(strategy_id="s", description="x", signal_block=BLOCK)
    rid = lib.start_run(version_pk=v.id, data_pk=data_pk, kind="backtest",
                        params={}, timeframe="M15")
    lib.finish_run(rid, metrics={"n_trades": 3})
    s = lib.stats()
    assert s["strategies"] == 1 and s["versions"] == 1
    assert s["runs"] == 1 and s["backtests"] == 1
    assert s["data_sources"] == 1 and s["artifacts"] >= 1
