"""The agent and the UI must agree, exactly. Plan §14, §15.

The failure this prevents is the quiet kind: a strategy the agent measured at
Sharpe 1.4 showing 0.9 in the browser, no error anywhere, and no way to tell
which one is real. Two code paths computing "the same" number will diverge —
not immediately, but on the tenth change to one of them.

So there is one implementation (`engine/service.py`) and these tests assert that
the MCP tools and the HTTP endpoints are adapters over it rather than
reimplementations: same strategy, same span, same numbers, field by field.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from engine.service import Quantor

BLOCK = '''
def signal(bars, p):
    fast = ema_fast(bars.close, p["fast"])
    slow = ema_fast(bars.close, p["slow"])
    n = len(bars.close)
    above = fast > slow
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    up[1:] = above[1:] & ~above[:-1]
    dn[1:] = ~above[1:] & above[:-1]
    warm = np.isnan(fast) | np.isnan(slow)
    up &= ~warm
    dn &= ~warm
    stop = atr_fast(bars.high, bars.low, bars.close, p["atr"])
    stop = np.where(np.isnan(stop) | (stop <= 0), 1.0, stop) * 2.0
    return {"long_entry": up, "short_entry": dn,
            "stop_distance": stop, "target_distance": stop * 1.5}
'''

PARAMS = {"fast": 12, "slow": 40, "atr": 14}
GRID = {"fast": [6, 18, 6], "slow": [30, 60, 15]}


def _tick_csv(path, n: int = 240_000, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    ms = 1_609_722_000_000 + np.cumsum(rng.integers(1, 700, n).astype(np.int64))
    bid = 1900.0 + np.cumsum(rng.normal(0, 0.02, n))
    ask = bid + rng.uniform(0.10, 0.40, n)
    stamps = np.datetime_as_string(
        np.datetime64(0, "ms") + ms.astype("timedelta64[ms]"), unit="ms")
    with open(path, "w") as fh:
        fh.write("timestamp,bidPrice,askPrice\n")
        fh.write("\n".join(f"{s.replace('T', ' ')},{b:.3f},{a:.3f}"
                           for s, b, a in zip(stamps, bid, ask)) + "\n")


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "ticks.csv"
    _tick_csv(path)
    return str(path)


@pytest.fixture()
def surfaces(tmp_path, csv_path, monkeypatch):
    """One library, reached three ways: service, MCP tools, HTTP endpoints."""
    monkeypatch.setenv("QUANTOR_LIBRARY", str(tmp_path))

    import importlib
    from fastapi.testclient import TestClient

    import app.api.main as api_module
    import quantor_mcp.server as mcp_module
    importlib.reload(mcp_module)
    importlib.reload(api_module)

    service = Quantor(tmp_path)
    service.load_data(name="d", path=csv_path, timeframe="M15", utc_offset_hours=3.0)
    service.save_strategy(strategy_id="s", description="probe",
                          signal_block=BLOCK, params=PARAMS, family="probe")
    return service, mcp_module, TestClient(api_module.app)


# --- the same numbers, whichever door you come in by -------------------------

def test_backtest_is_identical_through_mcp_and_http(surfaces):
    service, mcp, http = surfaces

    direct = service.backtest(strategy_id="s", data="d")
    via_mcp = json.loads(mcp.backtest_run(strategy_id="s", data="d"))
    via_http = http.post("/api/backtest", json={"strategy_id": "s", "data": "d"}).json()

    assert "error" not in via_mcp, via_mcp
    for key, value in direct["metrics"].items():
        assert via_mcp["metrics"][key] == value, f"MCP differs on {key}"
        assert via_http["metrics"][key] == value, f"HTTP differs on {key}"

    for key in ("timeframe", "bars", "ledger_reconciles", "ambiguity_rate",
                "rejected_zero_lots", "params"):
        assert via_mcp[key] == direct[key], key
        assert via_http[key] == direct[key], key


def test_the_timeframe_is_the_one_the_source_was_loaded_at(surfaces):
    """Defaulting to the M1 cache would report a Sharpe wrong by √15."""
    service, mcp, http = surfaces
    assert service.backtest(strategy_id="s", data="d")["timeframe"] == "M15"
    assert json.loads(mcp.backtest_run(strategy_id="s", data="d"))["timeframe"] == "M15"
    assert http.post("/api/backtest",
                     json={"strategy_id": "s", "data": "d"}).json()["timeframe"] == "M15"


def test_a_different_timeframe_changes_the_annualized_metrics(surfaces):
    service, _, _ = surfaces
    m15 = service.backtest(strategy_id="s", data="d", timeframe="M15")
    m5 = service.backtest(strategy_id="s", data="d", timeframe="M5")
    assert m15["metrics"]["n_trades"] != m5["metrics"]["n_trades"]
    assert m15["timeframe"] == "M15" and m5["timeframe"] == "M5"


def test_optimize_is_identical_through_mcp_and_http(surfaces):
    service, mcp, http = surfaces
    direct = service.optimize(strategy_id="s", data="d", param_grid=GRID, min_trades=1)
    via_mcp = json.loads(mcp.optimize_run(strategy_id="s", data="d",
                                          param_grid=GRID, min_trades=1))
    via_http = http.post("/api/optimize", json={
        "strategy_id": "s", "data": "d", "param_grid": GRID, "min_trades": 1}).json()

    assert via_mcp["comparisons"] == direct["comparisons"] == via_http["comparisons"]
    assert via_mcp["best"]["params"] == direct["best"]["params"]
    assert via_http["best"]["params"] == direct["best"]["params"]
    assert via_mcp["best"]["score"] == direct["best"]["score"]


def test_the_chart_payload_matches_the_backtest_it_draws(surfaces):
    """§14.4 — the markers must be the engine's own trades, not re-derived."""
    service, _, http = surfaces
    bt = service.backtest(strategy_id="s", data="d")
    chart = http.get("/api/chart", params={
        "data": "d", "strategy_id": "s", "limit": 100_000}).json()

    assert chart["total_trades"] == bt["metrics"]["n_trades"]
    for key, value in bt["metrics"].items():
        assert chart["metrics"][key] == value, f"chart differs on {key}"


def test_the_chart_sends_seconds_not_milliseconds(surfaces):
    """lightweight-charts takes UNIX seconds; ms plots in the year 55000."""
    _, _, http = surfaces
    chart = http.get("/api/chart", params={"data": "d", "limit": 50}).json()
    first = chart["bars"]["time"][0]
    assert 1_000_000_000 < first < 4_000_000_000, f"{first} is not UNIX seconds"
    assert len(chart["bars"]["time"]) == len(chart["bars"]["close"]) == 50


def test_control_test_is_identical_through_mcp_and_http(surfaces):
    service, mcp, http = surfaces
    direct = service.control_test(strategy_id="s", data="d", trials=2, seed=3)
    via_mcp = json.loads(mcp.control_test(strategy_id="s", data="d", trials=2, seed=3))
    via_http = http.post("/api/control", json={
        "strategy_id": "s", "data": "d", "trials": 2, "seed": 3}).json()

    assert via_mcp["real"] == direct["real"] == via_http["real"]
    assert via_mcp["controls"] == direct["controls"]
    assert via_http["passed"] == direct["passed"]


# --- what the surfaces record --------------------------------------------------

def test_every_surface_writes_to_the_same_library(surfaces):
    service, mcp, http = surfaces
    before = service.stats()["runs"]

    mcp.backtest_run(strategy_id="s", data="d")
    http.post("/api/backtest", json={"strategy_id": "s", "data": "d"})
    service.backtest(strategy_id="s", data="d")

    assert service.stats()["runs"] == before + 3
    kinds = {r["kind"] for r in service.run_history(limit=5)}
    assert kinds == {"backtest"}


def test_comparison_counts_accumulate_across_surfaces(surfaces):
    """§12 — the DSR denominator does not reset because you used a different door."""
    service, mcp, http = surfaces
    service.optimize(strategy_id="s", data="d", param_grid=GRID, min_trades=1)
    first = service.library.family_comparisons("probe")

    mcp.optimize_run(strategy_id="s", data="d", param_grid=GRID, min_trades=1)
    http.post("/api/optimize", json={"strategy_id": "s", "data": "d",
                                     "param_grid": GRID, "min_trades": 1})
    assert service.library.family_comparisons("probe") == first * 3


# --- errors reach the caller ---------------------------------------------------

def test_mcp_returns_the_error_text_rather_than_swallowing_it(surfaces):
    """Probe 9: an agent cannot fix a mistake it cannot see."""
    _, mcp, _ = surfaces
    out = json.loads(mcp.backtest_run(strategy_id="nope", data="d"))
    assert "unknown strategy" in out["error"]
    assert "'s'" in out["error"], "the message must name what does exist"
    assert out["tool"] == "backtest_run"


def test_http_returns_the_error_text_with_a_4xx(surfaces):
    _, _, http = surfaces
    res = http.post("/api/backtest", json={"strategy_id": "nope", "data": "d"})
    assert res.status_code == 400
    assert "unknown strategy" in res.json()["error"]


def test_a_windows_path_says_why_it_cannot_be_read(surfaces):
    _, _, http = surfaces
    res = http.post("/api/data", json={
        "name": "x", "path": r"C:\Users\HP\Documents\xau.csv", "timeframe": "M15"})
    assert res.status_code == 404
    assert "machine running this server" in res.json()["error"]


def test_asking_for_a_finer_timeframe_than_the_cache_is_refused(surfaces, tmp_path):
    """§4.6 — bars roll up, never split. The refusal has to name the fix."""
    service, _, _ = surfaces
    bars = tmp_path / "h1.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    for i in range(200):
        t = dt.datetime.fromtimestamp(1_609_722_000 + i * 3600, dt.timezone.utc)
        rows.append(f"{t:%Y-%m-%d %H:%M:%S},1900.0,1901.0,1899.0,1900.5,100")
    bars.write_text("\n".join(rows) + "\n")

    service.load_data(name="hourly", path=str(bars))
    assert service.bars_for("hourly").timeframe == "H1"

    with pytest.raises(ValueError, match="never split"):
        service.bars_for("hourly", "M5")


def test_rolling_a_bar_source_up_still_works(surfaces, tmp_path):
    service, _, _ = surfaces
    bars = tmp_path / "h1b.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    for i in range(200):
        t = dt.datetime.fromtimestamp(1_609_722_000 + i * 3600, dt.timezone.utc)
        rows.append(f"{t:%Y-%m-%d %H:%M:%S},1900.0,1901.0,1899.0,1900.5,100")
    bars.write_text("\n".join(rows) + "\n")

    service.load_data(name="hourly2", path=str(bars))
    h4 = service.bars_for("hourly2", "H4")
    assert h4.timeframe == "H4"
    assert len(h4.bars) == pytest.approx(200 / 4, abs=2)
