"""The app's own wiring: restart, the sidebar socket, and the static UI.

The first of these is the whole reason the library exists. "It is still there
after a restart" cannot be tested by checking a variable — it needs a second
process, or at minimum a second object built from nothing but the files on
disk. So these tests throw the first one away before they assert anything.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json

import numpy as np
import pytest

from engine.service import Quantor

# Note the idiom: build the full-length boolean first, then slice the LOCAL.
# The §9 validator rejects `bars.close[1:]` outright — the pattern is only
# sometimes look-ahead, and a validator that has to tell which is a validator
# that will one day be wrong in the expensive direction.
BLOCK = '''
def signal(bars, p):
    fast = ema_fast(bars.close, p["fast"])
    n = len(bars.close)
    above = bars.close > fast
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    up[1:] = above[1:] & ~above[:-1]
    dn[1:] = ~above[1:] & above[:-1]
    warm = np.isnan(fast)
    up &= ~warm
    dn &= ~warm
    stop = np.full(n, 1.5)
    return {"long_entry": up, "short_entry": dn,
            "stop_distance": stop, "target_distance": stop * 1.5}
'''


def _bars_csv(path, n: int = 4_000) -> None:
    rng = np.random.default_rng(11)
    close = 1900 + np.cumsum(rng.normal(0, 0.8, n))
    open_ = np.concatenate(([1900.0], close[:-1]))
    wick = np.abs(rng.normal(0, 0.5, n))
    with open(path, "w") as fh:
        fh.write("timestamp,open,high,low,close,volume\n")
        for i in range(n):
            t = dt.datetime.fromtimestamp(1_609_722_000 + i * 900, dt.timezone.utc)
            fh.write(f"{t:%Y-%m-%d %H:%M:%S},{open_[i]:.3f},"
                     f"{max(open_[i], close[i]) + wick[i]:.3f},"
                     f"{min(open_[i], close[i]) - wick[i]:.3f},{close[i]:.3f},100\n")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTOR_LIBRARY", str(tmp_path))
    from fastapi.testclient import TestClient
    import app.api.main as api
    importlib.reload(api)
    return TestClient(api.app), tmp_path


# --- the point of the library --------------------------------------------------

def test_everything_survives_rebuilding_the_service_from_disk(tmp_path, monkeypatch):
    csv = tmp_path / "bars.csv"
    _bars_csv(csv)

    first = Quantor(tmp_path / "lib")
    first.load_data(name="d", path=str(csv))
    first.save_strategy(strategy_id="s", description="v1", signal_block=BLOCK,
                        params={"fast": 20}, family="fam")
    run = first.backtest(strategy_id="s", data="d")
    first.optimize(strategy_id="s", data="d", param_grid={"fast": [10, 30, 10]},
                   min_trades=1)
    first.library.close()
    del first

    # Nothing in memory carries over; only the files on disk.
    second = Quantor(tmp_path / "lib")

    assert [d["name"] for d in second.list_data()] == ["d"]
    assert [s["strategy_id"] for s in second.list_strategies()] == ["s"]
    assert second.get_strategy("s")["signal_block"] == BLOCK
    assert second.library.family_comparisons("fam") == 3

    # And the run is not merely listed — it is reopenable, with its numbers.
    again = second.get_run(run["run_id"])
    assert again["metrics"] == run["metrics"]
    assert again["kind"] == "backtest"

    # A backtest on the reopened library reproduces the original exactly: the
    # cached bars are the same bars, not a re-read that drifted.
    assert second.backtest(strategy_id="s", data="d")["metrics"] == run["metrics"]


def test_a_second_process_sees_what_the_first_wrote(tmp_path):
    """The MCP server and the API server are two processes on one file."""
    csv = tmp_path / "bars.csv"
    _bars_csv(csv)

    writer = Quantor(tmp_path / "lib")
    writer.load_data(name="d", path=str(csv))
    writer.save_strategy(strategy_id="s", description="v1", signal_block=BLOCK,
                         params={"fast": 20})

    reader = Quantor(tmp_path / "lib")          # a separate connection
    assert reader.get_strategy("s")["version"] == 1

    writer.save_strategy(strategy_id="s", description="v2", signal_block=BLOCK,
                         params={"fast": 25})
    assert reader.get_strategy("s")["version"] == 2, "the reader must see new writes"


# --- HTTP surface ---------------------------------------------------------------

def test_health_reports_what_the_app_can_do(client):
    http, root = client
    body = http.get("/api/health").json()
    assert body["ok"] is True
    assert body["library"] == str(root)
    assert "M15" in body["timeframes"]
    assert isinstance(body["agent_available"], bool)


def test_the_ui_is_served(client):
    http, _ = client
    index = http.get("/")
    assert index.status_code == 200
    assert "QUANTOR" in index.text

    for asset in ("/styles.css", "/app.js", "/api.js", "/agent.js",
                  "/pages/strategies.js", "/pages/chart.js",
                  "/vendor/lightweight-charts.standalone.production.mjs"):
        res = http.get(asset)
        assert res.status_code == 200, asset
        assert res.content, f"{asset} is empty"


def test_the_chart_library_is_vendored_not_fetched(client):
    """A CDN outage must not break a local research tool."""
    http, _ = client
    body = http.get("/vendor/lightweight-charts.standalone.production.mjs").text
    assert "createChart" in body and "CandlestickSeries" in body


def test_run_history_is_empty_before_anything_runs(client):
    http, _ = client
    assert http.get("/api/runs").json() == []
    assert http.get("/api/strategies").json() == []
    assert http.get("/api/data").json() == []


def test_a_missing_run_is_a_404(client):
    http, _ = client
    assert http.get("/api/runs/9999").status_code == 404


def test_an_empty_param_grid_is_refused_with_a_reason(client, tmp_path):
    http, root = client
    csv = root / "bars.csv"
    _bars_csv(csv, 500)
    http.post("/api/data", json={"name": "d", "path": str(csv)})
    http.post("/api/strategies", json={
        "strategy_id": "s", "signal_block": BLOCK, "params": {"fast": 20}})

    res = http.post("/api/optimize", json={
        "strategy_id": "s", "data": "d", "param_grid": {}})
    assert res.status_code == 400
    assert "nothing to sweep" in res.json()["error"]


def test_a_malformed_grid_says_what_shape_it_wanted(client, tmp_path):
    http, root = client
    csv = root / "bars.csv"
    _bars_csv(csv, 500)
    http.post("/api/data", json={"name": "d", "path": str(csv)})
    http.post("/api/strategies", json={
        "strategy_id": "s", "signal_block": BLOCK, "params": {"fast": 20}})

    res = http.post("/api/optimize", json={
        "strategy_id": "s", "data": "d", "param_grid": {"fast": [10, 30]}})
    assert res.status_code == 400
    assert "[low, high, step]" in res.json()["error"]


# --- the agent sidebar ----------------------------------------------------------

def test_the_sidebar_socket_accepts_a_connection(client):
    """The socket must answer, whether or not the CLI is installed.

    Not a test that Claude replies — that needs a subscription and 70 seconds.
    It is a test that the endpoint exists, accepts a message, and reports a
    missing CLI as a readable error rather than closing silently.
    """
    http, _ = client
    with http.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "message", "text": "hello"})
        first = ws.receive_json()
        assert first["type"] in ("start", "error")
        if first["type"] == "error":
            assert "claude" in first["text"].lower()
