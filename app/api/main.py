"""HTTP and WebSocket front end. Plan §15.

An adapter over `engine/service.py` — the same object the MCP server drives. No
engine logic lives here, deliberately: when the agent's numbers and the UI's
numbers come from two implementations, they drift, and the drift is invisible
until someone trades the wrong one.

    .venv/bin/uvicorn app.api.main:app --port 8000
    # then open http://127.0.0.1:8000

Local by default and bound to 127.0.0.1 by `scripts/run.sh`. There is no
authentication, because there is no remote: this serves one person's own
research on their own machine, and adding a login to a localhost tool is
security theatre. Do not expose the port.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.service import Quantor           # noqa: E402
from engine.store import TIMEFRAMES          # noqa: E402

from .discover import discover, sniff        # noqa: E402
from .jobs import JobRunner                  # noqa: E402

UI = ROOT / "app" / "ui"

app = FastAPI(title="QUANTOR", docs_url="/api/docs", openapi_url="/api/openapi.json")
QUANTOR = Quantor(os.environ.get("QUANTOR_LIBRARY") or None)
JOBS = JobRunner()


# --- request bodies ----------------------------------------------------------

class LoadData(BaseModel):
    name: str
    path: str
    timeframe: str = ""
    utc_offset_hours: float = 0.0
    contract_size: float = 100.0
    tick_value: float = 0.10
    commission_per_lot_per_side: float = 3.5
    default_spread: float = 0.30
    base_timeframe: str = ""


class SaveStrategy(BaseModel):
    strategy_id: str
    description: str = ""
    signal_block: str
    params: dict[str, Any] = Field(default_factory=dict)
    parent_version: int | None = None
    family: str = ""
    source_url: str = ""
    origin: str = "human"


class Backtest(BaseModel):
    strategy_id: str
    data: str
    version: int = 0
    params: dict[str, Any] | None = None
    timeframe: str = ""
    initial_capital: float = 10_000.0
    risk_pct: float = 0.01
    intrabar: str = ""


class Optimize(BaseModel):
    strategy_id: str
    data: str
    param_grid: dict[str, list[float]]
    version: int = 0
    fixed: dict[str, Any] | None = None
    objective: str = "return_over_maxdd"
    min_trades: int = 30
    timeframe: str = ""
    top_k: int = 20


class Validate(BaseModel):
    strategy_id: str
    data: str
    param_grid: dict[str, list[float]]
    version: int = 0
    train: int = 12_000
    test: int = 4_000
    timeframe: str = ""
    min_trades: int = 10


class Control(BaseModel):
    strategy_id: str
    data: str
    version: int = 0
    params: dict[str, Any] | None = None
    timeframe: str = ""
    trials: int = 5
    seed: int = 0


class Archive(BaseModel):
    reason: str = ""
    archived: bool = True


# --- errors ------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error(_request, exc: ValueError):
    """Send the message, not a bare 500.

    Same reasoning as the MCP server's error surfacing: the caller — a person
    reading a toast, or an agent reading JSON — can only fix what it can see.
    """
    return JSONResponse(status_code=400,
                        content={"error": f"{type(exc).__name__}: {exc}"})


@app.exception_handler(FileNotFoundError)
async def not_found(_request, exc: FileNotFoundError):
    return JSONResponse(status_code=404,
                        content={"error": f"{type(exc).__name__}: {exc}"})


def _guard(fn, *args, **kwargs):
    """Anything unexpected still arrives as a readable message."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, FileNotFoundError):
        raise
    except HTTPException:
        raise
    except Exception as exc:                               # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


# --- meta --------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "library": str(QUANTOR.library.root),
        "timeframes": list(TIMEFRAMES),
        "agent_available": shutil.which("claude") is not None,
        "python": sys.version.split()[0],
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return _guard(QUANTOR.stats)


# --- data --------------------------------------------------------------------

@app.get("/api/data")
def list_data() -> list[dict[str, Any]]:
    return _guard(QUANTOR.list_data)


@app.post("/api/data")
def load_data(body: LoadData) -> dict[str, Any]:
    return _guard(QUANTOR.load_data, **body.model_dump())


@app.get("/api/data/discover")
def discover_data(extra: str = "") -> list[dict[str, Any]]:
    """Data files on this machine, so nobody types a path."""
    return _guard(discover, [e for e in extra.split("|") if e])


@app.post("/api/data/start")
def start_load(body: LoadData) -> dict[str, Any]:
    """Begin a load and return a job to poll.

    The synchronous POST stays for scripts and tests. The UI uses this one,
    because folding an 11 GB archive into bars takes minutes and a browser will
    not wait that long — it times out and the work finishes for nobody.
    """
    args = body.model_dump()
    src = Path(args["path"]).expanduser()
    total = src.stat().st_size if src.exists() else 0

    def work(job):
        job.total_bytes = total
        job.message = "reading"

        def progress(rows: int, bars: int) -> None:
            job.rows, job.bars = rows, bars

        out = QUANTOR.load_data(**args, progress=progress)
        job.rows, job.bars = out["rows"], out["base_bars"]
        job.message = "done"
        return out

    try:
        job = JOBS.start("load", f"{body.name} <- {Path(body.path).name}", work)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return job.as_dict()


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return [j.as_dict() for j in JOBS.all()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job.as_dict()


@app.delete("/api/data/{name}")
def delete_data(name: str) -> dict[str, Any]:
    return {"deleted": _guard(QUANTOR.library.delete_data_source, name)}


# --- strategies --------------------------------------------------------------

@app.get("/api/strategies")
def list_strategies(include_archived: bool = False) -> list[dict[str, Any]]:
    return _guard(QUANTOR.list_strategies, include_archived=include_archived)


@app.post("/api/strategies")
def save_strategy(body: SaveStrategy) -> dict[str, Any]:
    return _guard(QUANTOR.save_strategy, **body.model_dump())


@app.get("/api/strategies/{strategy_id}")
def get_strategy(strategy_id: str, version: int = 0) -> dict[str, Any]:
    return _guard(QUANTOR.get_strategy, strategy_id, version)


@app.post("/api/strategies/{strategy_id}/archive")
def archive_strategy(strategy_id: str, body: Archive) -> dict[str, Any]:
    return _guard(QUANTOR.archive_strategy, strategy_id, body.reason, body.archived)


# --- runs --------------------------------------------------------------------

@app.post("/api/backtest")
def backtest(body: Backtest) -> dict[str, Any]:
    return _guard(QUANTOR.backtest, **body.model_dump())


@app.post("/api/optimize")
def optimize(body: Optimize) -> dict[str, Any]:
    return _guard(QUANTOR.optimize, **body.model_dump())


@app.post("/api/validate")
def validate(body: Validate) -> dict[str, Any]:
    return _guard(QUANTOR.validate, **body.model_dump())


@app.post("/api/control")
def control(body: Control) -> dict[str, Any]:
    return _guard(QUANTOR.control_test, **body.model_dump())


@app.get("/api/chart")
def chart(data: str, strategy_id: str = "", version: int = 0, timeframe: str = "",
          limit: int = 1_500, offset: int = 0, max_markers: int = 400,
          params: str = "") -> dict[str, Any]:
    parsed = json.loads(params) if params else None
    return _guard(QUANTOR.chart, data=data, strategy_id=strategy_id,
                  version=version, timeframe=timeframe, limit=limit,
                  offset=offset, max_markers=max_markers, params=parsed)


@app.get("/api/runs")
def runs(strategy_id: str = "", kind: str = "", limit: int = 50) -> list[dict[str, Any]]:
    return _guard(QUANTOR.run_history, strategy_id=strategy_id, kind=kind, limit=limit)


@app.get("/api/runs/{run_id}")
def run(run_id: int) -> dict[str, Any]:
    out = _guard(QUANTOR.get_run, run_id)
    if out is None:
        raise HTTPException(404, f"no run {run_id}")
    return out


# --- the agent sidebar -------------------------------------------------------

AGENT_TOOLS = (
    "mcp__quantor-engine__data_list,mcp__quantor-engine__data_load,"
    "mcp__quantor-engine__strategy_save,mcp__quantor-engine__strategy_list,"
    "mcp__quantor-engine__strategy_get,mcp__quantor-engine__strategy_archive,"
    "mcp__quantor-engine__backtest_run,mcp__quantor-engine__optimize_run,"
    "mcp__quantor-engine__validate_run,mcp__quantor-engine__control_test,"
    "mcp__quantor-engine__chart_apply,"
    "mcp__quantor-engine__run_history,mcp__quantor-engine__run_get,"
    "mcp__quantor-engine__library_stats,"
    "mcp__luxalgo__library_search,mcp__luxalgo__library_get_concept,"
    "mcp__luxalgo__library_get_indicator,mcp__luxalgo__library_get_source_code,"
    "mcp__luxalgo__library_list_concepts,mcp__luxalgo__library_list_families"
)

SYSTEM_APPEND = (
    "You are the research assistant inside QUANTOR, a local backtesting "
    "platform. Reach the engine ONLY through the quantor-engine MCP tools — "
    "never write your own backtest script, and never read the data files "
    "directly. Everything you save persists in the library and appears in the "
    "user's UI immediately. When you take an idea from the LuxAlgo Library, "
    "pass its URL as source_url: the licence is free with attribution. Report "
    "results honestly, including when a strategy does not generalize."
)


@app.websocket("/ws/agent")
async def agent(ws: WebSocket) -> None:
    """Relay a `claude -p` session to the sidebar. Plan §14.7.

    No API key anywhere: this drives the Claude Code CLI the owner is already
    signed into, which was the constraint from the start. Each message spawns
    one turn; `--resume` carries the conversation so the sidebar is a
    conversation rather than a series of unrelated questions.
    """
    await ws.accept()
    session_id: str | None = None
    process: asyncio.subprocess.Process | None = None

    if shutil.which("claude") is None:
        await ws.send_json({"type": "error", "text":
                            "The `claude` CLI is not on PATH. Install Claude Code "
                            "to use the sidebar; everything else works without it."})
        await ws.close()
        return

    try:
        while True:
            incoming = await ws.receive_json()
            if incoming.get("type") == "stop":
                if process and process.returncode is None:
                    process.terminate()
                continue
            prompt = (incoming.get("text") or "").strip()
            if not prompt:
                continue

            cmd = [
                "claude", "-p", prompt,
                "--output-format", "stream-json", "--verbose",
                "--include-partial-messages",
                "--mcp-config", str(ROOT / ".mcp.json"),
                "--allowedTools", AGENT_TOOLS,
                "--append-system-prompt", SYSTEM_APPEND,
            ]
            if session_id:
                cmd += ["--resume", session_id]

            await ws.send_json({"type": "start", "prompt": prompt})
            process = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "QUANTOR_LIBRARY": str(QUANTOR.library.root)},
            )
            assert process.stdout is not None
            async for raw in process.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "log", "text": line})
                    continue
                if event.get("session_id"):
                    session_id = event["session_id"]
                await ws.send_json({"type": "event", "event": event})

            stderr = (await process.stderr.read()).decode(errors="replace") if process.stderr else ""
            await ws.send_json({"type": "done", "code": process.returncode,
                                "stderr": stderr[-2000:] if process.returncode else ""})
    except WebSocketDisconnect:
        pass
    finally:
        if process and process.returncode is None:
            process.terminate()


# --- static ------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI / "index.html")


if UI.exists():
    app.mount("/", StaticFiles(directory=str(UI), html=True), name="ui")
