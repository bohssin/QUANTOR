"""Check a local QUANTOR install and say exactly what to fix. Plan §14.

    python3 quantor_mcp/doctor.py

Written because the failure mode is uninformative by default: when the MCP
server cannot start, Claude Code reports `CONNECTION_CLOSED` and nothing else —
no traceback, no missing-module name, nothing pointing at the cause. Every check
here ends in a command you can paste.

Imports are deliberately lazy and guarded: this script has to run usefully on
the interpreter that is *missing* things, which is the whole point.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["mcp", "numpy", "numba", "pyarrow", "pandas"]
REQUIRED_APP = ["fastapi", "uvicorn"]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = RESET = ""

_results: list[tuple[str, bool]] = []


def report(name: str, ok: bool | None, detail: str = "", fix: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else (f"{YELLOW}WARN{RESET}" if ok is None
                                             else f"{RED}FAIL{RESET}")
    print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if fix and not ok:
        for line in fix.strip().splitlines():
            print(f"         {line}")
    if ok is not None:
        _results.append((name, ok))
    return bool(ok)


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 11)
    return report("Python >= 3.11", ok, f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
                  "Use a newer interpreter, or create a venv with one:\n"
                  "  python3.12 -m venv .venv")


def check_packages() -> bool:
    import importlib.util
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    return report(
        "engine + MCP packages", not missing,
        "all present" if not missing else f"missing: {', '.join(missing)}",
        f"{sys.executable} -m pip install {' '.join(missing)}\n"
        "This is the usual cause of CONNECTION_CLOSED: the interpreter named in\n"
        ".mcp.json must be the one with these installed.",
    )


def check_app_packages() -> bool | None:
    """The UI needs two more packages. Optional: the MCP path works without."""
    import importlib.util
    missing = [m for m in REQUIRED_APP if importlib.util.find_spec(m) is None]
    return report(
        "app packages (optional)", None if missing else True,
        "fastapi + uvicorn present" if not missing else f"missing: {', '.join(missing)}",
        f"{sys.executable} -m pip install fastapi 'uvicorn[standard]'\n"
        "Only needed for the browser UI (./scripts/run.sh). The agent works "
        "through MCP without them.",
    )


def check_library() -> bool:
    """The library is what makes anything survive a restart (§16)."""
    try:
        from engine.library import Library
        root = os.environ.get("QUANTOR_LIBRARY") or str(ROOT / "library")
        lib = Library(root)
        stats = lib.stats()
    except Exception as exc:
        return report("library (persistence)", False, f"{type(exc).__name__}: {exc}",
                      "If the schema version changed, move the old library aside:\n"
                      f"  mv {ROOT / 'library'} {ROOT / 'library.old'}")
    return report(
        "library (persistence)", True,
        f"{stats['strategies']} strategies, {stats['versions']} versions, "
        f"{stats['runs']} runs, {stats['data_sources']} sources, "
        f"{stats['artifacts']} artifacts at {stats['root']}",
    )


def check_ui_assets() -> bool:
    """A missing vendored chart library serves 200 and renders nothing."""
    ui = ROOT / "app" / "ui"
    needed = [
        "index.html", "styles.css", "app.js", "api.js", "agent.js",
        "pages/strategies.js", "pages/chart.js", "pages/data.js",
        "pages/history.js",
        "vendor/lightweight-charts.standalone.production.mjs",
    ]
    missing = [f for f in needed if not (ui / f).exists()]
    return report(
        "UI assets", not missing,
        f"{len(needed)} files present" if not missing else f"missing: {', '.join(missing)}",
        "The chart library is vendored, not fetched at runtime. Restore it with:\n"
        "  npm pack lightweight-charts@5.0.9 && tar xzf lightweight-charts-*.tgz\n"
        "  cp package/dist/lightweight-charts.standalone.production.mjs app/ui/vendor/",
    )


def check_engine() -> bool:
    sys.path.insert(0, str(ROOT))
    try:
        from engine.backtest import run_backtest          # noqa: F401
        from engine.indicators import ema_fast            # noqa: F401
        from engine.store import read_csv                 # noqa: F401
    except Exception as exc:
        return report("engine imports", False, f"{type(exc).__name__}: {exc}",
                      f"Run from the repo root, or add it to PYTHONPATH:\n"
                      f"  export PYTHONPATH={ROOT}")
    return report("engine imports", True, str(ROOT))


def check_numba() -> bool:
    """numba compiles on first call; a broken toolchain surfaces only then."""
    try:
        import numpy as np
        from engine.indicators import ema_fast
        out = ema_fast(np.arange(100, dtype=np.float64), 10)
        ok = bool(np.isfinite(out[-1]))
    except Exception as exc:
        return report("numba JIT compiles", False, f"{type(exc).__name__}: {str(exc)[:90]}",
                      "Try clearing the cache:  find . -name __pycache__ -exec rm -rf {} +")
    return report("numba JIT compiles", ok, "ema_fast returned finite values")


def check_server_handshake() -> bool:
    """The real test: start the server and complete an MCP initialize over stdio."""
    req = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "doctor", "version": "0"}},
    })
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "quantor_mcp" / "server.py")],
            input=req + "\n", capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return report("MCP server handshake", False, "timed out after 120s",
                      "Startup should take ~1s. Check for an import that blocks.")
    if '"result"' not in proc.stdout:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else "no output"
        return report("MCP server handshake", False, tail[:110],
                      "The server exited instead of answering. Whatever is printed\n"
                      "above is what the MCP client hides behind CONNECTION_CLOSED.")
    return report("MCP server handshake", True, "initialize answered over stdio")


def check_mcp_config() -> bool:
    path = ROOT / ".mcp.json"
    if not path.exists():
        return report(".mcp.json", False, "not found",
                      "python3 quantor_mcp/doctor.py --write-config")
    try:
        cfg = json.loads(path.read_text())
    except Exception as exc:
        return report(".mcp.json", False, f"invalid JSON: {exc}", "Fix or regenerate it.")

    entry = cfg.get("mcpServers", {}).get("quantor-engine", {})
    args = entry.get("args", [])
    script = next((a for a in args if a.endswith("server.py")), "")
    problems = []
    if not script:
        problems.append("no server.py in args")
    elif not Path(script).is_absolute():
        problems.append("path is not absolute")
    elif not Path(script).exists():
        problems.append(f"path does not exist: {script}")
    interp = entry.get("command", "")
    if interp and shutil.which(interp) is None and not Path(interp).exists():
        problems.append(f"interpreter not found: {interp}")

    return report(".mcp.json", not problems,
                  "quantor-engine configured" if not problems else "; ".join(problems),
                  "python3 quantor_mcp/doctor.py --write-config\n"
                  "Writes a correct config using THIS interpreter and absolute paths.")


def check_luxalgo() -> bool | None:
    if shutil.which("npx") is None:
        return report("LuxAlgo MCP (optional)", None, "npx not found",
                      "Install Node 18+ to use the Library and Edge Stats tools.")
    # Deliberately does not invoke the package: `npx -y` re-resolves it from the
    # registry on every call, which took 180s+ in testing. Presence of npx plus
    # the sign-in reminder is all this check can usefully assert.
    token = _luxalgo_token_path()
    signed_in = token is not None and token.exists()
    return report(
        "LuxAlgo MCP (optional)", None,
        f"npx found; signed in: {'yes' if signed_in else 'no'}"
        + (f" ({token})" if signed_in else ""),
        "npx -y @luxalgo/mcp login\n"
        "library_* needs a signed-in LuxAlgo account (free, not a paid plan).\n"
        "The token is per-machine, so sign in on the machine running the server.\n"
        "Concepts are also free as markdown with no sign-in at all:\n"
        "  https://www.luxalgo.com/library/concept/<slug>.md",
    )


def _luxalgo_token_path() -> Path | None:
    """Where `@luxalgo/mcp login` stores its token, per its own source."""
    override = os.environ.get("LUXALGO_MCP_AUTH_FILE")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return Path(base) / "luxalgo" / "mcp-auth.json" if base else None
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "luxalgo" / "mcp-auth.json"


def check_end_to_end() -> bool:
    """One real backtest through the MCP tool functions."""
    import tempfile
    import datetime as dt
    try:
        import numpy as np
        # A scratch library, so `doctor` never leaves a probe strategy in the
        # owner's real one. Persistence is checked separately, above.
        os.environ["QUANTOR_LIBRARY"] = tempfile.mkdtemp(prefix="quantor-doctor-")
        import importlib
        from quantor_mcp import server as S
        importlib.reload(S)

        n = 3000
        rng = np.random.default_rng(1)
        close = 1900 + np.cumsum(rng.normal(0, 0.9, n))
        open_ = np.concatenate(([1900.0], close[:-1]))
        wick = np.abs(rng.normal(0, 0.6, n))
        path = Path(tempfile.mkdtemp()) / "probe.csv"
        with path.open("w") as fh:
            fh.write("timestamp,open,high,low,close\n")
            for i in range(n):
                t = dt.datetime.fromtimestamp(1609722000 + i * 900, dt.timezone.utc)
                fh.write(f"{t:%Y-%m-%d %H:%M:%S},{open_[i]:.3f},"
                         f"{max(open_[i], close[i]) + wick[i]:.3f},"
                         f"{min(open_[i], close[i]) - wick[i]:.3f},{close[i]:.3f}\n")

        loaded = S.data_load(name="_doctor", path=str(path))
        if "error" in loaded:
            return report("end-to-end backtest", False, loaded[:110])
        S.strategy_save(
            strategy_id="_doctor", description="doctor probe",
            signal_block=(
                'def signal(bars, p):\n'
                '    f = ema_fast(bars.close, p["fast"])\n'
                '    s = ema_fast(bars.close, p["slow"])\n'
                '    a = f > s\n'
                '    up = np.zeros(len(bars.close), bool)\n'
                '    dn = np.zeros(len(bars.close), bool)\n'
                '    up[1:] = a[1:] & ~a[:-1]\n'
                '    dn[1:] = ~a[1:] & a[:-1]\n'
                '    w = np.isnan(f) | np.isnan(s)\n'
                '    up &= ~w\n'
                '    dn &= ~w\n'
                '    stop = np.full(len(bars.close), 2.0)\n'
                '    return {"long_entry": up, "short_entry": dn,\n'
                '            "stop_distance": stop, "target_distance": stop * 2}\n'
            ),
            params={"fast": 10, "slow": 30},
        )
        out = json.loads(S.backtest_run(strategy_id="_doctor", data="_doctor"))
    except Exception as exc:
        return report("end-to-end backtest", False, f"{type(exc).__name__}: {str(exc)[:90]}")

    if "error" in out:
        return report("end-to-end backtest", False, out["error"][:90])
    ok = out.get("ledger_reconciles") is True
    return report("end-to-end backtest", ok,
                  f"{out['metrics']['n_trades']} trades, ledger reconciles={ok}",
                  "The trade list and equity curve disagree — that is a real bug.")


def write_config() -> None:
    path = ROOT / ".mcp.json"
    cfg = {
        "mcpServers": {
            "quantor-engine": {
                "command": sys.executable,
                "args": [str(ROOT / "quantor_mcp" / "server.py")],
            },
            "luxalgo": {"command": "npx", "args": ["-y", "@luxalgo/mcp"]},
        }
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {path}")
    print(f"  interpreter: {sys.executable}")
    print(f"  server:      {ROOT / 'quantor_mcp' / 'server.py'}")


def main() -> int:
    if "--write-config" in sys.argv:
        write_config()
        return 0

    print(f"\nQUANTOR doctor — {ROOT}\n")
    check_python()
    have_pkgs = check_packages()
    check_app_packages()
    if have_pkgs:
        check_engine()
        check_numba()
        check_library()
        check_server_handshake()
        check_end_to_end()
    else:
        print(f"  {DIM}(skipping engine checks until the packages are installed){RESET}")
    check_ui_assets()
    check_mcp_config()
    check_luxalgo()

    failed = [n for n, ok in _results if not ok]
    print()
    if failed:
        print(f"{RED}{len(failed)} check(s) failed:{RESET} " + ", ".join(failed))
        print("Fix the first one and run again — later failures are usually downstream.\n")
        return 1
    print(f"{GREEN}All checks passed.{RESET} Start a session with:\n")
    print("  claude --mcp-config .mcp.json \\")
    print('    --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"\n')
    print("Or open the app:\n")
    print("  ./scripts/run.sh          # then http://127.0.0.1:8000\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
