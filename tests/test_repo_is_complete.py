"""A clone must contain the whole program.

This exists because of a real failure: `.gitignore` carried `library/` to hide
the runtime library directory, and an unanchored pattern matches a directory of
that name at **any** depth. It silently swallowed `engine/library/` — the entire
persistence package — and `tests/library/`. Everything kept working locally,
because the files were on disk. A fresh clone installed cleanly, passed nine of
the doctor's checks, and died at `No module named 'engine.library'`.

Nothing in the test suite could catch that, because the test suite ran against
the working tree rather than against what was committed. So this module asks git
directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Directories whose contents are generated, downloaded or per-machine, and are
#: correctly absent from a clone.
NOT_SHIPPED = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".git", "node_modules",
    "library", "data", "runs", "_data",
}

#: Files that are deliberately per-machine.
NOT_SHIPPED_FILES = {".mcp.json", ".server.pid", ".server.log"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def _tracked() -> set[str]:
    return {line for line in _git("ls-files").splitlines() if line}


def _is_shipped(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in NOT_SHIPPED for part in rel.parts):
        return False
    # tools/pinets_oracle is a dev-only AGPL fixture generator (see its README).
    return "pinets_oracle" not in rel.parts or rel.suffix in (".md", ".mjs", ".json")


def _source_files(suffixes: tuple[str, ...]) -> list[Path]:
    return [p for p in ROOT.rglob("*")
            if p.is_file() and p.suffix in suffixes and _is_shipped(p)]


@pytest.mark.parametrize("suffixes,label", [
    ((".py",), "Python"),
    ((".js", ".html", ".css", ".mjs"), "frontend"),
    ((".sql", ".sh", ".ps1", ".cmd"), "schema and scripts"),
])
def test_every_source_file_is_committed(suffixes, label):
    tracked = _tracked()
    missing = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in _source_files(suffixes)
        if str(p.relative_to(ROOT)).replace("\\", "/") not in tracked
    )
    assert not missing, (
        f"{len(missing)} {label} file(s) exist on disk but are not in the "
        f"repository — a clone would be missing them:\n  "
        + "\n  ".join(missing)
        + "\n\nCheck .gitignore: an unanchored pattern like `library/` matches "
          "that directory name at any depth. Anchor it as `/library/`."
    )


def test_every_python_package_ships_its_init():
    """A package directory without __init__.py is not importable from a clone."""
    tracked = _tracked()
    problems = []
    for init in ROOT.rglob("__init__.py"):
        if not _is_shipped(init):
            continue
        rel = str(init.relative_to(ROOT)).replace("\\", "/")
        if rel not in tracked:
            problems.append(rel)
    assert not problems, f"package markers missing from the repo: {problems}"


def test_the_packages_the_engine_imports_are_all_present():
    """Name them explicitly, so a rename cannot quietly drop one."""
    tracked = _tracked()
    required = [
        "engine/__init__.py",
        "engine/service.py",
        "engine/library/__init__.py",
        "engine/library/store.py",
        "engine/library/artifacts.py",
        "engine/library/schema.sql",
        "engine/store/stream.py",
        "engine/backtest/core.py",
        "app/api/main.py",
        "app/ui/index.html",
        "app/ui/vendor/lightweight-charts.standalone.production.mjs",
        "quantor_mcp/server.py",
        "quantor_mcp/doctor.py",
    ]
    missing = [f for f in required if f not in tracked]
    assert not missing, f"a clone would not contain: {missing}"


def test_gitignore_anchors_its_directory_patterns():
    """An unanchored directory pattern is the bug this module was written for."""
    risky = []
    for raw in (ROOT / ".gitignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if not line.endswith("/") or line.startswith("/") or "/" in line[:-1]:
            continue
        # A bare `name/` matches at every depth. Fine for build droppings that
        # are always junk; not fine for a word that could name a source package.
        if line.rstrip("/") not in {"__pycache__", "node_modules", ".pytest_cache",
                                    ".venv", "venv"}:
            risky.append(line)
    assert not risky, (
        "these .gitignore patterns match a directory of that name at ANY depth, "
        f"which can swallow source directories: {risky}. Anchor them with a "
        "leading slash, e.g. `/library/`."
    )


# --- Windows packaging ---------------------------------------------------------

def test_powershell_scripts_with_non_ascii_start_with_a_utf8_bom():
    """PowerShell 5.1 reads a BOM-less .ps1 as ANSI, mangling non-ASCII.

    `powershell.exe` — still the default on Windows — decodes a script without a
    byte-order mark using the current ANSI code page. An accented character in
    the file is therefore corrupted at PARSE time, which no amount of
    `[Console]::OutputEncoding` can undo afterwards. Setting only the output
    encoding is what shipped first, and the path printed for the user to copy
    came out as `TÃ©lÃ©chargements`.
    """
    bom = b"\xef\xbb\xbf"
    problems = []
    for script in ROOT.rglob("*.ps1"):
        if not _is_shipped(script):
            continue
        raw = script.read_bytes()
        text = raw.decode("utf-8-sig")
        if any(ord(c) > 127 for c in text) and not raw.startswith(bom):
            problems.append(str(script.relative_to(ROOT)))
    assert not problems, (
        f"these PowerShell scripts contain non-ASCII but have no UTF-8 BOM, so "
        f"Windows PowerShell will mangle it: {problems}"
    )


def test_the_windows_entry_points_exist_and_call_the_real_scripts():
    """cmd.exe does not execute .ps1 files; typing one silently does nothing."""
    for wrapper, target in (("setup.cmd", "setup.ps1"), ("run.cmd", "run.ps1")):
        path = ROOT / wrapper
        assert path.exists(), f"{wrapper} is missing — cmd.exe users have no entry point"
        body = path.read_text()
        assert target in body, f"{wrapper} does not invoke {target}"
        assert "ExecutionPolicy Bypass" in body, (
            f"{wrapper} will be blocked by the default execution policy")


def test_nothing_advertises_a_port_the_app_does_not_serve():
    """The doctor told people to open :8000 for a release after the move to 2026."""
    from quantor_mcp.doctor import DEFAULT_PORT

    stale = []
    for path in [ROOT / "quantor_mcp" / "doctor.py", ROOT / "scripts" / "run.sh",
                 ROOT / "scripts" / "run.ps1", ROOT / "scripts" / "serverctl.sh",
                 ROOT / "README.md"]:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if "8000" in line and str(DEFAULT_PORT) not in line:
                stale.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:70]}")
    assert not stale, (
        f"these lines still advertise port 8000; the app serves {DEFAULT_PORT}:\n  "
        + "\n  ".join(stale))
