"""Find the user's data files so nobody has to type a path. Plan §15.

Typing `C:\\Users\\HP\\Documents\\Téléchargements MEGA\\xau.csv` into a form is a
step that can only go wrong: a typo, an accented directory the console mangles,
a backslash eaten by something in between. The machine already knows where its
own files are.

So the Data page asks for candidates and shows them as cards. This module looks
in the obvious places, reads the **first two lines only** — never the whole
file — and reports what it found: tick or bar, which columns, how big, and the
first timestamp so the clock can be sanity-checked before a fifteen-minute read.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

__all__ = ["discover", "sniff"]

#: Where people actually keep downloaded market data. Checked in this order.
SEARCH_DIRS = [
    "Documents", "Downloads", "Desktop", "data", "Data",
    "Documents/MEGA", "Documents/MEGAsync", "Downloads/MEGA",
]

#: Only files big enough to be market data. A 3 KB csv is somebody's shopping
#: list; scanning them all would turn a page load into a disk crawl.
MIN_BYTES = 1 << 20          # 1 MB

#: How deep to walk inside each search directory.
MAX_DEPTH = 3

#: Stop after this many, so a Downloads folder with 400 CSVs cannot hang the page.
MAX_RESULTS = 40

_TIME_HINTS = ("timestamp", "time", "datetime", "date", "gmt")
_TICK_HINTS = ("bid", "ask")
_BAR_HINTS = ("open", "high", "low", "close")


@dataclass
class Candidate:
    path: str
    name: str
    bytes: int
    size: str
    kind: str                    # tick | bar | unknown
    columns: list[str]
    first_row: str
    suggested_name: str
    suggested_base: str          # S1 for ticks, "" for bars
    suggested_timeframe: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def sniff(path: Path) -> Candidate | None:
    """Read the header and one data row. Never more — the file may be 11 GB."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < MIN_BYTES:
        return None

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            header_line = fh.readline()
            first = fh.readline().strip()
    except OSError:
        return None
    if not header_line:
        return None

    try:
        columns = next(csv.reader([header_line]))
    except Exception:                                       # noqa: BLE001
        return None
    columns = [c.strip() for c in columns]
    if not 2 <= len(columns) <= 24:
        return None

    lowered = [c.lower().replace(" ", "").replace("_", "") for c in columns]
    has_time = any(any(h in c for h in _TIME_HINTS) for c in lowered)
    if not has_time:
        return None
    has_tick = sum(any(h in c for h in _TICK_HINTS) for c in lowered) >= 2
    has_bar = sum(any(c == h or h in c for h in _BAR_HINTS) for c in lowered) >= 4

    kind = "tick" if has_tick and not has_bar else "bar" if has_bar else "unknown"
    if kind == "unknown":
        return None

    stem = path.stem.lower()
    suggested = "".join(ch for ch in stem if ch.isalnum() or ch in "-_")[:24] or "data"

    note = ""
    if kind == "tick" and size > (2 << 30):
        note = ("Large tick archive. S1 keeps every second, which is what settles "
                "stop-versus-target order inside an M1 bar, and costs roughly "
                "sixty times the bars of M1. If memory is tight, load M1 first.")
    elif kind == "tick":
        note = "Tick data. S1 as the base lets the engine settle intrabar fills."

    return Candidate(
        path=str(path), name=path.name, bytes=size, size=human_bytes(size),
        kind=kind, columns=columns, first_row=first[:120],
        suggested_name=suggested,
        suggested_base="S1" if kind == "tick" else "",
        suggested_timeframe="M1" if kind == "tick" else "",
        note=note,
    )


def _walk(root: Path, depth: int = 0) -> Iterator[Path]:
    if depth > MAX_DEPTH:
        return
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                # Skip the places that are always big and never data.
                if entry.name.startswith(".") or entry.name in (
                        "node_modules", "__pycache__", "AppData", "Windows",
                        "Program Files", "Program Files (x86)", "$Recycle.Bin",
                        "venv", ".venv", "site-packages"):
                    continue
                yield from _walk(Path(entry.path), depth + 1)
            elif entry.is_file(follow_symlinks=False):
                if entry.name.lower().endswith((".csv", ".txt", ".tsv")):
                    yield Path(entry.path)
        except OSError:
            continue


def discover(extra: list[str] | None = None) -> list[dict[str, Any]]:
    """Candidate data files on this machine, biggest first.

    Biggest first because the archive someone actually wants to load is almost
    always the largest CSV they own, and a list that puts a 2 MB sample above an
    11 GB tick history has buried the answer.
    """
    home = Path.home()
    roots: list[Path] = []
    for rel in SEARCH_DIRS:
        p = home / rel
        if p.is_dir():
            roots.append(p)
    for e in (extra or []):
        p = Path(e).expanduser()
        if p.is_dir():
            roots.append(p)
        elif p.is_file():
            roots.append(p.parent)

    # The repo's own data directory, so a generated sample is offered too.
    local = Path(__file__).resolve().parents[2] / "data"
    if local.is_dir():
        roots.append(local)

    seen: set[str] = set()
    found: list[Candidate] = []
    for root in roots:
        for path in _walk(root):
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            candidate = sniff(path)
            if candidate:
                found.append(candidate)
            if len(found) >= MAX_RESULTS:
                break
        if len(found) >= MAX_RESULTS:
            break

    found.sort(key=lambda c: c.bytes, reverse=True)
    return [c.to_dict() for c in found]
