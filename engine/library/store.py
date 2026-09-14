"""The library: strategies, runs and data that survive a restart. Plan §16.

Before this, `quantor_mcp/server.py` held everything in two module-level dicts.
Close the session and every version, every run, every rejection and every reason
was gone — which also meant §16.4's failure taxonomy and §12's cumulative
comparison count could not exist, because both are claims about history.

One SQLite file plus a content-addressed artifact directory. No ORM: the schema
is twelve tables of plain columns, and a layer that hides SQL here would add
indirection without removing a single decision.

**Two processes write this.** Claude Code spawns the MCP server; the API server
runs separately; both save runs. So the connection opens in WAL mode with a busy
timeout — without those, the second writer gets `database is locked` and the
owner sees a backtest vanish. Connections are thread-local because SQLite
objects are not safe to share across threads and FastAPI answers on a pool.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import Artifacts

__all__ = ["Library", "DataSource", "Strategy", "Version", "Run", "default_root"]

SCHEMA = Path(__file__).with_name("schema.sql")

#: Bumped when the schema changes in a way old rows cannot satisfy.
SCHEMA_VERSION = "2"


def default_root() -> Path:
    """Where the library lives unless told otherwise.

    Environment first so the MCP server, the API server and the tests can be
    pointed at the same place — or deliberately different ones.
    """
    env = os.environ.get("QUANTOR_LIBRARY")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "library"


# --- row types ---------------------------------------------------------------

@dataclass
class DataSource:
    id: int
    name: str
    path: str
    digest: str
    kind: str
    base_timeframe: str
    default_timeframe: str
    default_bars: int
    utc_offset_hours: float
    rows: int
    bars: int
    first_ms: int
    last_ms: int
    tick_size: float
    instrument: dict[str, Any]
    quality: dict[str, Any]
    bars_sha: str | None
    ingest: str
    created_at: float

    def span_days(self) -> float:
        return (self.last_ms - self.first_ms) / 86_400_000


@dataclass
class Strategy:
    id: int
    strategy_id: str
    name: str
    family: str
    tags: str
    archived: bool
    archived_reason: str
    created_at: float
    versions: int = 0
    runs: int = 0
    last_run_at: float | None = None


@dataclass
class Version:
    id: int
    strategy_id: str
    strategy_pk: int
    version: int
    parent_version: int | None
    signal_block: str
    signal_sha: str
    params: dict[str, Any]
    description: str
    source_url: str
    origin: str
    warnings: list[str]
    created_at: float


@dataclass
class Run:
    id: int
    version_pk: int
    data_pk: int
    kind: str
    params: dict[str, Any]
    timeframe: str
    modeling_mode: str
    seed: int | None
    status: str
    metrics: dict[str, Any]
    artifact_sha: str | None
    error: str
    started_at: float
    finished_at: float | None
    seconds: float | None
    # joined, for display
    strategy_id: str = ""
    version: int = 0
    data_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --- the library -------------------------------------------------------------

class Library:
    """Everything that has to outlive the process."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "library.db"
        self.artifacts = Artifacts(self.root / "artifacts")
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._ensure_schema()

    # --- connection ------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0,
                                   isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL lets the API server read while the MCP server writes. Without
            # it the two processes serialize on the whole file and the loser
            # raises `database is locked` mid-backtest.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    #: Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS`
    #: silently leaves an existing table alone, so a new column has to be added
    #: explicitly — otherwise everyone who already has a library gets a
    #: "no such column" error and is told to delete their run history.
    _ADDED_COLUMNS = {
        "data_source": [
            ("default_timeframe", "TEXT NOT NULL DEFAULT ''"),
            ("default_bars", "INTEGER NOT NULL DEFAULT 0"),
        ],
    }

    def _migrate(self) -> None:
        for table, columns in self._ADDED_COLUMNS.items():
            existing = {r["name"] for r in self.conn.execute(
                f"PRAGMA table_info({table})")}
            if not existing:
                continue                      # the table is about to be created
            for name, decl in columns:
                if name not in existing:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _ensure_schema(self) -> None:
        with self._init_lock:
            self._migrate()
            self.conn.executescript(SCHEMA.read_text())
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            got = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            if got != SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.db_path} was written by schema version {got}, this build "
                    f"expects {SCHEMA_VERSION}. Move it aside rather than migrating "
                    "blind — the runs in it are evidence."
                )

    # --- data sources ----------------------------------------------------

    def put_data_source(
        self, *, name: str, path: str, digest: str, kind: str,
        base_timeframe: str, default_timeframe: str, default_bars: int,
        utc_offset_hours: float, rows: int,
        first_ms: int, last_ms: int, tick_size: float,
        instrument: dict[str, Any], quality: dict[str, Any],
        bars: dict[str, np.ndarray], ingest: str = "whole",
    ) -> DataSource:
        """Register a source and cache its bars as an artifact.

        The bars are stored, not just described. That is what makes the restart
        useful rather than merely tidy: re-opening an 11 GB tick archive would
        otherwise mean re-reading 11 GB, and nobody does that twice a day.
        """
        bars_sha = self.artifacts.put_arrays(bars)
        now = time.time()
        self.conn.execute(
            """INSERT INTO data_source
               (name, path, digest, kind, base_timeframe, default_timeframe,
                default_bars, utc_offset_hours, rows, bars, first_ms, last_ms,
                tick_size, instrument_json, quality_json, bars_sha, ingest,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (name) DO UPDATE SET
                 path=excluded.path, digest=excluded.digest, kind=excluded.kind,
                 base_timeframe=excluded.base_timeframe,
                 default_timeframe=excluded.default_timeframe,
                 default_bars=excluded.default_bars,
                 utc_offset_hours=excluded.utc_offset_hours, rows=excluded.rows,
                 bars=excluded.bars, first_ms=excluded.first_ms,
                 last_ms=excluded.last_ms, tick_size=excluded.tick_size,
                 instrument_json=excluded.instrument_json,
                 quality_json=excluded.quality_json, bars_sha=excluded.bars_sha,
                 ingest=excluded.ingest""",
            (name, str(path), digest, kind, base_timeframe.upper(),
             (default_timeframe or base_timeframe).upper(), int(default_bars),
             float(utc_offset_hours), int(rows), int(bars["ms"].shape[0]),
             int(first_ms), int(last_ms), float(tick_size),
             json.dumps(instrument), json.dumps(quality, default=_jsonable),
             bars_sha, ingest, now),
        )
        got = self.get_data_source(name)
        assert got is not None
        return got

    def get_data_source(self, name: str) -> DataSource | None:
        row = self.conn.execute(
            "SELECT * FROM data_source WHERE name = ?", (name,)
        ).fetchone()
        return _data_source(row) if row else None

    def list_data_sources(self) -> list[DataSource]:
        rows = self.conn.execute(
            "SELECT * FROM data_source ORDER BY created_at DESC"
        ).fetchall()
        return [_data_source(r) for r in rows]

    def load_bars(self, name: str) -> dict[str, np.ndarray]:
        """The cached bar arrays at the source's base timeframe."""
        src = self.get_data_source(name)
        if src is None:
            known = [d.name for d in self.list_data_sources()]
            raise ValueError(f"unknown data source {name!r}; loaded: {known or 'none'}")
        if not src.bars_sha:
            raise ValueError(f"{name!r} has no cached bars — reload it")
        return self.artifacts.get_arrays(src.bars_sha)

    def delete_data_source(self, name: str) -> bool:
        """Forget a source. The artifact stays — other rows may reference it."""
        cur = self.conn.execute("DELETE FROM data_source WHERE name = ?", (name,))
        return cur.rowcount > 0

    # --- strategies ------------------------------------------------------

    def save_version(
        self, *, strategy_id: str, description: str, signal_block: str,
        params: dict[str, Any] | None = None, parent_version: int | None = None,
        source_url: str = "", origin: str = "agent", family: str = "",
        name: str = "", tags: str = "", warnings: Iterable[str] = (),
    ) -> Version:
        """Append a version. Never overwrites; `parent_version` makes it a tree.

        Default parent is the latest version, which is the ordinary "I improved
        it" case. Passing an older number explicitly is how you branch from
        Tuesday's, which is exactly what a list of versions could not express.
        """
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT * FROM strategy WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
            if row is None:
                cur = self.conn.execute(
                    """INSERT INTO strategy (strategy_id, name, family, tags, created_at)
                       VALUES (?,?,?,?,?)""",
                    (strategy_id, name or strategy_id, family or strategy_id,
                     tags, now),
                )
                strategy_pk = int(cur.lastrowid)
            else:
                strategy_pk = int(row["id"])
                if family and row["family"] != family:
                    self.conn.execute(
                        "UPDATE strategy SET family = ? WHERE id = ?",
                        (family, strategy_pk),
                    )

            last = self.conn.execute(
                "SELECT MAX(version) AS v FROM strategy_version WHERE strategy_pk = ?",
                (strategy_pk,),
            ).fetchone()["v"]
            version = int(last or 0) + 1
            parent = parent_version if parent_version is not None else (last or None)
            if parent is not None and not 1 <= int(parent) <= int(last or 0):
                raise ValueError(
                    f"{strategy_id} has versions 1..{last or 0}; cannot branch "
                    f"from v{parent}"
                )

            cur = self.conn.execute(
                """INSERT INTO strategy_version
                   (strategy_pk, version, parent_version, signal_block, signal_sha,
                    params_json, description, source_url, origin, warnings_json,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (strategy_pk, version, parent, signal_block,
                 hashlib.sha256(signal_block.encode()).hexdigest(),
                 json.dumps(params or {}), description, source_url, origin,
                 json.dumps(list(warnings)), now),
            )
            version_pk = int(cur.lastrowid)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        got = self.get_version_by_pk(version_pk)
        assert got is not None
        return got

    def get_version(self, strategy_id: str, version: int = 0) -> Version:
        """Version `n`, or the latest when `version` is 0 or -1."""
        row = self.conn.execute(
            "SELECT * FROM strategy WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is None:
            known = [s.strategy_id for s in self.list_strategies(include_archived=True)]
            raise ValueError(
                f"unknown strategy {strategy_id!r}; saved: {known or 'none'}"
            )
        if version in (0, -1):
            got = self.conn.execute(
                """SELECT * FROM strategy_version WHERE strategy_pk = ?
                   ORDER BY version DESC LIMIT 1""", (row["id"],),
            ).fetchone()
        else:
            got = self.conn.execute(
                "SELECT * FROM strategy_version WHERE strategy_pk = ? AND version = ?",
                (row["id"], int(version)),
            ).fetchone()
        if got is None:
            have = self.conn.execute(
                "SELECT MAX(version) AS v FROM strategy_version WHERE strategy_pk = ?",
                (row["id"],),
            ).fetchone()["v"]
            raise ValueError(
                f"{strategy_id} has versions 1..{have or 0}, asked for {version}"
            )
        return _version(got, strategy_id)

    def get_version_by_pk(self, version_pk: int) -> Version | None:
        row = self.conn.execute(
            """SELECT v.*, s.strategy_id FROM strategy_version v
               JOIN strategy s ON s.id = v.strategy_pk WHERE v.id = ?""",
            (version_pk,),
        ).fetchone()
        return _version(row, row["strategy_id"]) if row else None

    def list_versions(self, strategy_id: str) -> list[Version]:
        rows = self.conn.execute(
            """SELECT v.* FROM strategy_version v
               JOIN strategy s ON s.id = v.strategy_pk
               WHERE s.strategy_id = ? ORDER BY v.version""",
            (strategy_id,),
        ).fetchall()
        return [_version(r, strategy_id) for r in rows]

    def list_strategies(self, *, include_archived: bool = False) -> list[Strategy]:
        rows = self.conn.execute(
            f"""SELECT s.*,
                       (SELECT COUNT(*) FROM strategy_version v
                         WHERE v.strategy_pk = s.id) AS versions,
                       (SELECT COUNT(*) FROM run r
                          JOIN strategy_version v2 ON v2.id = r.version_pk
                         WHERE v2.strategy_pk = s.id) AS runs,
                       (SELECT MAX(r.started_at) FROM run r
                          JOIN strategy_version v3 ON v3.id = r.version_pk
                         WHERE v3.strategy_pk = s.id) AS last_run_at
                FROM strategy s
                {"" if include_archived else "WHERE s.archived = 0"}
                ORDER BY COALESCE(last_run_at, s.created_at) DESC"""
        ).fetchall()
        return [
            Strategy(
                id=int(r["id"]), strategy_id=r["strategy_id"], name=r["name"],
                family=r["family"], tags=r["tags"], archived=bool(r["archived"]),
                archived_reason=r["archived_reason"], created_at=r["created_at"],
                versions=int(r["versions"]), runs=int(r["runs"]),
                last_run_at=r["last_run_at"],
            )
            for r in rows
        ]

    def version_tree(self, strategy_id: str) -> list[dict[str, Any]]:
        """Versions as a parent/child forest, for the UI's tree view."""
        versions = self.list_versions(strategy_id)
        nodes = {
            v.version: {
                "version": v.version, "parent": v.parent_version,
                "description": v.description, "params": v.params,
                "origin": v.origin, "source_url": v.source_url,
                "created_at": v.created_at, "signal_sha": v.signal_sha[:12],
                "children": [],
            }
            for v in versions
        }
        roots = []
        for v in versions:
            node = nodes[v.version]
            parent = nodes.get(v.parent_version) if v.parent_version else None
            (parent["children"] if parent else roots).append(node)
        return roots

    def archive(self, strategy_id: str, reason: str = "", archived: bool = True) -> None:
        """Hide a strategy without deleting anything. Plan §16.4.

        Rejections are the most informative rows in the library — "we tried this
        and it did not survive walk-forward" is a result. Deleting them means
        re-running the same idea in three months having forgotten why it failed.
        """
        self.conn.execute(
            "UPDATE strategy SET archived = ?, archived_reason = ? WHERE strategy_id = ?",
            (1 if archived else 0, reason, strategy_id),
        )

    # --- runs ------------------------------------------------------------

    def start_run(self, *, version_pk: int, data_pk: int, kind: str,
                  params: dict[str, Any], timeframe: str,
                  modeling_mode: str = "bar_close", seed: int | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO run (version_pk, data_pk, kind, params_json, timeframe,
                                modeling_mode, seed, status, started_at)
               VALUES (?,?,?,?,?,?,?,'running',?)""",
            (int(version_pk), int(data_pk), kind, json.dumps(params, default=_jsonable),
             timeframe.upper(), modeling_mode, seed, time.time()),
        )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, status: str = "ok",
                   metrics: dict[str, Any] | None = None,
                   artifact_sha: str | None = None, error: str = "") -> None:
        started = self.conn.execute(
            "SELECT started_at FROM run WHERE id = ?", (run_id,)
        ).fetchone()["started_at"]
        now = time.time()
        self.conn.execute(
            """UPDATE run SET status=?, metrics_json=?, artifact_sha=?, error=?,
                              finished_at=?, seconds=? WHERE id=?""",
            (status, json.dumps(metrics or {}, default=_jsonable), artifact_sha,
             error, now, now - started, run_id),
        )

    def record_optimization(self, run_id: int, *, mode: str, objective: str,
                            budget: dict[str, Any], comparisons: int,
                            results_sha: str | None = None,
                            robustness: float | None = None,
                            verdict: str = "") -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO optimization
               (run_pk, mode, objective, budget_json, comparisons, results_sha,
                robustness, verdict) VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, mode, objective, json.dumps(budget, default=_jsonable),
             int(comparisons), results_sha, robustness, verdict),
        )

    def record_validation(self, run_id: int, *, mode: str, folds: int,
                          efficiency: float | None, verdict: str = "",
                          comparisons: int = 0,
                          per_fold: list[dict[str, Any]] | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO validation
               (run_pk, mode, folds, efficiency, comparisons, verdict, folds_json)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, mode, int(folds),
             None if efficiency is None or not np.isfinite(efficiency) else float(efficiency),
             int(comparisons), verdict,
             json.dumps(per_fold or [], default=_jsonable)),
        )

    def get_run(self, run_id: int) -> Run | None:
        row = self.conn.execute(
            """SELECT r.*, s.strategy_id, v.version, d.name AS data_name
               FROM run r
               JOIN strategy_version v ON v.id = r.version_pk
               JOIN strategy s ON s.id = v.strategy_pk
               JOIN data_source d ON d.id = r.data_pk
               WHERE r.id = ?""", (run_id,),
        ).fetchone()
        if row is None:
            return None
        run = _run(row)
        for table, key in (("optimization", "optimization"), ("validation", "validation")):
            extra = self.conn.execute(
                f"SELECT * FROM {table} WHERE run_pk = ?", (run_id,)
            ).fetchone()
            if extra:
                run.extra[key] = {k: extra[k] for k in extra.keys() if k != "run_pk"}
        return run

    def list_runs(self, *, strategy_id: str | None = None, kind: str | None = None,
                  limit: int = 50) -> list[Run]:
        where, args = [], []
        if strategy_id:
            where.append("s.strategy_id = ?")
            args.append(strategy_id)
        if kind:
            where.append("r.kind = ?")
            args.append(kind)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(
            f"""SELECT r.*, s.strategy_id, v.version, d.name AS data_name
                FROM run r
                JOIN strategy_version v ON v.id = r.version_pk
                JOIN strategy s ON s.id = v.strategy_pk
                JOIN data_source d ON d.id = r.data_pk
                {clause} ORDER BY r.id DESC LIMIT ?""",
            (*args, int(limit)),
        ).fetchall()
        return [_run(r) for r in rows]

    # --- research integrity ----------------------------------------------

    def family_comparisons(self, family: str) -> int:
        """Every parameter combination ever scored against this family. §12.

        The deflated Sharpe ratio divides by how many things were tried. Counting
        one sweep is the standard way to make an overfit result look significant,
        so the number that comes out of here is cumulative and only ever grows.
        """
        row = self.conn.execute(
            "SELECT comparisons FROM family_comparisons WHERE family = ?", (family,)
        ).fetchone()
        return int(row["comparisons"]) if row else 0

    def family_of(self, strategy_id: str) -> str:
        row = self.conn.execute(
            "SELECT family FROM strategy WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        return row["family"] if row else strategy_id

    def stats(self) -> dict[str, Any]:
        def count(table: str, where: str = "") -> int:
            return int(self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} {where}"
            ).fetchone()["n"])

        files, size = self.artifacts.size()
        return {
            "root": str(self.root),
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "data_sources": count("data_source"),
            "strategies": count("strategy", "WHERE archived = 0"),
            "archived": count("strategy", "WHERE archived = 1"),
            "versions": count("strategy_version"),
            "runs": count("run"),
            "backtests": count("run", "WHERE kind = 'backtest'"),
            "optimizations": count("run", "WHERE kind = 'optimize'"),
            "validations": count("run", "WHERE kind = 'validate'"),
            "artifacts": files,
            "artifact_bytes": size,
        }

    # --- sessions --------------------------------------------------------

    def record_session(self, *, session_key: str, surface: str = "", model: str = "",
                       runtime: str = "", mcp_servers: list[str] | None = None,
                       transcript_path: str = "") -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO agent_session
               (session_key, surface, model, runtime, mcp_servers_json,
                transcript_path, started_at) VALUES (?,?,?,?,?,?,?)""",
            (session_key, surface, model, runtime,
             json.dumps(mcp_servers or []), transcript_path, time.time()),
        )


# --- row mapping -------------------------------------------------------------

def _data_source(row: sqlite3.Row) -> DataSource:
    return DataSource(
        id=int(row["id"]), name=row["name"], path=row["path"], digest=row["digest"],
        kind=row["kind"], base_timeframe=row["base_timeframe"],
        default_timeframe=row["default_timeframe"] or row["base_timeframe"],
        default_bars=int(row["default_bars"] or 0),
        utc_offset_hours=float(row["utc_offset_hours"]), rows=int(row["rows"]),
        bars=int(row["bars"]), first_ms=int(row["first_ms"]),
        last_ms=int(row["last_ms"]), tick_size=float(row["tick_size"]),
        instrument=json.loads(row["instrument_json"]),
        quality=json.loads(row["quality_json"]), bars_sha=row["bars_sha"],
        ingest=row["ingest"], created_at=float(row["created_at"]),
    )


def _version(row: sqlite3.Row, strategy_id: str) -> Version:
    return Version(
        id=int(row["id"]), strategy_id=strategy_id,
        strategy_pk=int(row["strategy_pk"]), version=int(row["version"]),
        parent_version=row["parent_version"], signal_block=row["signal_block"],
        signal_sha=row["signal_sha"], params=json.loads(row["params_json"]),
        description=row["description"], source_url=row["source_url"],
        origin=row["origin"], warnings=json.loads(row["warnings_json"]),
        created_at=float(row["created_at"]),
    )


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=int(row["id"]), version_pk=int(row["version_pk"]),
        data_pk=int(row["data_pk"]), kind=row["kind"],
        params=json.loads(row["params_json"]), timeframe=row["timeframe"],
        modeling_mode=row["modeling_mode"], seed=row["seed"], status=row["status"],
        metrics=json.loads(row["metrics_json"]), artifact_sha=row["artifact_sha"],
        error=row["error"], started_at=float(row["started_at"]),
        finished_at=row["finished_at"], seconds=row["seconds"],
        strategy_id=row["strategy_id"] if "strategy_id" in row.keys() else "",
        version=int(row["version"]) if "version" in row.keys() else 0,
        data_name=row["data_name"] if "data_name" in row.keys() else "",
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")
