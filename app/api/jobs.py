"""Long loads, run in the background with progress. Plan §15.

A tick archive takes minutes to fold into bars. A synchronous HTTP POST that
takes minutes is not slow — it is **broken**: the browser gives up, the proxy
gives up, the user gives up first of all, and the work carries on invisibly to
no purpose. There is also nothing to look at, which for a fifteen-minute
operation is its own failure.

So a load starts a job and returns immediately. The job reports rows read and
bars built as it goes, the UI polls it, and the result is fetched when it is
done. One worker thread at a time on purpose: two concurrent multi-gigabyte
reads on one disk are slower than doing them in order, and the memory is the
constraint that actually bites.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Job", "JobRunner"]


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | ok | error | cancelled
    rows: int = 0
    bars: int = 0
    total_bytes: int = 0
    message: str = ""
    error: str = ""
    result: dict[str, Any] | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        rate = self.rows / elapsed if elapsed > 0 and self.rows else 0.0
        return {
            "job_id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "rows": self.rows, "bars": self.bars,
            "total_bytes": self.total_bytes,
            "seconds": round(elapsed, 1),
            "rows_per_second": int(rate),
            "message": self.message, "error": self.error, "result": self.result,
            "done": self.status in ("ok", "error", "cancelled"),
        }


class JobRunner:
    """One background job at a time, with progress anyone can poll."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._current: str | None = None

    def start(self, kind: str, label: str,
              work: Callable[[Job], dict[str, Any]]) -> Job:
        with self._lock:
            if self._current and self._jobs[self._current].status == "running":
                running = self._jobs[self._current]
                raise RuntimeError(
                    f"a {running.kind} job is already running ({running.label}); "
                    "wait for it or cancel it. Two multi-gigabyte reads at once "
                    "are slower than one after the other."
                )
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
            self._jobs[job.id] = job
            self._current = job.id

        def run() -> None:
            job.status = "running"
            try:
                job.result = work(job)
                job.status = "ok"
            except Exception as exc:                        # noqa: BLE001
                job.status = "error"
                # The type name and message, not a bare 500: the caller can only
                # fix what it can read (same reasoning as the MCP error surfacing).
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3).splitlines()[-1][:200]
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, name=f"quantor-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def latest(self) -> Job | None:
        return self._jobs.get(self._current) if self._current else None

    def all(self, limit: int = 20) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at,
                      reverse=True)[:limit]
