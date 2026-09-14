"""Content-addressed blob store. Plan §16.2.

Equity curves, trade lists, optimization tables and cached bar arrays are all
"large, immutable, and referenced from several rows". Addressing them by the
sha256 of their own bytes gets three properties for free:

- **Nothing is ever overwritten.** A digest names one sequence of bytes forever,
  so a row pointing at an artifact points at the artifact it was written with.
  Re-running a backtest cannot retroactively change what an old run reported.
- **Identical results deduplicate.** Two runs that produced the same equity
  curve store one file, which matters when a sweep writes 500 of them.
- **Tampering is detectable**, since the name is a checksum of the content.

Layout is `artifacts/<first two hex>/<full digest>`, the usual fan-out so a
directory never holds a hundred thousand entries.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Artifacts"]


class Artifacts:
    """Immutable blobs on disk, named by their own sha256."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- writing ---------------------------------------------------------

    def put_bytes(self, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        path = self.path_for(digest)
        if path.exists():
            return digest                      # already stored, byte for byte
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a crash mid-write cannot leave a short file
        # sitting under a digest that promises different content.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return digest

    def put_json(self, obj: Any) -> str:
        """Store a JSON document, gzipped. Trade lists compress ~8x."""
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True,
                         default=_jsonable).encode()
        return self.put_bytes(gzip.compress(raw, mtime=0))

    def put_arrays(self, arrays: dict[str, np.ndarray]) -> str:
        """Store named numpy arrays — cached bars, equity curves, trade columns.

        `mtime=0` on the gzip header and sorted keys keep the bytes a pure
        function of the content: the same bars always land on the same digest,
        rather than on a new one every time they are saved.
        """
        buf = io.BytesIO()
        np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in sorted(arrays.items())})
        return self.put_bytes(gzip.compress(buf.getvalue(), mtime=0))

    # --- reading ---------------------------------------------------------

    def get_bytes(self, digest: str) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f"artifact {digest[:12]}... is not in {self.root}")
        return path.read_bytes()

    def get_json(self, digest: str) -> Any:
        return json.loads(gzip.decompress(self.get_bytes(digest)))

    def get_arrays(self, digest: str) -> dict[str, np.ndarray]:
        raw = gzip.decompress(self.get_bytes(digest))
        with np.load(io.BytesIO(raw)) as npz:
            return {k: npz[k] for k in npz.files}

    # --- housekeeping ----------------------------------------------------

    def path_for(self, digest: str) -> Path:
        clean = digest.strip()
        if len(clean) < 4 or not all(c in "0123456789abcdef" for c in clean):
            raise ValueError(f"not a digest: {digest!r}")
        return self.root / clean[:2] / clean

    def exists(self, digest: str) -> bool:
        try:
            return self.path_for(digest).exists()
        except ValueError:
            return False

    def size(self) -> tuple[int, int]:
        """(count, bytes) currently stored."""
        count = total = 0
        for p in self.root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                count += 1
                total += p.stat().st_size
        return count, total


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")
