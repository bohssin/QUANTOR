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

#: Parquet's magic bytes, and gzip's. Stored blobs are sniffed rather than
#: labelled, so an artifact written by an older build still reads.
_PARQUET_MAGIC = b"PAR1"
_GZIP_MAGIC = b"\x1f\x8b"

#: zstd where available, snappy otherwise. NOT gzip: gzip compresses these
#: arrays at roughly 25 MB/s, which on the 6.7M S1 bars of a 300 MB tick file
#: is ~45 seconds of silent wait after a 15-second read `[measured]`, and on an
#: 11 GB archive would be minutes. zstd does the same job an order of magnitude
#: faster and smaller on columnar floats.
def _codec() -> str:
    import pyarrow as pa
    for name in ("zstd", "lz4", "snappy"):
        try:
            pa.Codec(name)
            return name
        except Exception:                                   # noqa: BLE001
            continue
    return "none"


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

        Two shapes, two formats. Arrays that are all the same length are a table
        (bar series, overwhelmingly the large case) and go to **Parquet**, which
        is columnar, typed, and compresses numeric data an order of magnitude
        faster than gzip. Ragged sets — an equity curve beside a trade list —
        are not a table, are small, and stay in a compressed npz.

        Either way the bytes are a pure function of the content, so the same
        bars always land on the same digest instead of a new one each save.
        """
        clean = {k: np.ascontiguousarray(v) for k, v in sorted(arrays.items())}
        lengths = {a.shape[0] for a in clean.values() if a.ndim == 1}
        rectangular = (clean and len(lengths) == 1
                       and all(a.ndim == 1 for a in clean.values()))

        if rectangular:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({k: pa.array(v) for k, v in clean.items()})
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf, compression=_codec(),
                           # Deterministic bytes: no creation timestamp, no
                           # per-write statistics that vary with buffering.
                           write_statistics=False, store_schema=True)
            return self.put_bytes(buf.getvalue().to_pybytes())

        buf = io.BytesIO()
        np.savez(buf, **clean)
        # Level 1: these are already small, and the default level 9 costs ~6x
        # the time for a few percent of size.
        return self.put_bytes(gzip.compress(buf.getvalue(), compresslevel=1, mtime=0))

    # --- reading ---------------------------------------------------------

    def get_bytes(self, digest: str) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f"artifact {digest[:12]}... is not in {self.root}")
        return path.read_bytes()

    def get_json(self, digest: str) -> Any:
        return json.loads(gzip.decompress(self.get_bytes(digest)))

    def get_arrays(self, digest: str) -> dict[str, np.ndarray]:
        raw = self.get_bytes(digest)
        if raw[:4] == _PARQUET_MAGIC:
            import pyarrow.parquet as pq

            table = pq.read_table(io.BytesIO(raw))
            return {name: table.column(name).to_numpy(zero_copy_only=False)
                    for name in table.column_names}
        if raw[:2] == _GZIP_MAGIC:
            with np.load(io.BytesIO(gzip.decompress(raw))) as npz:
                return {k: npz[k] for k in npz.files}
        raise ValueError(
            f"artifact {digest[:12]}... is neither Parquet nor a gzipped npz")

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
