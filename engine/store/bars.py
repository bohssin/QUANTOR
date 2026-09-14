"""Bar construction from ticks. Plan §4.4.

MT5 semantics, and each one is a decision that changes results:

- **OHLC from bid.** Ask is bid + spread. This is MT5's default and what the
  owner's existing results were produced with.
- **Volume is tick count**, which is what MT5 reports for tick data.
- **Spread is carried per bar**, mean and max, because the fill engine applies it
  directionally and a bar-mode run has nothing else to go on.
- **A bar with no ticks does not exist.** Never forward-filled. MT5 skips empty
  bars, and inventing one manufactures a price that could not be traded.

Bar boundaries are computed on the timestamps as given. Those are already in the
clock the source declared (`SourceSpec.utc_offset_hours`, §4.3), so shifting the
offset at ingest is what moves the daily boundary — not anything here.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .ingest import MarketData, timeframe_ms

__all__ = ["build_bars", "bars_from_ticks", "aggregate_bars"]


@njit(cache=True, fastmath=False)
def _build(t, bid, ask, tf_ms):
    n = t.shape[0]
    bar_t = np.empty(n, np.int64)
    o = np.empty(n, np.float64)
    h = np.empty(n, np.float64)
    low = np.empty(n, np.float64)
    c = np.empty(n, np.float64)
    vol = np.empty(n, np.int64)
    sp_mean = np.empty(n, np.float64)
    sp_max = np.empty(n, np.float64)

    k = -1
    current = np.int64(-1)
    sp_sum = 0.0

    for i in range(n):
        boundary = (t[i] // tf_ms) * tf_ms
        spread = ask[i] - bid[i]
        if boundary != current:
            if k >= 0:
                sp_mean[k] = sp_sum / vol[k]
            k += 1
            current = boundary
            bar_t[k] = boundary
            o[k] = bid[i]
            h[k] = bid[i]
            low[k] = bid[i]
            c[k] = bid[i]
            vol[k] = 1
            sp_sum = spread
            sp_max[k] = spread
        else:
            if bid[i] > h[k]:
                h[k] = bid[i]
            if bid[i] < low[k]:
                low[k] = bid[i]
            c[k] = bid[i]
            vol[k] += 1
            sp_sum += spread
            if spread > sp_max[k]:
                sp_max[k] = spread
    if k >= 0:
        sp_mean[k] = sp_sum / vol[k]

    m = k + 1
    return bar_t[:m], o[:m], h[:m], low[:m], c[:m], vol[:m], sp_mean[:m], sp_max[:m]


def bars_from_ticks(ms: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                    timeframe: str) -> dict[str, np.ndarray]:
    """Aggregate ticks into OHLC bars at `timeframe`. Prices in, prices out."""
    tf = timeframe_ms(timeframe)
    if tf <= 0:
        raise ValueError(f"{timeframe!r} is not a bar timeframe")
    if ms.shape[0] == 0:
        empty = np.empty(0)
        return {"ms": np.empty(0, np.int64), "open": empty, "high": empty,
                "low": empty, "close": empty, "volume": np.empty(0, np.int64),
                "spread": empty, "spread_max": empty}

    out = _build(
        np.ascontiguousarray(ms, dtype=np.int64),
        np.ascontiguousarray(bid, dtype=np.float64),
        np.ascontiguousarray(ask, dtype=np.float64),
        np.int64(tf),
    )
    return {"ms": out[0], "open": out[1], "high": out[2], "low": out[3],
            "close": out[4], "volume": out[5], "spread": out[6],
            "spread_max": out[7]}


@njit(cache=True, fastmath=False)
def _aggregate(t, o, h, low, c, vol, sp_mean, sp_max, tf_ms):
    n = t.shape[0]
    bar_t = np.empty(n, np.int64)
    out_o = np.empty(n, np.float64)
    out_h = np.empty(n, np.float64)
    out_l = np.empty(n, np.float64)
    out_c = np.empty(n, np.float64)
    out_v = np.empty(n, np.int64)
    out_spm = np.empty(n, np.float64)
    out_spx = np.empty(n, np.float64)

    k = -1
    current = np.int64(-1)
    sp_weighted = 0.0

    for i in range(n):
        boundary = (t[i] // tf_ms) * tf_ms
        if boundary != current:
            if k >= 0:
                out_spm[k] = sp_weighted / out_v[k] if out_v[k] > 0 else 0.0
            k += 1
            current = boundary
            bar_t[k] = boundary
            out_o[k] = o[i]
            out_h[k] = h[i]
            out_l[k] = low[i]
            out_c[k] = c[i]
            out_v[k] = vol[i]
            out_spx[k] = sp_max[i]
            sp_weighted = sp_mean[i] * vol[i]
        else:
            if h[i] > out_h[k]:
                out_h[k] = h[i]
            if low[i] < out_l[k]:
                out_l[k] = low[i]
            out_c[k] = c[i]
            out_v[k] += vol[i]
            if sp_max[i] > out_spx[k]:
                out_spx[k] = sp_max[i]
            sp_weighted += sp_mean[i] * vol[i]
    if k >= 0:
        out_spm[k] = sp_weighted / out_v[k] if out_v[k] > 0 else 0.0

    m = k + 1
    return (bar_t[:m], out_o[:m], out_h[:m], out_l[:m], out_c[:m], out_v[:m],
            out_spm[:m], out_spx[:m])


def aggregate_bars(bars: dict[str, np.ndarray], to_timeframe: str,
                   from_timeframe: str | None = None) -> dict[str, np.ndarray]:
    """Roll finer bars up into coarser ones. Never the other way (§4.6).

    Exists so a large tick archive is streamed **once**: build M1, cache it, and
    derive every coarser timeframe from that in milliseconds instead of re-reading
    the file. The result is identical to building the coarser bars straight from
    the ticks, which `tests/store/test_stream.py` asserts — the reason spread is
    averaged **weighted by tick count** rather than as a mean of means, since a
    bar with 900 ticks and a bar with 3 are not equal evidence about the spread.
    """
    target = timeframe_ms(to_timeframe)
    if target <= 0:
        raise ValueError(f"{to_timeframe!r} is not a bar timeframe")
    if from_timeframe is not None:
        source = timeframe_ms(from_timeframe)
        if source > target:
            raise ValueError(
                f"cannot aggregate {from_timeframe.upper()} bars down to "
                f"{to_timeframe.upper()} — bars roll up, never split (§4.6)"
            )
        if source == target:
            return dict(bars)

    if bars["ms"].shape[0] == 0:
        return dict(bars)

    out = _aggregate(
        np.ascontiguousarray(bars["ms"], dtype=np.int64),
        np.ascontiguousarray(bars["open"], dtype=np.float64),
        np.ascontiguousarray(bars["high"], dtype=np.float64),
        np.ascontiguousarray(bars["low"], dtype=np.float64),
        np.ascontiguousarray(bars["close"], dtype=np.float64),
        np.ascontiguousarray(bars["volume"], dtype=np.int64),
        np.ascontiguousarray(bars["spread"], dtype=np.float64),
        np.ascontiguousarray(bars["spread_max"], dtype=np.float64),
        np.int64(target),
    )
    return {"ms": out[0], "open": out[1], "high": out[2], "low": out[3],
            "close": out[4], "volume": out[5], "spread": out[6],
            "spread_max": out[7]}


def build_bars(data: MarketData, timeframe: str) -> dict[str, np.ndarray]:
    """Bars at `timeframe` from any source — ticks aggregated, bars passed through.

    Refuses to invent detail: a source coarser than the requested timeframe
    cannot serve it (§4.6), and aggregating bars up is not implemented yet, so
    an equal-resolution bar source is passed through and anything else raises
    rather than quietly returning the wrong thing.
    """
    if data.kind == "tick":
        return bars_from_ticks(data.ms, data.to_float("bid"), data.to_float("ask"),
                               timeframe)

    want = timeframe_ms(timeframe)
    if data.resolution_ms > want:
        raise ValueError(
            f"source resolution is coarser than {timeframe.upper()} — bars can be "
            "aggregated up, never split down (§4.6)"
        )
    out = {
        "ms": data.ms,
        "open": data.to_float("open"),
        "high": data.to_float("high"),
        "low": data.to_float("low"),
        "close": data.to_float("close"),
    }
    out["volume"] = (data.volume if data.volume is not None
                     else np.zeros(len(data), np.int64))
    out["spread"] = (data.to_float("spread") if data.spread is not None
                     else np.zeros(len(data)))
    out["spread_max"] = out["spread"]
    if data.resolution_ms < want:
        return aggregate_bars(out, timeframe)
    return out
