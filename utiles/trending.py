# utiles/trending.py  (only the relevant lines)
from __future__ import annotations
import time
from typing import List, Dict, Any
from .bitget import make_exchange, fetch_ohlcv_safe

class TrendScanParams:
    def __init__(self, exchange_name: str, top_n: int, min_vol_z: float, timeframe: str, include_sentiment: bool):
        self.exchange_name = exchange_name
        self.top_n = top_n
        self.min_vol_z = min_vol_z
        self.timeframe = timeframe
        self.include_sentiment = include_sentiment

def _volume_zscore(ohlcv: List[List[float]], lookback: int = 50) -> float:
    import statistics
    vols = [x[5] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if len(vols) < 10:
        return float("nan")
    mean = statistics.mean(vols[:-1])
    stdev = statistics.pstdev(vols[:-1]) or 1e-12
    return (vols[-1] - mean) / stdev

def scan_trending(params: TrendScanParams) -> List[Dict[str, Any]]:
    ex = make_exchange(params.exchange_name)
    # share one clock for all symbols → same last closed boundary
    now_ms = int(time.time() * 1000)

    out: List[Dict[str, Any]] = []
    # however you enumerate symbols today
    symbols: List[str] = [s for s in getattr(ex, "symbols", []) if s.endswith("/USDT")]

    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv_safe(
                ex, symbol, params.timeframe, limit=240,
                since=None,
                closed_only=True,   # <— IMPORTANT
                now_ts=now_ms       # <— IMPORTANT
            )
            if not ohlcv:
                continue
            vz = _volume_zscore(ohlcv, lookback=50)
            if vz is not None and vz >= params.min_vol_z:
                out.append({"symbol": symbol, "vol_z": vz, "timeframe": params.timeframe})
        except Exception:
            continue

    # ... your existing sorting / top-N logic unchanged ...
    out.sort(key=lambda r: r["vol_z"], reverse=True)
    return out[: params.top_n]
