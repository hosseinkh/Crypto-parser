# utiles/trending.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics

from .bitget import make_exchange, fetch_ohlcv_safe, list_spot_usdt_symbols
try:
    from .sentiment import get_sentiment_for_symbol
    _HAS_SENT = True
except Exception:
    _HAS_SENT = False

# ---------- Params -------------------------------------------------------------

@dataclass
class TrendScanParams:
    exchange_name: str = "bitget"
    top_n: int = 10
    min_vol_z: float = 0.8
    timeframe: str = "15m"        # "15m" | "1h" | "4h"
    include_sentiment: bool = True

# Backward-compatible alias (in case some code imports without the 's')
TrendScanParam = TrendScanParams

# ---------- Small helpers ------------------------------------------------------

def _volume_zscore(ohlcv: List[List[float]], lookback: int = 50) -> float:
    vols = [x[5] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if len(vols) < 12:
        return float("nan")
    mean = statistics.mean(vols[:-1])
    stdev = statistics.pstdev(vols[:-1]) or 1e-12
    return (vols[-1] - mean) / stdev

def _safe_sent(symbol: str) -> Optional[Dict[str, Any]]:
    if not (_HAS_SENT):
        return None
    try:
        score, details = get_sentiment_for_symbol(symbol)
        return {"score": score, "details": details}
    except Exception:
        return None

# ---------- Public API ---------------------------------------------------------

def scan_trending(params: TrendScanParams) -> List[Dict[str, Any]]:
    """
    Return top-N symbols by current volume z-score on the chosen timeframe.
    A row looks like:
      {
        "symbol": "INJ/USDT",
        "timeframe": "15m",
        "vol_z": 1.73,
        "last": 9.12,
        "ohlcv_len": 240,
        "sentiment": {"score": 0.12, "details": {...}} or None,
        "reason": "vol_z=1.73 ≥ 0.80 on 15m"
      }
    """
    ex = make_exchange(params.exchange_name)
    symbols = list_spot_usdt_symbols(ex)
    out: List[Dict[str, Any]] = []

    for s in symbols:
        try:
            ohlcv = fetch_ohlcv_safe(ex, s, params.timeframe, limit=120)
            if not ohlcv:
                continue
            vz = _volume_zscore(ohlcv, lookback=50)
            if (vz != vz) or vz < params.min_vol_z:  # NaN or below threshold
                continue
            last = ohlcv[-1][4]
            row = {
                "symbol": s,
                "timeframe": params.timeframe,
                "vol_z": round(vz, 3),
                "last": last,
                "ohlcv_len": len(ohlcv),
                "sentiment": _safe_sent(s) if params.include_sentiment else None,
                "reason": f"vol_z={round(vz,3)} ≥ {params.min_vol_z} on {params.timeframe}",
            }
            out.append(row)
        except Exception:
            continue

    # Sort by vol_z desc and clip to top_n
    out.sort(key=lambda r: (r.get("vol_z") if r.get("vol_z") == r.get("vol_z") else -1e9), reverse=True)
    return out[: max(1, int(params.top_n))]

def explain_trending_row(row: Dict[str, Any]) -> str:
    s = row.get("symbol", "?")
    tf = row.get("timeframe", "?")
    vz = row.get("vol_z", "?")
    last = row.get("last", "?")
    base = f"{s} on {tf} — vol_z {vz}, last {last}"
    sent = row.get("sentiment")
    if isinstance(sent, dict) and "score" in sent and sent["score"] is not None:
        base += f", sentiment {round(sent['score'], 2)}"
    return base
