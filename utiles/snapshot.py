# utiles/snapshot.py
from __future__ import annotations
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .bitget import make_exchange, fetch_ohlcv_safe, fetch_ticker_safe

# ------------------------ indicators (self-contained) -------------------------

def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return float("nan")
    gains, losses = [], []
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        (gains if change > 0 else losses).append(abs(change))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr_pct(ohlcv: List[List[float]], period: int = 14) -> float:
    if len(ohlcv) < period + 1:
        return float("nan")
    trs: List[float] = []
    for i in range(1, period + 1):
        h = ohlcv[-i][2]; l = ohlcv[-i][3]; c_prev = ohlcv[-i - 1][4]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    atr = sum(trs) / period
    lc = ohlcv[-1][4]
    return (atr / lc) * 100.0 if lc else float("nan")

def body_pct(ohlcv: List[List[float]]) -> float:
    if not ohlcv:
        return float("nan")
    o, c = ohlcv[-1][1], ohlcv[-1][4]
    h, l = ohlcv[-1][2], ohlcv[-1][3]
    rng = max(h - l, 1e-12)
    return abs(c - o) / rng * 100.0

def volume_zscore(ohlcv: List[List[float]], lookback: int = 50) -> float:
    vols = [x[5] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if len(vols) < 10:
        return float("nan")
    mean = statistics.mean(vols[:-1])
    stdev = statistics.pstdev(vols[:-1]) or 1e-12
    return (vols[-1] - mean) / stdev

def last_close(ohlcv: List[List[float]]) -> float:
    return ohlcv[-1][4] if ohlcv else float("nan")

def dist_to_extremes_pct(ohlcv: List[List[float]], lookback: int = 120) -> Dict[str, float]:
    closes = [x[4] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if not closes:
        return {"to_high_pct": float("nan"), "to_low_pct": float("nan")}
    high = max(closes); low = min(closes); lc = closes[-1]
    to_high = (high - lc) / lc * 100.0 if lc else float("nan")
    to_low = (lc - low) / lc * 100.0 if lc else float("nan")
    return {"to_high_pct": to_high, "to_low_pct": to_low}

# ------------------------ configuration ---------------------------------------

DEFAULT_TFS = ["15m", "1h", "4h"]
FALLBACK_LIMIT = 240
ALWAYS_INCLUDE = ["GALA/USDT", "XLM/USDT"]

@dataclass
class SnapshotParams:
    timeframes: List[str]
    candles_limit: int = FALLBACK_LIMIT
    exchange_name: str = "bitget"
    favorites: Optional[List[str]] = None
    universe: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def with_defaults(timeframes: Optional[List[str]] = None, **kw) -> "SnapshotParams":
        return SnapshotParams(timeframes=timeframes or DEFAULT_TFS, **kw)

# ------------------------ feature computation ---------------------------------

def _compute_features_for_tf(ohlcv: List[List[float]]) -> Dict[str, Any]:
    feats = {
        "last": last_close(ohlcv),
        "rsi": rsi([x[4] for x in ohlcv], period=14),
        "atr_pct": atr_pct(ohlcv, period=14),
        "body_pct": body_pct(ohlcv),
        "vol_z": volume_zscore(ohlcv, lookback=50),
    }
    feats.update(dist_to_extremes_pct(ohlcv, lookback=120))
    return feats

# ------------------------ snapshot builder (parallel) -------------------------

def build_snapshot_v41(params: SnapshotParams) -> Dict[str, Any]:
    """
    Fast multi-timeframe snapshot:
      * builds ONLY for `params.universe` (or favorites)
      * downloads per symbol in a small thread pool (rate-limit friendly)
    """
    ex = make_exchange(params.exchange_name)

    base = params.universe or params.favorites or []
    if not base:
        raise ValueError("Universe is empty. Provide favorites or universe.")

    all_symbols = sorted(set(base + ALWAYS_INCLUDE))

    def work(symbol: str) -> Tuple[str, Any]:
        try:
            tf_blocks: Dict[str, Any] = {}
            for tf in params.timeframes:
                ohlcv = fetch_ohlcv_safe(ex, symbol, tf, params.candles_limit)
                tf_blocks[tf] = {
                    "features": _compute_features_for_tf(ohlcv),
                    "ohlcv_len": len(ohlcv),
                    "timeframe": tf,
                }
            tick = fetch_ticker_safe(ex, symbol)
            return symbol, {
                "symbol": symbol,
                "timeframes": tf_blocks,
                "tick": {
                    "last": tick.get("last"),
                    "bid": tick.get("bid"),
                    "ask": tick.get("ask"),
                    "ts": tick.get("timestamp"),
                },
                "meta": {"exchange": ex.id},
            }
        except Exception as e:
            return symbol, {"symbol": symbol, "error": str(e)}

    items: Dict[str, Any] = {}
    # 4–6 threads is generally safe for Bitget public endpoints
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(work, s): s for s in all_symbols}
        for fut in as_completed(futures):
            s, item = fut.result()
            items[s] = item

    return {
        "version": "4.1",
        "timeframes": params.timeframes,
        "candles_limit": params.candles_limit,
        "exchange": ex.id,
        "items": items,
        "meta": params.meta or {},
    }
