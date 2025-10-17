# utiles/snapshot.py
from __future__ import annotations
import time, statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .bitget import make_exchange, fetch_ohlcv_safe, fetch_ticker_safe
from .sentiment import get_sentiment_for_symbol

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

# --- Added: helpers for exit-intelligence & v4.3 structure checks -------------

def _vpr10_features(ohlcv: List[List[float]]) -> Dict[str, Any]:
    vols = [x[5] for x in ohlcv]
    out = {"vpr10": float("nan"), "vpr10_lt_0_8_last3": False}
    if len(vols) < 12:
        return out
    denom = statistics.mean(vols[-11:-1]) or 1e-12
    out["vpr10"] = vols[-1] / denom

    try:
        vprs = []
        for k in (0, 1, 2):
            denom_k = statistics.mean(vols[-(11 + k):-(1 + k)]) or 1e-12
            vprs.append(vols[-(1 + k)] / denom_k)
        out["vpr10_lt_0_8_last3"] = all(v < 0.8 for v in vprs)
    except Exception:
        pass
    return out

def _sma(vals: List[float], n: int) -> float:
    if len(vals) < n:
        return float("nan")
    return sum(vals[-n:]) / float(n)

def _lower_low_with_volume_15m(ohlcv_15: List[List[float]], vol_ma_len: int = 20, vol_mult: float = 1.2) -> bool:
    """Return True if last swing makes a lower low AND that bar's volume >= vol_mult * MA(vol)."""
    if len(ohlcv_15) < max(vol_ma_len + 2, 10):
        return False
    lows = [x[3] for x in ohlcv_15]
    vols = [x[5] for x in ohlcv_15]
    # naive swing: compare last low vs previous low
    last_low = lows[-1]
    prev_low = min(lows[-4:-1])  # small window
    vol_ma = _sma(vols[:-1], vol_ma_len)
    return (last_low < prev_low) and (vols[-1] >= (vol_ma * vol_mult if vol_ma == vol_ma else float("inf")))

def _count_higher_lows(ohlcv: List[List[float]], lookback: int = 20) -> int:
    """Count consecutive higher lows ending at last candle."""
    if len(ohlcv) < 3:
        return 0
    lows = [x[3] for x in ohlcv[-lookback:]]
    cnt = 0
    for i in range(len(lows) - 1, 0, -1):
        if lows[i] > lows[i - 1]:
            cnt += 1
        else:
            break
    return cnt

def _first_bounce_after_flush(ohlcv: List[List[float]], lookback: int = 60, drop_pct: float = 3.0) -> bool:
    """
    Heuristic: detect a sharp flush (>= drop_pct%) over recent bars, and if current structure shows first higher-low bounce.
    """
    if len(ohlcv) < lookback:
        return False
    closes = [x[4] for x in ohlcv[-lookback:]]
    if not closes or not closes[-1]:
        return False
    peak = max(closes)  # recent local peak in window
    curr = closes[-1]
    drop = (peak - curr) / peak * 100.0 if peak else 0.0
    higher_lows = _count_higher_lows(ohlcv, lookback=20)
    return (drop >= drop_pct) and (higher_lows == 1)

def _last_two(vals: List[float]) -> Tuple[float, float]:
    if len(vals) >= 2:
        return vals[-1], vals[-2]
    return float("nan"), float("nan")

def _rsi_prev(values: List[float], period: int = 14) -> float:
    if len(values) < period + 2:
        return float("nan")
    return rsi(values[:-1], period=period)

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
    include_sentiment: bool = True
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
    feats.update(_vpr10_features(ohlcv))  # adds vpr10 & vpr10_lt_0_8_last3
    return feats

# ------------------------ snapshot builder (parallel) -------------------------

def build_snapshot_v41(params: SnapshotParams) -> Dict[str, Any]:
    """
    Fast multi-timeframe snapshot with optional sentiment per symbol.
    """
    ex = make_exchange(params.exchange_name)

    base = params.universe or params.favorites or []
    if not base:
        raise ValueError("Universe is empty. Provide favorites or universe.")

    all_symbols = sorted(set(base + ALWAYS_INCLUDE))

    def work(symbol: str) -> Tuple[str, Any]:
        try:
            tf_blocks: Dict[str, Any] = {}
            ohlcv_by_tf: Dict[str, List[List[float]]] = {}

            for tf in params.timeframes:
                ohlcv = fetch_ohlcv_safe(ex, symbol, tf, params.candles_limit)
                ohlcv_by_tf[tf] = ohlcv
                tf_blocks[tf] = {
                    "features": _compute_features_for_tf(ohlcv),
                    "ohlcv_len": len(ohlcv),
                    "timeframe": tf,
                }

            tick = fetch_ticker_safe(ex, symbol)

            sentiment_block = None
            if params.include_sentiment:
                sent_score, sent_details = get_sentiment_for_symbol(symbol)
                sentiment_block = {"score": sent_score, "details": sent_details}

            # Derived structure bits for v4.3 gates (if we have 5m/15m)
            derived_block: Dict[str, Any] = {}

            # L1 micro-divergence (optional, only if 5m present)
            if ("5m" in ohlcv_by_tf) and ("15m" in tf_blocks):
                closes_5m = [x[4] for x in ohlcv_by_tf["5m"]]
                rsi5_now = rsi(closes_5m, period=14)
                rsi5_prev = _rsi_prev(closes_5m, period=14)
                rsi15_now = tf_blocks["15m"]["features"]["rsi"]
                c_now, c_prev = _last_two(closes_5m)

                rsi_gap = rsi5_now - rsi15_now
                price_higher_high = (c_now > c_prev)
                rsi5_lower_high = (rsi5_now < rsi5_prev)
                exhaustion = (rsi_gap <= -5.0) and price_higher_high and rsi5_lower_high

                derived_block["micro_divergence"] = {
                    "rsi5_now": rsi5_now,
                    "rsi5_prev": rsi5_prev,
                    "rsi15_now": rsi15_now,
                    "rsi_gap": rsi_gap,
                    "price_higher_high": bool(price_higher_high),
                    "rsi5_lower_high": bool(rsi5_lower_high),
                    "exhaustion": bool(exhaustion),
                }

            # NEW: structure metrics for gates
            struct_block: Dict[str, Any] = {}
            if "5m" in ohlcv_by_tf:
                struct_block["higher_lows_5m"] = _count_higher_lows(ohlcv_by_tf["5m"], lookback=20)
                struct_block["is_first_bounce_after_flush_5m"] = _first_bounce_after_flush(ohlcv_by_tf["5m"])
            if "15m" in ohlcv_by_tf:
                struct_block["higher_lows_15m"] = _count_higher_lows(ohlcv_by_tf["15m"], lookback=20)
                struct_block["is_first_bounce_after_flush_15m"] = _first_bounce_after_flush(ohlcv_by_tf["15m"])

            if struct_block:
                derived_block.setdefault("structure", {}).update(struct_block)

            return symbol, {
                "symbol": symbol,
                "timeframes": tf_blocks,
                "tick": {
                    "last": tick.get("last"),
                    "bid": tick.get("bid"),
                    "ask": tick.get("ask"),
                    "ts": tick.get("timestamp"),
                },
                "sentiment": sentiment_block,
                "derived": derived_block if derived_block else None,
                "meta": {"exchange": ex.id},
            }
        except Exception as e:
            return symbol, {"symbol": symbol, "error": str(e)}

    items: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(work, s): s for s in all_symbols}
        for fut in as_completed(futures):
            s, item = fut.result()
            items[s] = item

    # --- BTC Anchor Bias + v4.3 extra fields ----------------------------------
    market_block: Dict[str, Any] = {}
    btc_key = None
    for k in ("BTC/USDT", "BTCUSDT", "BTC-USDT"):
        if k in items:
            btc_key = k
            break

    if btc_key:
        btc_tfs = (items[btc_key] or {}).get("timeframes", {})
        btc_15_rsi = (btc_tfs.get("15m", {}).get("features", {}) or {}).get("rsi")
        btc_1h_rsi = (btc_tfs.get("1h",  {}).get("features", {}) or {}).get("rsi")
        btc_4h_rsi = (btc_tfs.get("4h",  {}).get("features", {}) or {}).get("rsi")

        # Bear bias (same as before)
        bear_bias = (isinstance(btc_15_rsi, float) and btc_15_rsi < 45.0) and (isinstance(btc_1h_rsi, float) and btc_1h_rsi < 50.0)

        # NEW: lower-low-with-volume on 15m for Master/BTC gate definition
        btc_15m_ohlcv = None
        # Find original candles length param to refetch only if not present; we already have ohlcv from builder:
        try:
            # We didn't store raw ohlcv in items to keep snapshot light; re-fetch a small window here safely:
            ex2 = make_exchange(params.exchange_name)
            btc_15m_ohlcv = fetch_ohlcv_safe(ex2, btc_key.replace("-", "/"), "15m", limit=120)
        except Exception:
            btc_15m_ohlcv = None

        lower_low_w_vol = _lower_low_with_volume_15m(btc_15m_ohlcv) if btc_15m_ohlcv else False

        # NEW: 1h ATR/Close for the calm-exception volatility gate
        atr_over_close_1h = None
        try:
            atr_over_close_1h = (btc_tfs.get("1h", {}).get("features", {}) or {}).get("atr_pct")
        except Exception:
            atr_over_close_1h = None

        market_block["btc_anchor_bias"] = {
            "bear": bool(bear_bias),
            "rsi15": btc_15_rsi,
            "rsi1h": btc_1h_rsi,
            "rsi4h": btc_4h_rsi,
            "lower_low_w_volume_15m": bool(lower_low_w_vol),
            "atr_over_close_1h": atr_over_close_1h,
            "anchor_symbol": btc_key,
        }

        # Mirror boolean to each symbol for convenience
        for sym, item in items.items():
            if isinstance(item, dict):
                item.setdefault("derived", {})
                item["derived"]["btc_bear_bias"] = bool(bear_bias)

    # -------------------------------------------------------------------------

    return {
        "version": "4.2",
        "timeframes": params.timeframes,
        "candles_limit": params.candles_limit,
        "exchange": ex.id,
        "items": items,
        "market": market_block,
        "meta": (params.meta or {}) | {"now_ts": int(time.time() * 1000)},
    }
