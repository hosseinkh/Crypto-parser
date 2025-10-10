# utiles/snapshot.py
from __future__ import annotations
import statistics
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

# --- >>> ADDED: helpers for exit intelligence metrics -------------------------

def _vpr10_features(ohlcv: List[List[float]]) -> Dict[str, Any]:
    """
    Volume Pulse Ratio over 10 bars:
      vpr10 = current_volume / mean(prev 10 volumes)
    Also exposes a momentum-fade flag if last 3 vpr10 readings < 0.8.
    """
    vols = [x[5] for x in ohlcv]
    out = {"vpr10": float("nan"), "vpr10_lt_0_8_last3": False}
    if len(vols) < 12:  # need at least 11 bars for first vpr10
        return out
    denom = statistics.mean(vols[-11:-1]) or 1e-12
    out["vpr10"] = vols[-1] / denom

    # compute last 3 vpr10 values
    try:
        vprs = []
        for k in (0, 1, 2):
            denom_k = statistics.mean(vols[-(11 + k):-(1 + k)]) or 1e-12
            vprs.append(vols[-(1 + k)] / denom_k)
        out["vpr10_lt_0_8_last3"] = all(v < 0.8 for v in vprs)
    except Exception:
        pass
    return out

def _last_two(vals: List[float]) -> Tuple[float, float]:
    """Return (last, previous) with NaNs if unavailable."""
    if len(vals) >= 2:
        return vals[-1], vals[-2]
    return float("nan"), float("nan")

def _rsi_prev(values: List[float], period: int = 14) -> float:
    """
    Previous-bar RSI value (compute RSI on all but the last bar).
    Returns NaN if not enough data.
    """
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
    include_sentiment: bool = True      # ✅ new flag
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

    # --- >>> ADDED: VPR10 metrics (volume pulse ratio) ------------------------
    vpr_block = _vpr10_features(ohlcv)
    feats.update(vpr_block)  # adds feats["vpr10"], feats["vpr10_lt_0_8_last3"]

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
            ohlcv_by_tf: Dict[str, List[List[float]]] = {}   # --- >>> ADDED

            for tf in params.timeframes:
                ohlcv = fetch_ohlcv_safe(ex, symbol, tf, params.candles_limit)
                ohlcv_by_tf[tf] = ohlcv                         # --- >>> ADDED
                tf_blocks[tf] = {
                    "features": _compute_features_for_tf(ohlcv),
                    "ohlcv_len": len(ohlcv),
                    "timeframe": tf,
                }
            tick = fetch_ticker_safe(ex, symbol)

            # ✅ Sentiment (optional)
            sentiment_block = None
            if params.include_sentiment:
                sent_score, sent_details = get_sentiment_for_symbol(symbol)
                sentiment_block = {
                    "score": sent_score,          # [-1 .. +1]
                    "details": sent_details,      # provider breakdown
                }

            # --- >>> ADDED: Derived micro-momentum divergence (5m vs 15m) ----
            derived_block = None
            if ("5m" in ohlcv_by_tf) and ("15m" in tf_blocks):
                closes_5m = [x[4] for x in ohlcv_by_tf["5m"]]
                closes_15m = [x[4] for x in ohlcv_by_tf["15m"]]
                rsi5_now = rsi(closes_5m, period=14)
                rsi5_prev = _rsi_prev(closes_5m, period=14)
                rsi15_now = tf_blocks["15m"]["features"]["rsi"]
                c_now, c_prev = _last_two(closes_5m)

                rsi_gap = rsi5_now - rsi15_now
                price_higher_high = (c_now > c_prev)
                rsi5_lower_high = (rsi5_now < rsi5_prev)

                # Exhaustion when 5m underperforms 15m by ≥5 AND price HH while RSI LH
                exhaustion = (rsi_gap <= -5.0) and price_higher_high and rsi5_lower_high

                derived_block = {
                    "micro_divergence": {
                        "rsi5_now": rsi5_now,
                        "rsi5_prev": rsi5_prev,
                        "rsi15_now": rsi15_now,
                        "rsi_gap": rsi_gap,
                        "price_higher_high": bool(price_higher_high),
                        "rsi5_lower_high": bool(rsi5_lower_high),
                        "exhaustion": bool(exhaustion),
                    }
                }

            return symbol, {
                "symbol": symbol,
                "timeframes": tf_blocks,
                "tick": {
                    "last": tick.get("last"),
                    "bid": tick.get("bid"),
                    "ask": tick.get("ask"),
                    "ts": tick.get("timestamp"),
                },
                "sentiment": sentiment_block,     # ✅ included
                "derived": derived_block,         # --- >>> ADDED (optional)
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

    # --- >>> ADDED: BTC Anchor Bias (market context) -------------------------
    def _get_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in d:
                return k
        return None

    market_block = {}
    btc_key = _get_first_key(items, ["BTC/USDT", "BTCUSDT", "BTC-USDT"])
    if btc_key and ("15m" in items[btc_key].get("timeframes", {})) and ("1h" in items[btc_key]["timeframes"]):
        btc_15 = items[btc_key]["timeframes"]["15m"]["features"].get("rsi")
        btc_1h = items[btc_key]["timeframes"]["1h"]["features"].get("rsi")

        # Simple definition: bear bias if short-term weak AND 1h below neutral
        bear_bias = (isinstance(btc_15, float) and btc_15 < 45.0) and (isinstance(btc_1h, float) and btc_1h < 50.0)

        market_block["btc_anchor_bias"] = {
            "bear": bool(bear_bias),
            "rsi15": btc_15,
            "rsi1h": btc_1h,
            "anchor_symbol": btc_key,
        }

        # Mirror boolean on each symbol for convenience
        for sym, item in items.items():
            if isinstance(item, dict):
                item.setdefault("derived", {})
                item["derived"]["btc_bear_bias"] = bool(bear_bias)
    # -------------------------------------------------------------------------

    return {
        "version": "4.1",
        "timeframes": params.timeframes,
        "candles_limit": params.candles_limit,
        "exchange": ex.id,
        "items": items,
        "market": market_block,   # --- >>> ADDED (top-level)
        "meta": params.meta or {},
    }
