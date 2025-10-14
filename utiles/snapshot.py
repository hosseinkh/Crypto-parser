# utiles/snapshot.py
from __future__ import annotations
import time
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

def ema(values: List[float], period: int) -> float:
    """
    Simple EMA for the full series; returns the latest EMA value.
    """
    if len(values) < period:
        return float("nan")
    k = 2.0 / (period + 1.0)
    ema_val = statistics.mean(values[:period])
    for v in values[period:]:
        ema_val = v * k + ema_val * (1.0 - k)
    return ema_val

def dist_to_extremes_pct(ohlcv: List[List[float]], lookback: int = 120) -> Dict[str, float]:
    closes = [x[4] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if not closes:
        return {"to_high_pct": float("nan"), "to_low_pct": float("nan")}
    high = max(closes); low = min(closes); lc = closes[-1]
    to_high = (high - lc) / lc * 100.0 if lc else float("nan")
    to_low = (lc - low) / lc * 100.0 if lc else float("nan")
    return {"to_high_pct": to_high, "to_low_pct": to_low}

# --- enhanced: helpers for exit-intelligence metrics --------------------------

def _vpr10_features(ohlcv: List[List[float]]) -> Dict[str, Any]:
    """
    Volume Pulse Ratio (VPR10):
      vpr10 = current_volume / mean(prev 10 volumes)
    Also returns:
      - vpr10_lt_0_8_last3 (existing)
      - vpr10_peak_last3 (peak within the last 3 completed bars)
      - vpr10_drop_pct_from_peak_last3 (drop from that peak to current, in %)
    """
    vols = [x[5] for x in ohlcv]
    out = {
        "vpr10": float("nan"),
        "vpr10_lt_0_8_last3": False,
        "vpr10_peak_last3": float("nan"),
        "vpr10_drop_pct_from_peak_last3": float("nan"),
    }
    if len(vols) < 12:
        return out

    # current
    denom = statistics.mean(vols[-11:-1]) or 1e-12
    vpr_now = vols[-1] / denom
    out["vpr10"] = vpr_now

    # last 3 completed bars (including current as "bar 0")
    vprs = []
    peakscope = []
    try:
        for k in (0, 1, 2):
            denom_k = statistics.mean(vols[-(11 + k):-(1 + k)]) or 1e-12
            vpr_k = vols[-(1 + k)] / denom_k
            vprs.append(vpr_k)
            peakscope.append(vpr_k)
        out["vpr10_lt_0_8_last3"] = all(v < 0.8 for v in vprs)
        peak_last3 = max(peakscope)
        out["vpr10_peak_last3"] = peak_last3
        if peak_last3 > 0:
            out["vpr10_drop_pct_from_peak_last3"] = (peak_last3 - vpr_now) / peak_last3 * 100.0
    except Exception:
        pass

    return out

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
    meta: Optional[Dict[str, Any]] = None  # can carry positions: {"positions": {sym: {"entry_price": ...}}}

    @staticmethod
    def with_defaults(timeframes: Optional[List[str]] = None, **kw) -> "SnapshotParams":
        return SnapshotParams(timeframes=timeframes or DEFAULT_TFS, **kw)

# ------------------------ feature computation ---------------------------------

def _compute_features_for_tf(ohlcv: List[List[float]]) -> Dict[str, Any]:
    closes = [x[4] for x in ohlcv]
    feats = {
        "last": last_close(ohlcv),
        "rsi": rsi(closes, period=14),
        "atr_pct": atr_pct(ohlcv, period=14),
        "body_pct": body_pct(ohlcv),
        "vol_z": volume_zscore(ohlcv, lookback=50),
        "ema50": ema(closes, period=50),   # NEW: for BTC_1h EMA50 check
    }
    feats.update(dist_to_extremes_pct(ohlcv, lookback=120))

    # --- VPR10 metrics (Layer 2 / L0 uses the peak & drop too)
    vpr_block = _vpr10_features(ohlcv)
    feats.update(vpr_block)

    return feats

# ------------------------ snapshot builder (parallel) -------------------------

def build_snapshot_v41(params: SnapshotParams) -> Dict[str, Any]:
    """
    Fast multi-timeframe snapshot with optional sentiment per symbol.
    Enforces closed-candle alignment across TFs by sharing a single now_ts.
    """
    ex = make_exchange(params.exchange_name)

    base = params.universe or params.favorites or []
    if not base:
        raise ValueError("Universe is empty. Provide favorites or universe.")

    all_symbols = sorted(set(base + ALWAYS_INCLUDE))

    # Shared clock for closed-candle alignment
    now_ms = int(time.time() * 1000)

    # Optional positions map for profit calculations:
    # params.meta = {"positions": {"SOL/USDT": {"entry_price": 132.5, "entry_ts": 1699999999000}, ...}}
    positions_map: Dict[str, Dict[str, Any]] = {}
    if params.meta and isinstance(params.meta.get("positions"), dict):
        positions_map = params.meta["positions"]

    def work(symbol: str) -> Tuple[str, Any]:
        try:
            tf_blocks: Dict[str, Any] = {}
            ohlcv_by_tf: Dict[str, List[List[float]]] = {}

            for tf in params.timeframes:
                ohlcv = fetch_ohlcv_safe(
                    ex, symbol, tf, params.candles_limit,
                    since=None,
                    closed_only=True,            # enforce closed candles
                    now_ts=now_ms,               # align all TFs to the same boundary
                )
                ohlcv_by_tf[tf] = ohlcv
                tf_blocks[tf] = {
                    "features": _compute_features_for_tf(ohlcv),
                    "ohlcv_len": len(ohlcv),
                    "timeframe": tf,
                    "last_closed_ts": (ohlcv[-1][0] if ohlcv else None),
                }

            tick = fetch_ticker_safe(ex, symbol)

            # Sentiment (optional)
            sentiment_block = None
            if params.include_sentiment:
                sent_score, sent_details = get_sentiment_for_symbol(symbol)
                sentiment_block = {
                    "score": sent_score,          # [-1 .. +1]
                    "details": sent_details,      # provider breakdown
                }

            # --- Derived blocks ------------------------------------------------

            derived_block: Dict[str, Any] = {}

            # L1: micro-divergence (requires 5m + 15m)
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

            # L0: early partial (Surge Exhaustion)
            # Needs: profit_pct (if entry known), ATR(15m) pct, and VPR10 drop from recent peak.
            early_partial = None
            if "15m" in tf_blocks:
                atr15_pct = tf_blocks["15m"]["features"].get("atr_pct")
                vpr_drop_pct = tf_blocks["15m"]["features"].get("vpr10_drop_pct_from_peak_last3")

                entry_price = None
                if symbol in positions_map:
                    entry_price = positions_map[symbol].get("entry_price")

                profit_pct = float("nan")
                if entry_price and isinstance(entry_price, (int, float)) and entry_price > 0:
                    last_px = tf_blocks["15m"]["features"].get("last")
                    if last_px:
                        profit_pct = (last_px - entry_price) / entry_price * 100.0

                atr_gain_multiple = float("nan")
                if atr15_pct and isinstance(profit_pct, float):
                    # how many ATR(15m) (in %) the unrealized gain represents
                    if atr15_pct > 0:
                        atr_gain_multiple = profit_pct / atr15_pct

                # Condition (defaults from our design):
                # profit ≥ 1.2%, atr_gain_multiple ≥ 1.5, vpr10_drop ≥ 25% from last3 peak
                if (
                    isinstance(profit_pct, float) and profit_pct >= 1.2 and
                    isinstance(atr_gain_multiple, float) and atr_gain_multiple >= 1.5 and
                    isinstance(vpr_drop_pct, float) and vpr_drop_pct >= 25.0
                ):
                    early_partial = True
                elif (
                    isinstance(profit_pct, float) and not (profit_pct != profit_pct)  # not NaN
                ):
                    early_partial = False  # explicitly mark evaluated & false

                derived_block["early_partial"] = {
                    "enabled": True,
                    "profit_pct": profit_pct,
                    "atr_gain_multiple_since_entry": atr_gain_multiple,
                    "vpr10_drop_pct_from_peak_last3": vpr_drop_pct,
                }

            # -------------------------------------------------------------------

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

    # --- BTC Anchor Bias (Layer 3) --------------------------------------------
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
        btc_1h_close = items[btc_key]["timeframes"]["1h"]["features"].get("last")
        btc_1h_ema50 = items[btc_key]["timeframes"]["1h"]["features"].get("ema50")

        # Bear bias if short-term is weak AND 1h is below neutral
        bear_bias = (isinstance(btc_15, float) and btc_15 < 45.0) and (isinstance(btc_1h, float) and btc_1h < 50.0)

        market_block["btc_anchor_bias"] = {
            "bear": bool(bear_bias),
            "rsi15": btc_15,
            "rsi1h": btc_1h,
            "anchor_symbol": btc_key,
            "btc_1h_close": btc_1h_close,
            "btc_1h_ema50": btc_1h_ema50,
            "btc_1h_above_ema50": (isinstance(btc_1h_close, float) and isinstance(btc_1h_ema50, float) and btc_1h_close > btc_1h_ema50),
        }

        # Mirror boolean to each symbol for convenience
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
        "market": market_block,   # includes BTC EMA50 & bear bias
        "meta": {
            **(params.meta or {}),
            "now_ts": now_ms,     # shared clock used for closed-candle alignment
        },
    }
