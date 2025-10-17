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
    high = ohlcv[-1][2]; low = ohlcv[-1][3]
    rng = max(high - low, 1e-12)
    return abs(c - o) / rng * 100.0

def volume_zscore(ohlcv: List[List[float]], lookback: int = 50) -> float:
    if not ohlcv or len(ohlcv) < lookback:
        return float("nan")
    vols = [x[5] for x in ohlcv[-lookback:]]
    mean = statistics.mean(vols)
    stdev = statistics.pstdev(vols) if len(vols) > 1 else 0.0
    if stdev == 0:
        return 0.0
    return (vols[-1] - mean) / stdev

def last_close(ohlcv: List[List[float]]) -> float:
    return ohlcv[-1][4] if ohlcv else float("nan")

def ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return float("nan")
    k = 2.0 / (period + 1.0)
    ema_val = statistics.mean(values[:period])
    for v in values[period:]:
        ema_val = v * k + ema_val * (1.0 - k)
    return ema_val

def dist_to_range_pct(ohlcv: List[List[float]], lookback: int = 120) -> Dict[str, float]:
    """
    Playbooks expect keys: to_high_pct / to_low_pct (percent distances).
    """
    closes = [x[4] for x in (ohlcv[-lookback:] if len(ohlcv) >= lookback else ohlcv)]
    if not closes:
        return {"to_high_pct": float("nan"), "to_low_pct": float("nan")}
    high = max(closes); low = min(closes); lc = closes[-1]
    return {
        "to_high_pct": (high - lc) / (lc or 1e-12) * 100.0,
        "to_low_pct":  (lc - low) / (lc or 1e-12) * 100.0,
    }

# --- helpers for exit-intelligence metrics ------------------------------------

def _vpr10_features(ohlcv: List[List[float]]) -> Dict[str, Any]:
    """
    Returns:
      - vpr10: volume / SMA10(volume)
      - vpr10_lt_0_8_last3: (kept for UI)
      - vpr10_lt_0_75_last3: strict (all last 3 bars < 0.75) — used by Exit L2
      - vpr10_drop_pct_from_peak_last3: drop % from recent peak to current
    """
    out = {
        "vpr10": float("nan"),
        "vpr10_lt_0_8_last3": None,
        "vpr10_lt_0_75_last3": False,
        "vpr10_drop_pct_from_peak_last3": None,
    }
    if not ohlcv or len(ohlcv) < 12:
        return out

    vols = [x[5] for x in ohlcv]
    sma10 = sum(vols[-11:-1]) / 10.0 if len(vols) >= 11 else float("nan")
    vpr10 = vols[-1] / (sma10 or 1e-12) if sma10 == sma10 else float("nan")
    out["vpr10"] = vpr10

    # legacy: count of last 3 bars where vpr10 < 0.80
    flags_08 = 0
    for i in range(3):
        j_end = len(vols) - (2 - i)
        j_start = j_end - 10
        if j_start >= 0:
            ma10 = sum(vols[j_start:j_end]) / 10.0
            vv = vols[j_end - 1] / (ma10 or 1e-12)
            if vv < 0.80:
                flags_08 += 1
    out["vpr10_lt_0_8_last3"] = flags_08

    # strict: ALL last 3 bars below 0.75
    strict_last3 = []
    for i in range(3):
        j_end = len(vols) - (2 - i)
        j_start = j_end - 10
        if j_start >= 0:
            ma10 = sum(vols[j_start:j_end]) / 10.0
            vv = vols[j_end - 1] / (ma10 or 1e-12)
            strict_last3.append(vv < 0.75)
    out["vpr10_lt_0_75_last3"] = (len(strict_last3) == 3 and all(strict_last3))

    # drop from peak of last ~10 vpr points to current
    vpr_series = []
    for idx in range(max(0, len(vols) - 10), len(vols)):
        m = sum(vols[max(0, idx - 10):idx]) / 10.0 if idx - 10 >= 0 else float("nan")
        vpr_series.append(vols[idx] / (m or 1e-12) if m == m else float("nan"))
    vpr_series = [x for x in vpr_series if isinstance(x, (int, float)) and x == x]
    if vpr_series:
        peak = max(vpr_series)
        if isinstance(vpr10, float) and vpr10 == vpr10 and peak > 0:
            out["vpr10_drop_pct_from_peak_last3"] = (peak - vpr10) / peak * 100.0
        else:
            out["vpr10_drop_pct_from_peak_last3"] = float("nan")
    else:
        out["vpr10_drop_pct_from_peak_last3"] = float("nan")

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
    meta: Optional[Dict[str, Any]] = None  # e.g. {"positions": {"SOL/USDT": {"entry_price": ...}}}

    @classmethod
    def with_defaults(
        cls,
        timeframes: Optional[List[str]] = None,
        candles_limit: Optional[int] = None,
        exchange_name: str = "bitget",
        favorites: Optional[List[str]] = None,
        universe: Optional[List[str]] = None,
        include_sentiment: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        ensure_5m: bool = True,
    ) -> "SnapshotParams":
        tfs = list(timeframes) if timeframes else list(DEFAULT_TFS)
        if ensure_5m and "5m" not in tfs:
            tfs = ["5m"] + tfs
        limit = candles_limit if (isinstance(candles_limit, int) and candles_limit > 0) else FALLBACK_LIMIT
        uni = universe or favorites or []
        fav = favorites
        return cls(
            timeframes=tfs,
            candles_limit=limit,
            exchange_name=exchange_name or "bitget",
            favorites=fav,
            universe=uni,
            include_sentiment=include_sentiment if include_sentiment is not None else True,
            meta=meta or {},
        )

# ------------------------ TF feature pack -------------------------------------

def _compute_features_for_tf(ohlcv: List[List[float]]) -> Dict[str, Any]:
    closes = [x[4] for x in ohlcv]
    feats = {
        "last": last_close(ohlcv),
        "rsi": rsi(closes, period=14),
        "atr_pct": atr_pct(ohlcv, period=14),
        "body_pct": body_pct(ohlcv),
        "vol_z": volume_zscore(ohlcv, lookback=50),
        "ema50": ema(closes, period=50),
    }
    feats.update(dist_to_range_pct(ohlcv, lookback=120))

    # entry volume vs MA20 ratio (for 15m trigger checks)
    def _vol_ma20_ratio(series):
        if not ohlcv or len(ohlcv) < 21:
            return float("nan")
        vols = [x[5] for x in ohlcv]
        ma20 = sum(vols[-21:-1]) / 20.0
        return vols[-1] / (ma20 or 1e-12)
    feats["entry_volume_ma_ratio"] = _vol_ma20_ratio(ohlcv)

    feats.update(_vpr10_features(ohlcv))
    return feats

# ------------------------ main snapshot builder --------------------------------

def build_snapshot_v41(params: SnapshotParams) -> Dict[str, Any]:
    """
    Multi-timeframe snapshot with optional sentiment, aligned to closed candles.
    """
    ex = make_exchange(params.exchange_name)

    base = params.universe or params.favorites or []
    if not base:
        raise ValueError("Universe is empty. Provide favorites or universe.")

    all_symbols = sorted(set(base + ALWAYS_INCLUDE))

    # Ensure BTC anchor always computed even if not explicitly in universe
    if "BTC/USDT" not in all_symbols:
        all_symbols.append("BTC/USDT")

    # Shared clock for closed-candle alignment
    now_ms = int(time.time() * 1000)

    # positions map for early-partial computations
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
                    closed_only=True,
                    now_ts=now_ms,
                )
                ohlcv_by_tf[tf] = ohlcv
                tf_blocks[tf] = {
                    "features": _compute_features_for_tf(ohlcv),
                    "ohlcv_len": len(ohlcv),
                    "timeframe": tf,
                    "last_closed_ts": (ohlcv[-1][0] if ohlcv else None),
                    "last_candles": (ohlcv[-50:] if ohlcv else []),  # tail used by BTC 15m checks & L1/L2
                }

            tick = fetch_ticker_safe(ex, symbol)

            # Sentiment (optional)
            sentiment_block = None
            if params.include_sentiment:
                sent_score, sent_details = get_sentiment_for_symbol(symbol)
                sentiment_block = {"score": sent_score, "details": sent_details}

            # -------------------------------------------------------------------
            # Derived blocks ----------------------------------------------------
            derived_block: Dict[str, Any] = {}

            # Higher-lows counters on 5m and 15m
            def _swing_lows(closes, left=1, right=1):
                lows = []
                n = len(closes)
                for i in range(left, n - right):
                    c = closes[i]
                    if all(c <= closes[i - k] for k in range(1, left + 1)) and \
                       all(c <  closes[i + k] for k in range(1, right + 1)):
                        lows.append((i, c))
                return lows

            def _consecutive_higher_lows(closes, lookback=40):
                if len(closes) < max(8, lookback):
                    return 0
                lows = _swing_lows(closes[-lookback:])
                if len(lows) < 2:
                    return 0
                cnt = 1
                for j in range(1, len(lows)):
                    if lows[j][1] > lows[j - 1][1]:
                        cnt += 1
                    else:
                        cnt = 1
                return cnt

            try:
                if "5m" in ohlcv_by_tf:
                    derived_block["higher_lows_5m"] = _consecutive_higher_lows([x[4] for x in ohlcv_by_tf["5m"]])
                if "15m" in ohlcv_by_tf:
                    derived_block["higher_lows_15m"] = _consecutive_higher_lows([x[4] for x in ohlcv_by_tf["15m"]])
            except Exception:
                pass

            # L1: micro-divergence (5m vs 15m) — playbook spec: gap <= -7.0
            if ("5m" in ohlcv_by_tf) and ("15m" in tf_blocks):
                closes_5m = [x[4] for x in ohlcv_by_tf["5m"]]
                rsi5_now = rsi(closes_5m, period=14)
                rsi5_prev = _rsi_prev(closes_5m, period=14)
                rsi15_now = tf_blocks["15m"]["features"]["rsi"]
                c_now, c_prev = _last_two(closes_5m)

                rsi_gap = rsi5_now - rsi15_now
                price_higher_high = (c_now > c_prev)
                rsi5_lower_high = (rsi5_now < rsi5_prev)
                exhaustion = (rsi_gap <= -7.0) and price_higher_high and rsi5_lower_high

                derived_block["micro_divergence"] = {
                    "rsi5_now": rsi5_now,
                    "rsi5_prev": rsi5_prev,
                    "rsi15_now": rsi15_now,
                    "rsi_gap": rsi_gap,
                    "price_higher_high": bool(price_higher_high),
                    "rsi5_lower_high": bool(rsi5_lower_high),
                    "exhaustion": bool(exhaustion),
                }

            # L0: early partial (SET) — profit % vs ATR and VPR drop
            early_partial = None
            if "15m" in tf_blocks:
                atr15_pct = tf_blocks["15m"]["features"].get("atr_pct")
                vpr_drop_pct = tf_blocks["15m"]["features"].get("vpr10_drop_pct_from_peak_last3")

                entry_price = positions_map.get(symbol, {}).get("entry_price")
                profit_pct = float("nan")
                if entry_price and isinstance(entry_price, (int, float)) and entry_price > 0:
                    last_px = tf_blocks["15m"]["features"].get("last")
                    if last_px:
                        profit_pct = (last_px - entry_price) / entry_price * 100.0

                atr_gain_multiple = float("nan")
                if isinstance(atr15_pct, float) and atr15_pct == atr15_pct and isinstance(profit_pct, float):
                    atr_gain_multiple = (profit_pct / atr15_pct) if atr15_pct else float("nan")

                early_partial = {
                    "profit_pct": profit_pct,
                    "atr15_pct": atr15_pct,
                    "atr_gain_multiple": atr_gain_multiple,
                    "vpr10_drop_pct_from_peak_last3": vpr_drop_pct,
                }

            if early_partial:
                derived_block["early_partial"] = {
                    "enabled": True,
                    "profit_pct": early_partial["profit_pct"],
                    "atr_gain_multiple_since_entry": early_partial["atr_gain_multiple"],
                    "vpr10_drop_pct_from_peak_last3": early_partial["vpr10_drop_pct_from_peak_last3"],
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

        # NEW: rsi15_drop_pts and ll_with_volume_15m (from 15m tail)
        try:
            btc_15_block = items[btc_key]["timeframes"]["15m"]
            tail = btc_15_block.get("last_candles") or []
            closes_15 = [x[4] for x in tail] if tail else []
            rsi15_now = btc_15_block["features"].get("rsi")
            rsi15_prev = _rsi_prev(closes_15, period=14) if closes_15 else float("nan")

            def _vol_ma20_ratio_tail(t):
                if not t or len(t) < 21:
                    return float("nan")
                vols = [x[5] for x in t]
                ma20 = sum(vols[-21:-1]) / 20.0
                return vols[-1] / (ma20 or 1e-12)

            def _new_lower_low_tail(t):
                if not t or len(t) < 5:
                    return False
                lows = [x[3] for x in t]
                return lows[-1] < min(lows[-3:-1])

            ll_ratio = _vol_ma20_ratio_tail(tail)
            ll_with_vol = bool(_new_lower_low_tail(tail) and isinstance(ll_ratio, float) and ll_ratio >= 1.2)

            market_block["btc_anchor_bias"]["rsi15_drop_pts"] = (rsi15_now - rsi15_prev) if isinstance(rsi15_now, float) and isinstance(rsi15_prev, float) else float("nan")
            market_block["btc_anchor_bias"]["ll_with_volume_15m"] = ll_with_vol
        except Exception:
            pass

        # Mirror booleans into each symbol for playbook convenience
        for s, block in items.items():
            try:
                if "timeframes" in block:
                    block.setdefault("derived", {})
                    block.setdefault("market_flags", {})
                    block["market_flags"]["btc_bear"] = bool(bear_bias)
                    block["market_flags"]["btc_1h_above_ema50"] = market_block["btc_anchor_bias"]["btc_1h_above_ema50"]
                    # Alias used by Exit Intelligence L3:
                    block["derived"]["btc_anchor_bias_bear"] = bool(bear_bias)
            except Exception:
                pass

    # -------------------------------------------------------------------------

    return {
        "version": "4.1",
        "timeframes": params.timeframes,
        "candles_limit": params.candles_limit,
        "exchange": ex.id,
        "items": items,
        "market": market_block,
        "meta": {
            **(params.meta or {}),
            "now_ts": now_ms,
        },
    }
