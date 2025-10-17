# utiles/snapshot.py
from __future__ import annotations
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .bitget import make_exchange, fetch_ohlcv_safe, fetch_ticker_safe
from .sentiment import get_sentiment_for_symbol

# ─────────────────────────── Indicators (self-contained) ──────────────────────

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
        h = ohlcv[-i][2]
        l = ohlcv[-i][3]
        c_prev = ohlcv[-i - 1][4]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    atr = sum(trs) / period
    lc = ohlcv[-1][4]
    return (atr / lc) * 100.0 if lc else float("nan")

def body_pct(ohlcv: List[List[float]]) -> float:
    """Body as % of candle range (close-open vs high-low)."""
    if not ohlcv:
        return float("nan")
    o, h, l, c = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
    rng = (h - l) if (h is not None and l is not None) else 0.0
    if not rng:
        return 0.0
    return abs(c - o) / rng * 100.0

def body_pct_open(ohlcv: List[List[float]]) -> float:
    """Absolute % change from open (|c-o|/o)."""
    if not ohlcv:
        return float("nan")
    o, c = ohlcv[-1][1], ohlcv[-1][4]
    if not o:
        return float("nan")
    return abs(c - o) / o

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
    """Simple EMA for the full series; returns the latest EMA value."""
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
    high = max(closes)
    low = min(closes)
    lc = closes[-1]
    to_high = (high - lc) / lc * 100.0 if lc else float("nan")
    to_low = (lc - low) / lc * 100.0 if lc else float("nan")
    return {"to_high_pct": to_high, "to_low_pct": to_low}

# ── VPR10 helpers (supports exit-intelligence + volume-quality diagnostics) ───

def _vpr10_features(ohlcv: List[List[float]]) -> Dict[str, Any]:
    """
    Returns:
      - vpr10: volume / SMA10(volume)
      - vpr10_lt_0_8_last3: count of last 3 bars where vpr10 < 0.8
      - vpr10_lt_0_75_last3: bool (all of last 3 < 0.75)
      - vpr10_drop_pct_from_peak_last3: drop % from last-10 peak VPR to current
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

    # current vpr10 (last closed bar)
    sma10 = sum(vols[-11:-1]) / 10.0 if len(vols) >= 11 else float("nan")
    vpr10_now = vols[-1] / (sma10 or 1e-12) if sma10 == sma10 else float("nan")
    out["vpr10"] = vpr10_now

    # last 3 vpr flags
    last3 = vols[-3:]
    last3_vprs: List[float] = []
    lt_08_count = 0
    lt_075_all = True
    for i in range(3):
        j_end = len(vols) - (2 - i)
        j_start = j_end - 10
        if j_start >= 0:
            ma = sum(vols[j_start:j_end]) / 10.0
            v = last3[i] / (ma or 1e-12)
            last3_vprs.append(v)
            if v < 0.8:
                lt_08_count += 1
            if v >= 0.75:
                lt_075_all = False
        else:
            last3_vprs.append(float("nan"))
            lt_075_all = False
    out["vpr10_lt_0_8_last3"] = lt_08_count
    out["vpr10_lt_0_75_last3"] = bool(lt_075_all)

    # drop from peak (last 10 bars window) to current
    vpr_series = []
    for idx in range(max(0, len(vols) - 10), len(vols)):
        m = sum(vols[max(0, idx - 10):idx]) / 10.0 if idx - 10 >= 0 else float("nan")
        v = vols[idx] / (m or 1e-12) if m == m else float("nan")
        if isinstance(v, (int, float)) and v == v:
            vpr_series.append(v)
    if vpr_series:
        peak = max(vpr_series)
        if isinstance(vpr10_now, float) and vpr10_now == vpr10_now and peak > 0:
            out["vpr10_drop_pct_from_peak_last3"] = (peak - vpr10_now) / peak * 100.0
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

# ───────────────────────────── Configuration ──────────────────────────────────

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
    meta: Optional[Dict[str, Any]] = None  # e.g. {"positions": {sym: {"entry_price": ...}}}

    @classmethod
    def with_defaults(cls,
                      timeframes: Optional[List[str]] = None,
                      universe: Optional[List[str]] = None,
                      favorites: Optional[List[str]] = None,
                      candles_limit: int = FALLBACK_LIMIT,
                      include_sentiment: bool = True,
                      meta: Optional[Dict[str, Any]] = None,
                      exchange_name: str = "bitget") -> "SnapshotParams":
        return cls(
            timeframes=timeframes or DEFAULT_TFS,
            candles_limit=candles_limit,
            exchange_name=exchange_name,
            favorites=favorites,
            universe=universe,
            include_sentiment=include_sentiment,
            meta=meta or {},
        )

# ───────────────────────── TF feature pack (per timeframe) ────────────────────

def _compute_features_for_tf(ohlcv: List[List[float]]) -> Dict[str, Any]:
    closes = [x[4] for x in ohlcv]
    feats = {
        "last": last_close(ohlcv),
        "rsi": rsi(closes, period=14),
        "atr_pct": atr_pct(ohlcv, period=14),
        "body_pct": body_pct(ohlcv),             # body vs range %
        "body_pct_open": body_pct_open(ohlcv),   # abs % vs open (fraction)
        "vol_z": volume_zscore(ohlcv, lookback=50),
        "ema20": ema(closes, period=20),
        "ema50": ema(closes, period=50),
    }
    feats.update(dist_to_extremes_pct(ohlcv, lookback=120))

    # Entry bar volume vs MA20 ratio (used for trigger quality checks)
    def _vol_ma20_ratio(series: List[List[float]]) -> float:
        if not series or len(series) < 21:
            return float("nan")
        vols = [x[5] for x in series]
        ma20 = sum(vols[-21:-1]) / 20.0
        return vols[-1] / (ma20 or 1e-12)
    feats["entry_volume_ma_ratio"] = _vol_ma20_ratio(ohlcv)

    # VPR10 block
    feats.update(_vpr10_features(ohlcv))
    return feats

# ───────────────────────────── Main snapshot builder ──────────────────────────

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
                    "last_candles": (ohlcv[-50:] if ohlcv else []),
                }

            tick = fetch_ticker_safe(ex, symbol)

            # Sentiment (optional)
            sentiment_block = None
            if params.include_sentiment:
                sent_score, sent_details = get_sentiment_for_symbol(symbol)
                sentiment_block = {
                    "score": sent_score,
                    "details": sent_details,
                }

            # ───────────── Derived blocks ─────────────
            derived_block: Dict[str, Any] = {}

            # Higher-lows counters (5m / 15m)
            def _swing_lows(closes: List[float], left=1, right=1):
                lows = []
                n = len(closes)
                for i in range(left, n - right):
                    c = closes[i]
                    if all(c <= closes[i - k] for k in range(1, left + 1)) and all(c < closes[i + k] for k in range(1, right + 1)):
                        lows.append((i, c))
                return lows

            def _consecutive_higher_lows(closes: List[float], lookback=40) -> int:
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

            # Micro-divergence (5m vs 15m)
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

            # Early partial (SET) inputs
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
                try:
                    atr_gain_multiple = profit_pct / atr15_pct if (isinstance(atr15_pct, float) and atr15_pct) else float("nan")
                except Exception:
                    atr_gain_multiple = float("nan")

                derived_block["early_partial"] = {
                    "enabled": True,
                    "profit_pct": profit_pct,
                    "atr_gain_multiple_since_entry": atr_gain_multiple,
                    "vpr10_drop_pct_from_peak_last3": vpr_drop_pct,
                }

            # Return symbol block
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

    # ─────────────────────────── BTC Anchor Bias (Market) ──────────────────────
    def _get_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in d:
                return k
        return None

    market_block: Dict[str, Any] = {}
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
            "btc_1h_above_ema50": (
                isinstance(btc_1h_close, float)
                and isinstance(btc_1h_ema50, float)
                and btc_1h_close > btc_1h_ema50
            ),
        }

        # RSI15 momentum delta + lower-low-with-volume on 15m
        try:
            btc_15_block = items[btc_key]["timeframes"]["15m"]
            tail = btc_15_block.get("last_candles") or []
            closes_15 = [x[4] for x in tail] if tail else []
            rsi15_now = btc_15_block["features"].get("rsi")
            rsi15_prev = _rsi_prev(closes_15, period=14) if closes_15 else float("nan")

            def _vol_ma20_ratio_tail(t) -> float:
                if not t or len(t) < 21:
                    return float("nan")
                vols = [x[5] for x in t]
                ma20 = sum(vols[-21:-1]) / 20.0
                return vols[-1] / (ma20 or 1e-12)

            def _new_lower_low_tail(t) -> bool:
                if not t or len(t) < 5:
                    return False
                lows = [x[3] for x in t]
                return lows[-1] < min(lows[-3:-1])

            ll_ratio = _vol_ma20_ratio_tail(tail)
            ll_with_vol = bool(_new_lower_low_tail(tail) and isinstance(ll_ratio, float) and ll_ratio >= 1.2)

            market_block["btc_anchor_bias"]["rsi15_drop_pts"] = (
                (rsi15_now - rsi15_prev)
                if isinstance(rsi15_now, float) and isinstance(rsi15_prev, float)
                else float("nan")
            )
            market_block["btc_anchor_bias"]["ll_with_volume_15m"] = ll_with_vol
        except Exception:
            pass

        # Mirror flags to each symbol for convenience
        for s, block in items.items():
            try:
                if "timeframes" in block:
                    block.setdefault("market_flags", {})
                    block["market_flags"]["btc_bear"] = bool(bear_bias)
                    block["market_flags"]["btc_1h_above_ema50"] = market_block["btc_anchor_bias"]["btc_1h_above_ema50"]
                    # also mirror inside derived for quick reads (as seen in your snapshot)
                    block.setdefault("derived", {})
                    block["derived"]["btc_anchor_bias_bear"] = bool(bear_bias)
            except Exception:
                pass

    # ───────────────────────────────── Return ─────────────────────────────────

    return {
        "version": "4.1",
        "timeframes": params.timeframes,
        "candles_limit": params.candles_limit,
        "exchange": ex.id,
        "items": items,
        "market": market_block,
        "meta": {
            **(params.meta or {}),
            "now_ts": now_ms,   # shared clock used for closed-candle alignment
        },
    }
