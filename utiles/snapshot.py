# utiles/snapshot.py
# -----------------------------------------------------------
# Build an LLM-friendly snapshot (v4.1) with explicit decisions,
# reasons, thresholds, data-quality, and snapshot-time prices.
# -----------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import math
import pandas as pd
import numpy as np

# Project deps
try:
    from utiles.indicators import compute_indicators
except Exception:
    from .indicators import compute_indicators  # type: ignore

try:
    from utiles.bitget import ticker_bitget
except Exception:
    # Safe fallback
    def ticker_bitget(symbol: str) -> Dict[str, Any]:
        return {"last": None, "bid": None, "ask": None, "ts": None}


SCHEMA_VERSION = "4.1"


# ---------------- Time helpers ----------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None or pd.isna(ms):
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return None


# ---------------- Math helpers ----------------
def _pct_change_over_n(df: pd.DataFrame, n: int) -> float:
    if df.shape[0] <= n:
        return float("nan")
    c0 = float(df["close"].iloc[-n - 1])
    c1 = float(df["close"].iloc[-1])
    if c0 == 0:
        return float("nan")
    return (c1 - c0) / c0 * 100.0


def _rolling_z_last(series: pd.Series, window: int, ddof: int = 1) -> float:
    if series.shape[0] < window:
        return float("nan")
    mu = series.rolling(window=window, min_periods=window).mean().iloc[-1]
    sd = series.rolling(window=window, min_periods=window).std(ddof=ddof).iloc[-1]
    if not sd or pd.isna(sd) or sd == 0:
        return 0.0
    return float((series.iloc[-1] - mu) / sd)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _body_wicks(last_open: float, last_high: float, last_low: float, last_close: float) -> Dict[str, float]:
    rng = max(last_high - last_low, 1e-12)
    body = abs(last_close - last_open) / rng * 100.0
    up_wick = (last_high - max(last_close, last_open)) / rng * 100.0
    low_wick = (min(last_close, last_open) - last_low) / rng * 100.0
    direction = "up" if last_close >= last_open else "down"
    return {
        "dir": direction,
        "body_pct": round(body, 2),
        "upper_wick_pct": round(up_wick, 2),
        "lower_wick_pct": round(low_wick, 2),
    }


# --------------- Feature extraction ---------------
def _bars_for(hours: int, timeframe: str) -> int:
    # approximate bar counts for given hours per TF
    mapping = {
        "5m": int(60 / 5) * hours,
        "15m": int(60 / 15) * hours,
        "30m": int(60 / 30) * hours,
        "1h": hours,
        "4h": int(hours / 4),
    }
    return mapping.get(timeframe, int(60 / 15) * hours)  # default 15m


def _compute_common_features(df: pd.DataFrame, ind: pd.DataFrame) -> Dict[str, Any]:
    # Ensure basic columns exist
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    # RSI from indicators if present
    rsi = float(ind["rsi_14"].iloc[-1]) if "rsi_14" in ind else None

    # EMA alignment (bull/bear/mixed)
    ema20 = ind["ema_20"].iloc[-1] if "ema_20" in ind else None
    ema50 = ind["ema_50"].iloc[-1] if "ema_50" in ind else None
    ema200 = ind["ema_200"].iloc[-1] if "ema_200" in ind else None

    # If EMAs not provided, compute quick EMAs on the fly
    if ema20 is None or ema50 is None or ema200 is None:
        _ema20 = _ema(df["close"], 20).iloc[-1]
        _ema50 = _ema(df["close"], 50).iloc[-1]
        _ema200 = _ema(df["close"], 200).iloc[-1] if df.shape[0] >= 200 else _ema(df["close"], min(200, df.shape[0])).iloc[-1]
        ema20 = float(ema20) if ema20 is not None else float(_ema20)
        ema50 = float(ema50) if ema50 is not None else float(_ema50)
        ema200 = float(ema200) if ema200 is not None else float(_ema200)
    else:
        ema20, ema50, ema200 = float(ema20), float(ema50), float(ema200)

    if c >= ema20 >= ema50 >= ema200:
        ema_alignment = "bull"
    elif c <= ema20 <= ema50 <= ema200:
        ema_alignment = "bear"
    else:
        ema_alignment = "mixed"

    # Range distances if present
    dist_low = float(ind["dist_to_range_low"].iloc[-1]) if "dist_to_range_low" in ind else None
    dist_high = float(ind["dist_to_range_high"].iloc[-1]) if "dist_to_range_high" in ind else None

    # Candle anatomy
    candle = _body_wicks(o, h, l, c)

    return {
        "last": float(c),
        "rsi": rsi,
        "ema20": round(float(ema20), 8),
        "ema50": round(float(ema50), 8),
        "ema200": round(float(ema200), 8),
        "ema_alignment": ema_alignment,
        "dist_to_low_pct": dist_low,
        "dist_to_high_pct": dist_high,
        "last_candle": candle,
        "last_candle_ts": _ms_to_iso(df["timestamp"].iloc[-1]) if "timestamp" in df else None,
    }


def _atr_pct(df: pd.DataFrame, n: int = 15) -> Optional[float]:
    # lightweight ATR% using last n bars (n defaults ~15 days on daily, here we apply on TF)
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.DataFrame({
            "h-l": high - low,
            "h-pc": (high - prev_close).abs(),
            "l-pc": (low - prev_close).abs(),
        }).max(axis=1)
        atr = tr.rolling(n, min_periods=max(2, n // 3)).mean().iloc[-1]
        pct = float(atr / close.iloc[-1] * 100.0)
        return round(pct, 3)
    except Exception:
        return None


# --------------- Playbook evaluation ---------------
def _eval_breakout(features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any], List[str]]:
    """
    Breakout Momentum playbook (v4 baseline from chat):
      - within 1.5% of resistance (dist_to_high_pct <= 1.5)
      - RSI caps: 15m < 70, 1h < 70, 4h < 65 (we only have current TF RSI here; upstream should pass multi-TF if available)
      - trigger candle: green body >= 30% (approx with last_candle)
      - volume anomaly: vol_z >= 0.5 (fallback) / >= 1.0 (primary)
    """
    th = {"near_resistance_pct": 1.5, "max_rsi_15m": 70.0, "min_vol_z": 0.5, "primary_vol_z": 1.0, "min_body_pct": 30.0}
    reasons: List[str] = []
    status = "no"
    score = 0.0

    dist_high = features.get("dist_to_high_pct")
    rsi_15m = features.get("rsi_15m")
    vol_z = features.get("vol_z_24h")
    candle = (features.get("last_candle") or {})
    body_pct = candle.get("body_pct")
    is_green = (candle.get("dir") == "up")

    if dist_high is not None and dist_high <= th["near_resistance_pct"]:
        reasons.append(f"Near resistance ({dist_high:.2f}%)")
        score += 0.25

    if rsi_15m is not None and rsi_15m < th["max_rsi_15m"]:
        reasons.append(f"RSI15m {rsi_15m:.1f} < {th['max_rsi_15m']}")
        score += 0.20
    elif rsi_15m is not None:
        reasons.append(f"RSI15m {rsi_15m:.1f} ≥ {th['max_rsi_15m']} (hot)")

    if vol_z is not None:
        if vol_z >= th["primary_vol_z"]:
            reasons.append(f"Strong volume (z={vol_z:.2f})")
            score += 0.40
        elif vol_z >= th["min_vol_z"]:
            reasons.append(f"Moderate volume (z={vol_z:.2f})")
            score += 0.25
        else:
            reasons.append(f"Low volume (z={vol_z:.2f})")
    else:
        reasons.append("Missing volume z-score")

    if is_green and body_pct is not None and body_pct >= th["min_body_pct"]:
        reasons.append(f"Trigger body {body_pct:.0f}%")
        score += 0.25

    # status mapping
    if score >= 0.85:
        status = "confirm"
    elif score >= 0.50:
        status = "watch"
    else:
        status = "no"

    return status, round(score, 3), th, reasons


def _eval_mean_reversion(features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any], List[str]]:
    """
    Mean Reversion playbook (v4 baseline from chat):
      - within 2.5% of support (dist_to_low_pct <= 2.5)
      - RSI window: 15m in [35, 55], 1h/4h in mid-band (not enforced here unless supplied upstream)
      - trigger: green body >= 30%
      - vol_z >= 0.3 (fallback) / >= 0.8 (primary)
    """
    th = {
        "near_support_pct": 2.5, "rsi15m_min": 35.0, "rsi15m_max": 55.0,
        "min_vol_z": 0.3, "primary_vol_z": 0.8, "min_body_pct": 30.0
    }
    reasons: List[str] = []
    status = "no"
    score = 0.0

    dist_low = features.get("dist_to_low_pct")
    rsi_15m = features.get("rsi_15m")
    vol_z = features.get("vol_z_24h")
    candle = (features.get("last_candle") or {})
    body_pct = candle.get("body_pct")
    is_green = (candle.get("dir") == "up")

    if dist_low is not None and dist_low <= th["near_support_pct"]:
        reasons.append(f"Near support ({dist_low:.2f}%)")
        score += 0.30

    if rsi_15m is not None and th["rsi15m_min"] <= rsi_15m <= th["rsi15m_max"]:
        reasons.append(f"RSI15m {rsi_15m:.1f} in [{th['rsi15m_min']:.0f},{th['rsi15m_max']:.0f}]")
        score += 0.25
    elif rsi_15m is not None:
        reasons.append(f"RSI15m {rsi_15m:.1f} out of window")

    if vol_z is not None:
        if vol_z >= th["primary_vol_z"]:
            reasons.append(f"Strong volume (z={vol_z:.2f})")
            score += 0.30
        elif vol_z >= th["min_vol_z"]:
            reasons.append(f"Moderate volume (z={vol_z:.2f})")
            score += 0.18
        else:
            reasons.append(f"Low volume (z={vol_z:.2f})")
    else:
        reasons.append("Missing volume z-score")

    if is_green and body_pct is not None and body_pct >= th["min_body_pct"]:
        reasons.append(f"Trigger body {body_pct:.0f}%")
        score += 0.20

    if score >= 0.80:
        status = "confirm"
    elif score >= 0.45:
        status = "watch"
    else:
        status = "no"

    return status, round(score, 3), th, reasons


# --------------- Public API ---------------
def build_snapshot_v41(
    exchange,
    symbols: Iterable[str],
    *,
    timeframe: str = "15m",
    limit: int = 240,
    quote_asset: str = "USDT",
    exchange_name: str = "bitget",
) -> Dict[str, Any]:
    """
    Returns a self-contained LLM-friendly snapshot document (v4.1).
    """
    bars_4h = _bars_for(4, timeframe)
    bars_24h = _bars_for(24, timeframe)

    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "exchange": exchange_name,
        "quote_asset": quote_asset,
        "market_type": "spot",
        "timeframe": timeframe,
        "items": [],
        "units": {
            "dist_to_high_pct": "percent",
            "dist_to_low_pct": "percent",
            "rsi_*": "index(0-100)",
            "vol_z_24h": "zscore",
            "atr_pct_15d": "percent",
            "spread_pct": "percent",
            "body_pct": "percent",
            "upper_wick_pct": "percent",
            "lower_wick_pct": "percent",
        },
    }

    for sym in symbols:
        # Fetch OHLCV
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        except Exception:
            ohlcv = None

        missing: List[str] = []
        if not ohlcv or len(ohlcv) < max(60, bars_24h + 5):
            out["items"].append({
                "symbol": sym,
                "error": "insufficient_data",
                "nl_summary": f"{sym}: not enough candles to compute reliable features.",
                "data_quality": {"missing_fields": ["ohlcv"], "latency_sec": None, "candle_count": {timeframe: len(ohlcv) if ohlcv else 0}},
                "confidence": 0.0,
                "safe_to_trade": False
            })
            continue

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Indicators
        try:
            ind = compute_indicators(df)
            if not isinstance(ind, pd.DataFrame):
                raise ValueError("compute_indicators must return DataFrame")
        except Exception:
            ind = pd.DataFrame(index=df.index)
            missing.append("indicators")

        # Common features
        feats = _compute_common_features(df, ind if not ind.empty else pd.DataFrame(index=df.index))

        # RSI 15m naming for LLM
        feats["rsi_15m"] = feats.pop("rsi", None)

        # 4h momentum (approx on current TF)
        feats["pct4h"] = round(_pct_change_over_n(df, bars_4h), 3)

        # 24h volume anomaly (sum z-score)
        sums_24h = df["volume"].rolling(window=bars_24h, min_periods=bars_24h).sum()
        feats["vol_z_24h"] = round(_rolling_z_last(sums_24h.dropna(), window=20, ddof=1), 3) if not sums_24h.dropna().empty else None

        # ATR% ~15 "days" equivalent on this TF (heuristic)
        feats["atr_pct_15d"] = _atr_pct(df, n=15)

        # Spread (from snapshot-time ticker)
        tick = ticker_bitget(sym) or {}
        last_tick = tick.get("last")
        bid, ask = tick.get("bid"), tick.get("ask")
        if bid and ask and bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2.0) * 100.0
            feats["spread_pct"] = round(float(spread_pct), 4)
        else:
            feats["spread_pct"] = None

        # Add tick info (snapshot-time prices)
        tick_struct = {
            "last": last_tick if isinstance(last_tick, (int, float)) else None,
            "bid": bid if isinstance(bid, (int, float)) else None,
            "ask": ask if isinstance(ask, (int, float)) else None,
            "ts": tick.get("ts")
        }

        # Data quality / latency
        last_bar_ts = df["timestamp"].iloc[-1] if "timestamp" in df else None
        latency_sec = None
        if last_bar_ts is not None:
            try:
                latency_sec = max(0, int((datetime.now(timezone.utc).timestamp() - (last_bar_ts / 1000.0))))
            except Exception:
                latency_sec = None

        # Playbook evaluations
        play_feats = {
            "dist_to_high_pct": feats.get("dist_to_high_pct"),
            "dist_to_low_pct": feats.get("dist_to_low_pct"),
            "rsi_15m": feats.get("rsi_15m"),
            "vol_z_24h": feats.get("vol_z_24h"),
            "last_candle": feats.get("last_candle"),
        }
        bk_status, bk_score, bk_th, bk_reasons = _eval_breakout(play_feats)
        mr_status, mr_score, mr_th, mr_reasons = _eval_mean_reversion(play_feats)

        # Final recommendation
        if bk_status == "confirm" and mr_status != "confirm":
            final_rec = "enter_breakout"
        elif mr_status == "confirm" and bk_status != "confirm":
            final_rec = "enter_mean_reversion"
        elif bk_status == "watch" or mr_status == "watch":
            final_rec = "alert_only"
        else:
            final_rec = "no_trade"

        # Confidence: blend data completeness + best score
        completeness = 1.0
        # penalize if key fields missing
        for k in ["dist_to_low_pct", "dist_to_high_pct", "rsi_15m", "vol_z_24h"]:
            if play_feats.get(k) is None or (isinstance(play_feats.get(k), float) and math.isnan(play_feats.get(k))):
                completeness -= 0.15
        completeness = max(0.0, min(1.0, completeness))

        best_score = max(bk_score, mr_score)
        confidence = round(float(0.5 * completeness + 0.5 * min(1.0, best_score)), 3)

        safe_to_trade = bool(confidence >= 0.55 and latency_sec is not None and latency_sec < 3600)

        # Natural-language summary
        nl_bits: List[str] = []
        if bk_status == "confirm":
            nl_bits.append("confirmed breakout conditions")
        elif bk_status == "watch":
            nl_bits.append("near breakout")
        if mr_status == "confirm":
            nl_bits.append("confirmed mean reversion at support")
        elif mr_status == "watch":
            nl_bits.append("near support bounce")
        if not nl_bits:
            nl_bits.append("no tradable edge right now")
        nl_summary = f"{sym}: " + ", ".join(nl_bits) + "."

        # Assemble
        item = {
            "symbol": sym,
            "features": {
                "last": feats.get("last"),
                "rsi_15m": feats.get("rsi_15m"),
                "dist_to_high_pct": feats.get("dist_to_high_pct"),
                "dist_to_low_pct": feats.get("dist_to_low_pct"),
                "vol_z_24h": feats.get("vol_z_24h"),
                "ema_alignment": feats.get("ema_alignment"),
                "ema20": feats.get("ema20"),
                "ema50": feats.get("ema50"),
                "ema200": feats.get("ema200"),
                "atr_pct_15d": feats.get("atr_pct_15d"),
                "last_candle": feats.get("last_candle"),
                "spread_pct": feats.get("spread_pct"),
                "last_candle_ts": feats.get("last_candle_ts"),
            },
            "tick": tick_struct,  # snapshot-time prices
            "playbook_eval": {
                "breakout": {
                    "status": bk_status,
                    "score": bk_score,
                    "thresholds": bk_th,
                    "reasons": bk_reasons,
                },
                "mean_reversion": {
                    "status": mr_status,
                    "score": mr_score,
                    "thresholds": mr_th,
                    "reasons": mr_reasons,
                },
                "final_recommendation": final_rec,
            },
            "nl_summary": nl_summary,
            "data_quality": {
                "missing_fields": missing,
                "latency_sec": latency_sec,
                "candle_count": { "timeframe": timeframe, "count": int(df.shape[0]) },
            },
            "confidence": confidence,
            "safe_to_trade": safe_to_trade,
        }

        out["items"].append(item)

    return out


__all__ = ["build_snapshot_v41"]
