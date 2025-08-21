# utiles/snapshot.py
from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime, timezone

import pandas as pd

from utiles.bitget import fetch_ohlcv_df
from utiles.indicators import compute_indicators
from utiles.sentiment import get_fear_greed, per_crypto_sentiment


# --------------------------
# Small utilities
# --------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _symbol_base(symbol: str) -> str:
    s = symbol.upper().strip()
    return s.split("/")[0] if "/" in s else s


# Cache per-crypto sentiment during one scan to avoid repeating API calls
_SENTIMENT_CACHE: Dict[str, Dict[str, Any]] = {}


# --------------------------
# Candles -> compact format
# --------------------------

def df_to_last_candles(df: pd.DataFrame, max_rows: int = 30) -> list[dict]:
    """
    Convert the tail of an OHLCV dataframe to compact list-of-dicts for JSON.
    Assumes df has columns: ts, open, high, low, close, volume
    """
    rows = df.tail(max_rows)
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "t": r["ts"].isoformat().replace("+00:00", "Z"),
                "o": float(r["open"]),
                "h": float(r["high"]),
                "l": float(r["low"]),
                "c": float(r["close"]),
                "v": float(r["volume"]),
            }
        )
    return out


# --------------------------
# Per-timeframe block builder
# --------------------------

def build_tf_block(
    ex,
    symbol: str,
    tf: str,
    limit: Optional[int] = None,
    lc_count: Optional[int] = None,  # tolerate older caller signature
) -> Dict[str, Any]:
    """
    Build a per-timeframe block:
      {
        "tf": "15m",
        "last_candles": [...],
        "indicators": {...},
        "structure": {...},
        "sentiment": {...}   # <-- added
      }
    """
    use_limit = int(limit or lc_count or 50)

    # 1) Fetch + indicators
    df = fetch_ohlcv_df(ex, symbol, tf, use_limit)
    df = compute_indicators(df)

    # 2) Simple structure summary (unchanged)
    close = df["close"].iloc[-1]
    ema20 = df["ema_20"].iloc[-1]
    ema50 = df["ema_50"].iloc[-1]
    ema200 = df["ema_200"].iloc[-1]

    trend = (
        "up"
        if (ema20 > ema50 and ema50 > ema200)
        else "down"
        if (ema20 < ema50 and ema50 < ema200)
        else "mixed"
    )

    # 3) Per-crypto sentiment (cached per base symbol for this scan)
    base = _symbol_base(symbol)
    if base not in _SENTIMENT_CACHE:
        # This calls CryptoPanic+VADER if key is configured, otherwise returns nulls
        _SENTIMENT_CACHE[base] = per_crypto_sentiment(symbol)

    per_sym_sent = _SENTIMENT_CACHE.get(base, {}) or {}
    sentiment_block: Dict[str, Any] = {
        # normalized 0..1 where available; otherwise null
        "twitter_score": per_sym_sent.get("twitter_score"),
        "news_score": per_sym_sent.get("news_score"),
        "community_score": per_sym_sent.get("community_score"),
        "overall": per_sym_sent.get("overall"),
        "source": per_sym_sent.get("source", "none"),
        "as_of_utc": _now_utc_iso(),
        "explain": (
            "Per-crypto sentiment is computed from recent news headlines (CryptoPanic) scored with VADER; "
            "if no API key configured, fields remain null."
        ),
    }

    # 4) Assemble block (existing fields + new 'sentiment')
    block = {
        "tf": tf,
        "last_candles": df_to_last_candles(df, max_rows=min(use_limit, 120)),
        "indicators": {
            "close": float(close),
            "ema20": float(ema20),
            "ema50": float(ema50),
            "ema200": float(ema200),
            "rsi14": float(df["rsi_14"].iloc[-1]),
            "macd": float(df["macd"].iloc[-1]),
            "macd_signal": float(df["macd_signal"].iloc[-1]),
            "macd_hist": float(df["macd_hist"].iloc[-1]),
            "bb_upper": float(df["bb_upper"].iloc[-1]),
            "bb_basis": float(df["bb_basis"].iloc[-1]),
            "bb_lower": float(df["bb_lower"].iloc[-1]),
            "atr14": float(df["atr_14"].iloc[-1]),
            "vol_z": float(df["vol_z"].iloc[-1]),
            # keep your normalized distances
            "dist_to_high": float(df["dist_to_range_high"].iloc[-1]),
            "dist_to_low": float(df["dist_to_range_low"].iloc[-1]),
        },
        "structure": {
            "trend": trend,
        },
        "sentiment": sentiment_block,  # <-- new
    }
    return block


# --------------------------
# Global sentiment helper
# --------------------------

def build_global_sentiment() -> Dict[str, Any]:
    """
    Build a global sentiment block using the free Fear & Greed Index.
    Safe to call even without network (returns nulls).
    """
    fng = get_fear_greed()  # {"value": int|None, "label": str|None, "source": "alternative.me"}
    value = fng.get("value")
    label = (fng.get("label") or "").lower() if isinstance(fng.get("label"), str) else None

    # Optional: map label to a coarse 'overall'
    if label in {"extreme fear", "fear"}:
        overall = "bearish"
    elif label in {"greed", "extreme greed"}:
        overall = "bullish"
    elif label:
        overall = "neutral"
    else:
        overall = None

    return {
        "fear_greed_index": value,   # 0..100 or null
        "label": fng.get("label"),
        "overall": overall,          # bearish / neutral / bullish / null
        "source": fng.get("source", "alternative.me"),
        "as_of_utc": _now_utc_iso(),
        "explain": (
            "Global market sentiment from the Crypto Fear & Greed Index (alternative.me). "
            "Values closer to 0 indicate fear; closer to 100 indicate greed."
        ),
    }


# --------------------------
# Meta guide helper
# --------------------------

def meta_guide() -> Dict[str, Any]:
    """
    Human-readable guide describing how to interpret the JSON snapshot.
    Embed this under the 'meta' key in your final packed JSON.
    """
    return {
        "source": "bitget (candles) + indicators + per-crypto news sentiment (CryptoPanic+VADER) + global Fear & Greed",
        "note": "LLM-friendly JSON snapshot for multi-timeframe screening.",
        "guide": {
            "symbols": "Each entry is one trading pair (e.g., BTC/USDT).",
            "timeframes": "Inside each symbol, data is grouped by timeframe (15m, 1h, 4h).",
            "last_candles": "Recent OHLCV candles: t=open time (UTC), o/h/l/c prices, v=volume.",
            "indicators": (
                "Precomputed metrics such as EMA20/50/200, RSI14, MACD (line/signal/hist), "
                "Bollinger Bands (upper/basis/lower), ATR14, volume z-score (vol_z), and normalized "
                "distances to recent range extremes (dist_to_high/dist_to_low)."
            ),
            "structure": "Trend classification derived from EMA stacking: up/down/mixed.",
            "sentiment_per_crypto": (
                "Per-crypto sentiment is attached to each timeframe block. "
                "Currently based on recent news headlines (CryptoPanic) scored with VADER; "
                "twitter_score and community_score are placeholders."
            ),
            "sentiment_global": (
                "Global sentiment from the Crypto Fear & Greed Index; label and overall map to bearish/neutral/bullish."
            ),
        },
    }
