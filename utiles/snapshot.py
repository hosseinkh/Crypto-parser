# utiles/snapshot.py
# --------------------------------------------------------------------
# Snapshot builder:
# - CCXT-based OHLCV fetch with retries
# - Multi-timeframe indicator blocks (15m, 1h, 4h)
# - Includes vol_z, ATR, RSI, MACD hist, BBs, EMAs, dist_to_range_*
# - Optional sentiment hook per symbol
# - Returns a clean JSON-ready dict
# --------------------------------------------------------------------

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as e:
    raise RuntimeError("ccxt is required. Install with: pip install ccxt") from e

from .indicators import compute_indicators, summarize_last_row


# ------------------------ Exchange helpers ------------------------ #

def make_exchange(ex_name: str, enable_rate_limit: bool = True) -> Any:
    """
    Build a CCXT exchange instance with common sane defaults.
    """
    ex_name = (ex_name or "bitget").lower()
    if not hasattr(ccxt, ex_name):
        raise ValueError(f"Unsupported exchange '{ex_name}' for CCXT.")
    ex_cls = getattr(ccxt, ex_name)
    ex = ex_cls({
        "enableRateLimit": enable_rate_limit,
        "timeout": 30_000,
        # if you use auth endpoints later, set your keys here or via ccxt.env
        # "apiKey": "...",
        # "secret": "...",
        # "password": "...",
    })
    return ex


def safe_fetch_ohlcv(
    ex: Any,
    symbol: str,
    timeframe: str = "15m",
    limit: int = 240,
    since: Optional[int] = None,
    max_retries: int = 3,
    retry_wait: float = 0.75,
) -> List[List[float]]:
    """
    Robust OHLCV fetch with light retry/backoff.
    Returns list of [ts, open, high, low, close, volume].
    """
    last_exc: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except Exception as e:
            last_exc = e
            time.sleep(retry_wait)
    raise RuntimeError(f"fetch_ohlcv failed for {symbol} {timeframe}: {last_exc}")


def last_price_snapshot(ex: Any, symbol: str, fallback_close: float) -> float:
    """
    Try to read a fresh 'last' price from ticker; if unavailable, use fallback_close.
    """
    try:
        t = ex.fetch_ticker(symbol)
        val = t.get("last") or t.get("close")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return float(fallback_close)


# ------------------------ Timeframe block ------------------------ #

def build_tf_block_ccxt(
    ex: Any,
    symbol: str,
    tf: str,
    limit: int = 240,
) -> Dict[str, Any]:
    """
    Fetch candles for `tf`, compute indicators, and return a compact TF block.
    """
    ohlcv = safe_fetch_ohlcv(ex, symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty or df.shape[0] < 50:
        return {"tf": tf, "error": "insufficient_data", "limit": df.shape[0] if not df.empty else 0}

    ind = compute_indicators(df, fill_partial=True)
    last = summarize_last_row(ind)

    block = {
        "tf": tf,
        "last_candle": {
            "t": int(df["timestamp"].iloc[-1]),
            "o": float(df["open"].iloc[-1]),
            "h": float(df["high"].iloc[-1]),
            "l": float(df["low"].iloc[-1]),
            "c": float(df["close"].iloc[-1]),
            "v": float(df["volume"].iloc[-1]),
        },
        "indicators": last,  # includes rsi14, atr14, macd_hist, bb_upper/lower, ema20/50/200, vol_z, distances
    }
    return block


# ------------------------ Sentiment hook ------------------------ #

def default_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Placeholder sentiment provider. Replace with your real function.
    Expected return: {"score": float in [-1,1], "label": "negative|neutral|positive"}
    """
    return {"score": 0.0, "label": "neutral"}


# ------------------------ Snapshot builder ------------------------ #

def build_snapshot(
    ex_name: str = "bitget",
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    limit: int = 240,
    sentiment_func: Callable[[str], Dict[str, Any]] = default_sentiment,
) -> Dict[str, Any]:
    """
    Build a full multi-timeframe snapshot for a set of symbols.

    Output format:
    {
      "meta": {...},
      "rows": [
        {
          "symbol": "BTC/USDT",
          "last": 115000.0,
          "timeframes": {
            "15m": {...},
            "1h":  {...},
            "4h":  {...}
          },
          "sentiment": {"score": 0.13, "label": "positive"}
        },
        ...
      ],
      "errors": [...]
    }
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", "XRP/USDT", "BCH/USDT", "BNB/USDT"]
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]

    ex = make_exchange(ex_name)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            # Use base TF to fetch a quick last price fallback
            base_tf = "15m" if "15m" in timeframes else timeframes[0]
            base_ohlcv = safe_fetch_ohlcv(ex, sym, timeframe=base_tf, limit=max(50, min(limit, 240)))
            if not base_ohlcv:
                errors.append({"symbol": sym, "error": "no_ohlcv"})
                continue
            base_df = pd.DataFrame(base_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            last_close = float(base_df["close"].iloc[-1])
            last_px = last_price_snapshot(ex, sym, last_close)

            tf_blocks: Dict[str, Any] = {}
            for tf in timeframes:
                tf_blocks[tf] = build_tf_block_ccxt(ex, sym, tf=tf, limit=limit)

            sent = sentiment_func(sym)

            rows.append({
                "symbol": sym,
                "last": float(last_px),
                "timeframes": tf_blocks,
                "sentiment": {
                    "score": float(sent.get("score", 0.0)),
                    "label": str(sent.get("label", "neutral")),
                },
            })

        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})

    snap = {
        "meta": {
            "exchange": ex_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbols_count": len(rows),
            "timeframes": timeframes,
            "limit_per_tf": limit,
        },
        "rows": rows,
        "errors": errors,
    }
    return snap
