# utiles/snapshot.py

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from datetime import timezone

# Use the indicators module that already exists in utiles/
# (streamlit_app also imports it as `ind`)
from .indicators import (
    rsi,
    ema,
    atr,
    volume_zscore,
    macd,
    bbands,
    rolling_extrema,
    distance_to_levels,
)

# --------------------------
# Helpers
# --------------------------

def _to_iso_utc(x) -> str:
    """
    Convert many possible timestamp formats to ISO UTC (yyyy-mm-ddTHH:MM:SSZ).
    Accepts pandas Timestamps, epoch ms / s, strings, etc.
    """
    try:
        # pandas Timestamp or parseable string
        ts = pd.to_datetime(x, utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        try:
            # numeric epoch (try ms, then s)
            xi = int(x)
            if xi > 10_000_000_000:  # likely ms
                ts = pd.to_datetime(xi, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(xi, unit="s", utc=True)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            # last resort
            return str(x)

def _ensure_ohlcv_columns(df: pd.DataFrame) -> None:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {missing}")

def df_to_last_candles(df: pd.DataFrame, max_rows: int = 60) -> list[dict]:
    """
    Convert the tail of an OHLCV dataframe to the compact list-of-dicts format:
    [{t,o,h,l,c,v}, ...]
    """
    _ensure_ohlcv_columns(df)
    rows = df.tail(max_rows).copy()

    # Make sure 'time' is a column, not index
    if "time" not in rows.columns and rows.index.name in (None, "time"):
        rows = rows.reset_index()

    out = []
    for _, r in rows.iterrows():
        out.append({
            "t": _to_iso_utc(r["time"]),
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": float(r["volume"]),
        })
    return out

def summarize_frame(df: pd.DataFrame) -> dict:
    """
    Compute a lightweight summary set of indicators and a basic trend label.
    Keeps it cheap so it runs fine on Streamlit Cloud.
    """
    _ensure_ohlcv_columns(df)
    close = df["close"]

    # Core indicators
    rsi_14 = rsi(close, length=14).iloc[-1]
    ema_fast = ema(close, length=21).iloc[-1]
    ema_slow = ema(close, length=50).iloc[-1]
    atr_14 = atr(df, length=14).iloc[-1]
    vol_z = volume_zscore(df["volume"], length=20).iloc[-1]

    # Very simple trend classification
    if ema_fast > ema_slow:
        trend = "up"
    elif ema_fast < ema_slow:
        trend = "down"
    else:
        trend = "sideways"

    return {
        "rsi": float(rsi_14) if pd.notna(rsi_14) else None,
        "ema_fast": float(ema_fast) if pd.notna(ema_fast) else None,
        "ema_slow": float(ema_slow) if pd.notna(ema_slow) else None,
        "atr": float(atr_14) if pd.notna(atr_14) else None,
        "vol_z": float(vol_z) if pd.notna(vol_z) else None,
        "trend": trend,
    }

# --------------------------
# Public API used by the app
# --------------------------

def build_tf_block(
    ex,
    symbol: str,
    timeframe: str,
    lc_count: int = 50,
    now_iso: str | None = None,
    **kwargs,
) -> dict:
    """
    Fetch OHLCV for (symbol, timeframe) using `lc_count` candles and build a block:
    {
      "tf": "...",
      "n_candles": N,
      "last_candles": [{t,o,h,l,c,v}, ...],
      "indicators": {...},
      "structure": {...},
      "notes": []
    }

    Extra **kwargs are accepted to keep compatibility with any future UI params.
    """
    # 1) Fetch data from the exchange helper (utiles/bitget.py)
    #    Expected columns: time, open, high, low, close, volume
    df = ex.fetch_ohlcv_df(symbol=symbol, timeframe=timeframe, limit=int(lc_count))
    _ensure_ohlcv_columns(df)

    # 2) Indicators + compact candles
    summary = summarize_frame(df)
    candles = df_to_last_candles(df, max_rows=min(int(lc_count), 120))

    # 3) Minimal structure (placeholders for future enrichment)
    structure = {
        "trend": summary["trend"],
        "local_sr": None,
        "breakout": None,
        "divergence": None,
    }

    # 4) Compose block
    block = {
        "tf": timeframe,
        "n_candles": int(len(df)),
        "last_candles": candles,
        "indicators": {
            "rsi": summary["rsi"],
            "ema_fast": summary["ema_fast"],
            "ema_slow": summary["ema_slow"],
            "atr": summary["atr"],
            "vol_z": summary["vol_z"],
        },
        "structure": structure,
        "notes": [],
    }
    return block
