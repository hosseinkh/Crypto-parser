# utiles/snapshot.py

from __future__ import annotations
import numpy as np
import pandas as pd

# fetch directly from utiles.bitget (do not rely on `ex`)
from .bitget import fetch_ohlcv_df

# indicators from your local module
from .indicators import (
    rsi, ema, atr, volume_zscore,
)

# ---------- helpers ----------

def _to_iso_utc(x) -> str:
    try:
        ts = pd.to_datetime(x, utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        try:
            xi = int(x)
            unit = "ms" if xi > 10_000_000_000 else "s"
            ts = pd.to_datetime(xi, unit=unit, utc=True)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(x)

def _ensure_cols(df: pd.DataFrame):
    need = {"time", "open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {missing}")

def df_to_last_candles(df: pd.DataFrame, max_rows: int = 60) -> list[dict]:
    _ensure_cols(df)
    rows = df.tail(max_rows).reset_index(drop=True)
    return [
        {
            "t": _to_iso_utc(rows.loc[i, "time"]),
            "o": float(rows.loc[i, "open"]),
            "h": float(rows.loc[i, "high"]),
            "l": float(rows.loc[i, "low"]),
            "c": float(rows.loc[i, "close"]),
            "v": float(rows.loc[i, "volume"]),
        }
        for i in range(len(rows))
    ]

def summarize_frame(df: pd.DataFrame) -> dict:
    _ensure_cols(df)
    close = df["close"]
    rsi14 = rsi(close, 14).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    atr14 = atr(df, 14).iloc[-1]
    vz20 = volume_zscore(df["volume"], 20).iloc[-1]

    if ema21 > ema50:
        trend = "up"
    elif ema21 < ema50:
        trend = "down"
    else:
        trend = "sideways"

    def f(x): return float(x) if pd.notna(x) else None
    return {
        "rsi": f(rsi14),
        "ema_fast": f(ema21),
        "ema_slow": f(ema50),
        "atr": f(atr14),
        "vol_z": f(vz20),
        "trend": trend,
    }

# ---------- public API ----------

def build_tf_block(
    ex_unused,                 # kept for signature compatibility; not used
    symbol: str,
    timeframe: str,
    lc_count: int = 50,
    **kwargs,
) -> dict:
    """
    Return a per-timeframe block used by the app. Fetches candles directly from
    utiles.bitget.fetch_ohlcv_df to avoid relying on an `ex` object.
    """
    df = fetch_ohlcv_df(symbol=symbol, timeframe=timeframe, limit=int(lc_count))
    _ensure_cols(df)

    summary = summarize_frame(df)
    candles = df_to_last_candles(df, max_rows=min(int(lc_count), 120))

    return {
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
        "structure": {
            "trend": summary["trend"],
            "local_sr": None,
            "breakout": None,
            "divergence": None,
        },
        "notes": [],
    }
