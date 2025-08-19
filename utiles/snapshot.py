# utiles/snapshot.py
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timezone
from .indicators import rsi, ema, atr, volume_zscore
from .bitget import fetch_ohlcv_df, now_utc_iso

def df_to_last_candles(df: pd.DataFrame, max_rows: int) -> list[dict]:
    rows = df.tail(max_rows)
    out = []
    for _, r in rows.iterrows():
        out.append({
            "t": r["ts"].to_pydatetime().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": float(r["volume"]),
        })
    return out

def build_tf_block(ex, symbol: str, tf: str, lc_count: int) -> tuple[dict, pd.DataFrame]:
    # FETCH — NOTE: positional call, no keyword 'limit'
    df = fetch_ohlcv_df(ex, symbol, tf, int(lc_count))

    # INDICATORS
    closes = df["close"]
    vols   = df["volume"]
    df["ema_20"] = ema(closes, 20)
    df["ema_50"] = ema(closes, 50)
    df["rsi_14"] = rsi(closes, 14)
    df["atr_14"] = atr(df["high"], df["low"], closes, 14)
    df["vol_z20"] = volume_zscore(vols, 20)

    # STRUCTURE (simple)
    dist_to_high = float((closes.iloc[-1] - df["high"].rolling(50).max().iloc[-1]) / closes.iloc[-1])
    dist_to_low  = float((closes.iloc[-1] - df["low"].rolling(50).min().iloc[-1])  / closes.iloc[-1])
    trend = "up" if df["ema_20"].iloc[-1] > df["ema_50"].iloc[-1] else ("down" if df["ema_20"].iloc[-1] < df["ema_50"].iloc[-1] else "flat")

    # JSON block
    block = {
        "timeframe": tf,
        "candle_count": int(lc_count),
        "last_candles": df_to_last_candles(df, max_rows=int(lc_count)),
        "indicators": {
            "rsi_14": float(df["rsi_14"].iloc[-1]),
            "ema_20": float(df["ema_20"].iloc[-1]),
            "ema_50": float(df["ema_50"].iloc[-1]),
            "atr_14": float(df["atr_14"].iloc[-1]),
            "vol_z20": float(df["vol_z20"].iloc[-1]),
        },
        "structure": {
            "trend": trend,
            "dist_to_high": dist_to_high,
            "dist_to_low":  dist_to_low,
        },
        "build_ts": now_utc_iso(),
    }
    return block, df
