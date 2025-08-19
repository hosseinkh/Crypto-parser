# utiles/snapshot.py
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import timezone

# ✅ import the module instead of individual symbols
from . import indicators as ind

def df_to_last_candles(
    df: pd.DataFrame,
    max_rows: int = 20,
) -> list[dict]:
    """
    Convert the last N rows of a standard OHLCV DataFrame into
    a compact list of candle dicts.
    """
    rows = df.tail(max_rows)
    out = []
    for _, r in rows.iterrows():
        out.append({
            "t": pd.to_datetime(r["timestamp"], utc=True)
                    .tz_convert(timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": float(r["volume"]),
        })
    return out

def summarize_frame(df: pd.DataFrame, symbol: str, tf: str) -> dict:
    """
    Build one timeframe block of the JSON snapshot.
    Assumes df has columns: timestamp, open, high, low, close, volume
    """
    # compute indicators on a copy
    feat = ind.compute_indicators(df)

    last = feat.iloc[-1]
    prev = feat.iloc[-2] if len(feat) > 1 else last

    trend = "up" if last["ema20"] > last["ema50"] else ("down" if last["ema20"] < last["ema50"] else "flat")

    frame = {
        "timeframe": tf,
        "last_candles": df_to_last_candles(feat, max_rows=20),
        "price": {
            "last": float(last["close"]),
            "change_1": float((last["close"] / prev["close"] - 1) * 100) if len(feat) > 1 else 0.0,
        },
        "indicators": {
            "ema": {"ema20": float(last["ema20"]), "ema50": float(last["ema50"])},
            "rsi14": float(last["rsi14"]),
            "atr14": float(last["atr14"]),
            "macd": {"line": float(last["macd"]), "signal": float(last["macd_signal"]), "hist": float(last["macd_hist"])},
            "bbands": {"upper": float(last["bb_upper"]), "mid": float(last["bb_mid"]), "lower": float(last["bb_lower"])},
            "volume_z20": float(last["vol_z20"]),
        },
        "structure": {
            "trend": trend,
            "dist_to_range_high": float(last["dist_to_high"]),
            "dist_to_range_low": float(last["dist_to_low"]),
        },
    }
    return frame
