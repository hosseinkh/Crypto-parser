# utiles/snapshot.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

from utiles.bitget import fetch_ohlcv_df
from utiles.indicators import compute_indicators


def df_to_last_candles(df: pd.DataFrame, max_rows: int = 30) -> list[dict]:
    """
    Convert the tail of an OHLCV dataframe to a compact list of dicts
    for JSON output, keeping only the essentials.
    """
    rows = df.tail(max_rows)
    out: list[dict] = []
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
        "structure": {...}
      }
    """
    use_limit = int(limit or lc_count or 50)

    # fetch + indicators
    df = fetch_ohlcv_df(ex, symbol, tf, use_limit)
    df = compute_indicators(df)

    # simple structure summary
    close = float(df["close"].iloc[-1])
    ema20 = float(df["ema_20"].iloc[-1])
    ema50 = float(df["ema_50"].iloc[-1])
    ema200 = float(df["ema_200"].iloc[-1])

    if ema20 > ema50 > ema200:
        trend = "up"
    elif ema20 < ema50 < ema200:
        trend = "down"
    else:
        trend = "mixed"

    block = {
        "tf": tf,
        "last_candles": df_to_last_candles(df, max_rows=min(use_limit, 120)),
        "indicators": {
            "close": close,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "rsi14": float(df["rsi_14"].iloc[-1]),
            "macd": float(df["macd"].iloc[-1]),
            "macd_signal": float(df["macd_signal"].iloc[-1]),
            "macd_hist": float(df["macd_hist"].iloc[-1]),
            "bb_upper": float(df["bb_upper"].iloc[-1]),
            "bb_basis": float(df["bb_basis"].iloc[-1]),
            "bb_lower": float(df["bb_lower"].iloc[-1]),
            "atr14": float(df["atr_14"].iloc[-1]),
            "vol_z": float(df["vol_z"].iloc[-1]),
            "dist_to_high": float(df["dist_to_range_high"].iloc[-1]),
            "dist_to_low": float(df["dist_to_range_low"].iloc[-1]),
        },
        "structure": {
            "trend": trend,
            "dist_to_range_high": float(df["dist_to_range_high"].iloc[-1]),
            "dist_to_range_low": float(df["dist_to_range_low"].iloc[-1]),
        },
    }
    return block
