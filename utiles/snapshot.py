# utiles/snapshot.py

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timezone

from utiles.bitget import fetch_ohlcv_df, now_utc_iso
from utiles.indicators import (
    rsi,
    ema,
    atr,
    bbands,
    rolling_extrema,
    distance_to_levels,
    volume_zscore,
)


def df_to_last_candles(df: pd.DataFrame, max_rows: int) -> list[dict]:
    """
    Convert the most-recent rows of OHLCV DataFrame into the compact JSON-friendly list.
    Expects columns: ts (datetime), open, high, low, close, volume
    """
    rows = df.tail(int(max_rows)).copy()
    # Ensure timezone-aware RFC3339 strings
    rows["t"] = rows["ts"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT", errors="coerce")
    rows["t"] = rows["t"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    cols = ["t", "open", "high", "low", "close", "volume"]
    out = rows[cols].to_dict(orient="records")
    return out


def build_tf_block(
    ex,
    symbol: str,
    tf: str,
    lc_count: int,
) -> dict:
    """
    Build the per-timeframe block for one symbol.
    - ex: ccxt exchange instance
    - symbol: e.g. "FET/USDT"
    - tf: "15m" | "1h" | "4h"
    - lc_count: number of last candles to include
    """
    # ---- Fetch OHLCV once (PASS LIMIT ONLY ONCE!) ----
    df = fetch_ohlcv_df(ex, symbol, tf, int(lc_count))  # <— only positional 'limit'

    # ---- Indicators (vectorized) ----
    close = df["close"]
    vol   = df["volume"]

    rsi14 = rsi(close, length=14)
    ema12 = ema(close, length=12)
    ema26 = ema(close, length=26)
    atr14 = atr(df, length=14)

    bb_u, bb_b, bb_l = bbands(close, length=20)

    rh, rl = rolling_extrema(close, lookback=20)  # recent high/low
    dists  = distance_to_levels(close, rh, rl)   # distances to range high/low

    vz20  = volume_zscore(vol, length=20)

    last_close = float(close.iloc[-1])

    # Simple trend heuristic
    trend_ema = "up" if ema12.iloc[-1] > ema26.iloc[-1] else ("down" if ema12.iloc[-1] < ema26.iloc[-1] else "flat")

    # ---- Assemble block ----
    block = {
        "timeframe": tf,
        "last_candles": df_to_last_candles(df, lc_count),
        "indicators": {
            "rsi": round(float(rsi14.iloc[-1]), 3),
            "ema_fast": round(float(ema12.iloc[-1]), 6),
            "ema_slow": round(float(ema26.iloc[-1]), 6),
            "atr": round(float(atr14.iloc[-1]), 6),
            "bb_upper": round(float(bb_u.iloc[-1]), 6),
            "bb_basis": round(float(bb_b.iloc[-1]), 6),
            "bb_lower": round(float(bb_l.iloc[-1]), 6),
            "volume_z": round(float(vz20.iloc[-1]), 3),
        },
        "structure": {
            "trend_ema": trend_ema,
            "range_high": round(float(rh.iloc[-1]), 6),
            "range_low": round(float(rl.iloc[-1]), 6),
            "dist_to_high": round(float(dists["dist_to_high"].iloc[-1]), 6),
            "dist_to_low": round(float(dists["dist_to_low"].iloc[-1]), 6),
        },
        "raw": {
            "last_close": last_close,
        },
    }
    return block


def build_snapshot_for_symbol(ex, symbol: str, tf_counts: dict[str, int]) -> dict:
    """
    Build the full snapshot for one symbol across multiple timeframes.
    tf_counts example: {"15m": 30, "1h": 50, "4h": 80}
    """
    blocks: list[dict] = []
    errors: list[str] = []

    for tf, lc_count in tf_counts.items():
        try:
            blocks.append(build_tf_block(ex, symbol, tf, lc_count))
        except Exception as e:
            errors.append(f"{symbol} ({tf}) fetch error: {e}")

    snapshot = {
        "symbol": symbol,
        "ts_utc": now_utc_iso(),
        "timeframes": blocks,
    }
    if errors:
        snapshot["errors"] = errors
    return snapshot
