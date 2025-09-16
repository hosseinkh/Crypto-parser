# utiles/snapshot.py
# -----------------------------------------------------------
# Build a snapshot for a set of symbols using the provided exchange.
# IMPORTANT: this module does NOT define make_exchange (to avoid clashes).
# -----------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

try:
    from utiles.indicators import compute_indicators
except Exception:
    from .indicators import compute_indicators  # type: ignore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct_change_over_n(df: pd.DataFrame, n: int) -> float:
    if df.shape[0] <= n:
        return float("nan")
    c0 = float(df["close"].iloc[-n - 1])
    c1 = float(df["close"].iloc[-1])
    if c0 == 0:
        return float("nan")
    return (c1 - c0) / c0 * 100.0


def _zscore_last(series: pd.Series, window: int, ddof: int = 1) -> float:
    if series.shape[0] < window:
        return float("nan")
    mu = series.rolling(window=window, min_periods=window).mean().iloc[-1]
    sd = series.rolling(window=window, min_periods=window).std(ddof=ddof).iloc[-1]
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((series.iloc[-1] - mu) / sd)


def _compute_24h_vol_z(df: pd.DataFrame, bars_24h: int, window: int = 20) -> float:
    if df.shape[0] < bars_24h + window:
        return float("nan")
    sums = df["volume"].rolling(window=bars_24h, min_periods=bars_24h).sum()
    return _zscore_last(sums.dropna(), window=window, ddof=1)


def build_snapshot(
    exchange,
    symbols: Iterable[str],
    *,
    timeframe: str = "15m",
    limit: int = 240,
) -> Dict[str, Any]:
    """
    Build a lightweight snapshot for UI:
      - last price
      - 4h % change
      - 24h volume z-score
      - 15m RSI
      - distances to recent range (if available from indicators)
    """
    bars_4h = {"5m": 48, "15m": 16, "30m": 8, "1h": 4}.get(timeframe, 16)
    bars_24h = {"5m": 288, "15m": 96, "30m": 48, "1h": 24}.get(timeframe, 96)

    out: Dict[str, Any] = {"generated_at": _utc_iso(), "timeframe": timeframe, "items": []}

    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 60:
                continue
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            ind = compute_indicators(df)

            snap = {
                "symbol": sym,
                "last": float(df["close"].iloc[-1]),
                "pct4h": _pct_change_over_n(df, bars_4h),
                "vol_z24h": _compute_24h_vol_z(df, bars_24h, window=20),
                "rsi14_15m": float(ind["rsi_14"].iloc[-1]) if "rsi_14" in ind else None,
                "dist_to_low_pct": float(ind["dist_to_range_low"].iloc[-1]) if "dist_to_range_low" in ind else None,
                "dist_to_high_pct": float(ind["dist_to_range_high"].iloc[-1]) if "dist_to_range_high" in ind else None,
            }
            out["items"].append(snap)
        except Exception:
            # skip symbol on any error
            continue

    return out


__all__ = ["build_snapshot"]
