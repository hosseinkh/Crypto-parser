# utiles/trending.py
# --------------------------------------------------------------------
# Trend scanner blending technicals + sentiment:
# - 4h momentum (pct change across ~16 x 15m candles)
# - 24h volume z-score (from rolling z on 15m; or a coarse proxy if not available)
# - 15m RSI guard-rails
# - Proximity to support (dist_to_range_low <= 2.5%)
# - Sentiment score from utiles.sentiment
# Returns: (DataFrame, passing_symbols_list)
# --------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .snapshot import safe_fetch_ohlcv, make_exchange
from .indicators import compute_indicators
from .sentiment import get_sentiment_for_symbol


@dataclass
class TrendScanParams:
    # Technical gates
    min_pct_4h: float = 0.5        # +0.5% in last ~4h
    min_vol_z_24h: float = 0.8     # strong 24h volume spike
    rsi_bounds_15m: Tuple[float, float] = (40.0, 65.0)
    max_dist_to_low_pct: float = 4.0  # allow looser than playbook for trend discovery

    # Blending weights
    w_vol: float = 0.4
    w_momo: float = 0.4
    w_sent: float = 0.2
    bonus_near_support: float = 0.1   # extra if dist_to_low <= 2.5%

    # Data params
    limit_15m: int = 240
    limit_1h: int = 240


def _pct_change(a: float, b: float) -> float:
    # pct change from b -> a in %
    try:
        return (a - b) / b * 100.0 if b else 0.0
    except Exception:
        return 0.0


def _zscore_last(series: np.ndarray) -> float:
    if series is None or len(series) < 20:
        return 0.0
    mu = float(series.mean())
    sd = float(series.std(ddof=1))
    if sd == 0:
        return 0.0
    return float((series[-1] - mu) / sd)


def scan_trending(
    exchange_name: str = "bitget",
    universe: Optional[List[str]] = None,
    params: Optional[TrendScanParams] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Scan a universe and compute a blended trend score.

    Returns:
      df: columns [symbol, score, pct4h, vol_z24h, sentiment_score, rsi14, last, dist_to_low_pct, ...]
      passing: list of symbols that pass all hard gates
    """
    if params is None:
        params = TrendScanParams()
    if universe is None:
        universe = ["BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","DOT/USDT","LINK/USDT","TRX/USDT","XRP/USDT","BCH/USDT","BNB/USDT"]

    ex = make_exchange(exchange_name)
    rows: List[Dict[str, float | str]] = []

    for sym in universe:
        try:
            # 15m block for indicators & distances
            ohlcv_15 = safe_fetch_ohlcv(ex, sym, timeframe="15m", limit=params.limit_15m)
            df15 = pd.DataFrame(ohlcv_15, columns=["timestamp","open","high","low","close","volume"])
            if df15.shape[0] < 60:
                continue
            ind15 = compute_indicators(df15, fill_partial=True)

            last = float(df15["close"].iloc[-1])
            dist_to_low_pct = float(ind15["dist_to_range_low"].iloc[-1])
            rsi14 = float(ind15["rsi_14"].iloc[-1])

            # Momentum: approximate 4h change using 16*15m candles
            close_now = float(df15["close"].iloc[-1])
            close_4h_ago = float(df15["close"].iloc[-16]) if df15.shape[0] >= 16 else float(df15["close"].iloc[0])
            pct4h = _pct_change(close_now, close_4h_ago)

            # 24h volume z-score (coarse) — use last 96*15m ≈ 24h if available
            vol_series = df15["volume"].tail(96).to_numpy(dtype=float)
            vol_z24h = _zscore_last(vol_series)

            # Sentiment (news-based)
            s = get_sentiment_for_symbol(sym)
            s_score = float(s.get("score", 0.0))

            rows.append({
                "symbol": sym,
                "last": last,
                "rsi14": rsi14,
                "pct4h": pct4h,
                "vol_z24h": vol_z24h,
                "sentiment_score": s_score,
                "dist_to_low_pct": dist_to_low_pct,
            })
        except Exception:
            # skip symbol on any error
            continue

    if not rows:
        return pd.DataFrame(columns=["symbol","score"]), []

    df = pd.DataFrame(rows)

    # Hard gates (pass/fail)
    lo, hi = params.rsi_bounds_15m
    df["gate_rsi"] = (df["rsi14"] >= lo) & (df["rsi14"] <= hi)
    df["gate_pct4h"] = df["pct4h"] >= params.min_pct_4h
    df["gate_vol"] = df["vol_z24h"] >= params.min_vol_z_24h
    df["gate_dist"] = df["dist_to_low_pct"] <= params.max_dist_to_low_pct

    # Normalize features for scoring (robust to outliers)
    for col in ["vol_z24h", "pct4h", "sentiment_score"]:
        mu = float(df[col].mean())
        sd = float(df[col].std(ddof=1)) or 1.0
        df[f"{col}_norm"] = (df[col] - mu) / sd

    def blend_row(r):
        score = params.w_vol * r["vol_z24h_norm"] + params.w_momo * r["pct4h_norm"] + params.w_sent * r["sentiment_score_norm"]
        if r.get("dist_to_low_pct", 999) <= 2.5:
            score += params.bonus_near_support
        return float(score)

    df["score"] = df.apply(blend_row, axis=1)

    # Passing list: must satisfy all gates
    passing_df = df[(df["gate_rsi"]) & (df["gate_pct4h"]) & (df["gate_vol"]) & (df["gate_dist"])]
    passing = passing_df.sort_values("score", ascending=False)["symbol"].tolist()

    # Sort display by score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df, passing
