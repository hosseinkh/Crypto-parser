# utiles/trending.py
# --------------------------------------------------------------------
# Trend scanner that returns a ranked DataFrame with all columns needed
# to justify why each coin is "trending". Also exposes a helper to
# produce human-readable reasons per row.
# --------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from .indicators import compute_indicators  # our unified TA engine


# --------------------- Parameters & helpers --------------------------

@dataclass
class TrendScanParams:
    timeframe_15m: str = "15m"
    timeframe_1h: str = "1h"
    limit_15m: int = 240       # enough for vol_z & ranges
    limit_1h: int = 200

    # selection thresholds
    min_pct_4h: float = 0.5    # >= +0.5% over last ~4h
    min_vol_z_24h: float = 0.8 # day-over-day volume anomaly on 1h
    rsi_bounds_15m: Tuple[float, float] = (35.0, 65.0)
    min_sentiment: float = 0.05
    near_support_max_pct: float = 2.5
    near_resist_max_pct: float = 1.5


def _safe(series: pd.Series, idx: int = -1, default: float = np.nan) -> float:
    try:
        return float(series.iloc[idx])
    except Exception:
        return float(default)


def _pct_change_last_n(close: pd.Series, bars: int) -> float:
    if close.shape[0] < bars + 1:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-bars - 1] - 1.0) * 100.0)


def _vol_z_24h_from_1h(df_1h: pd.DataFrame) -> float:
    """Use 1h volumes, compute z-score of last 24 bars vs prior 24-bar window mean/sd."""
    vols = df_1h["volume"].astype(float)
    if vols.shape[0] < 48:
        return np.nan
    last24 = vols.iloc[-24:]
    prev24 = vols.iloc[-48:-24]
    mu = prev24.mean()
    sd = prev24.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((last24.mean() - mu) / sd)


# -------------------------- Public API --------------------------------

def scan_trending(
    exchange,
    universe: Optional[Iterable[str]] = None,
    params: Optional[TrendScanParams] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    symbol, last, pct4h, vol_z24h, rsi14_15m, sentiment_score,
    dist_to_low_pct, dist_to_high_pct, score

    `exchange` is a ccxt instance (already created).
    `universe` if None -> scan all /USDT symbols on the exchange.
    """
    if params is None:
        params = TrendScanParams()

    # Build universe
    try:
        all_syms = [s for s in exchange.symbols if s.endswith("/USDT")]
    except Exception:
        all_syms = []
    symbols = list(universe) if universe else all_syms

    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            # ---- 15m fetch & indicators (for RSI, distances, last) ----
            ohlcv_15 = exchange.fetch_ohlcv(sym, timeframe=params.timeframe_15m, limit=params.limit_15m)
            df15 = pd.DataFrame(ohlcv_15, columns=["timestamp","open","high","low","close","volume"])
            if df15.empty or df15.shape[0] < 60:
                continue
            ind15 = compute_indicators(df15)

            # ---- 1h fetch for pct(4h) & day-over-day volume anomaly ----
            ohlcv_1h = exchange.fetch_ohlcv(sym, timeframe=params.timeframe_1h, limit=params.limit_1h)
            df1h = pd.DataFrame(ohlcv_1h, columns=["timestamp","open","high","low","close","volume"])
            if df1h.empty or df1h.shape[0] < 30:
                continue

            last = _safe(df15["close"])
            pct4h = _pct_change_last_n(df1h["close"], bars=4)  # ~4 hours on 1h
            volz24h = _vol_z_24h_from_1h(df1h)

            rsi15 = _safe(ind15["rsi_14"])
            dlow = _safe(ind15["dist_to_range_low"])
            dhigh = _safe(ind15["dist_to_range_high"])

            # sentiment_score is optional — if your sentiment module is not wired,
            # set to 0.0. If you have a function, call it here.
            sentiment_score = 0.0

            rows.append({
                "symbol": sym,
                "last": last,
                "pct4h": pct4h,
                "vol_z24h": volz24h,
                "rsi14_15m": rsi15,
                "sentiment_score": sentiment_score,
                "dist_to_low_pct": dlow,
                "dist_to_high_pct": dhigh,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","last","pct4h","vol_z24h","rsi14_15m","sentiment_score",
            "dist_to_low_pct","dist_to_high_pct","score"
        ])

    df = pd.DataFrame(rows)

    # Normalize & blended score
    z = lambda s: (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) else 1.0)
    for col in ["vol_z24h","pct4h","sentiment_score"]:
        if col not in df.columns:
            df[col] = 0.0
    df["score"] = 0.4*z(df["vol_z24h"]) + 0.4*z(df["pct4h"]) + 0.2*z(df["sentiment_score"])

    # small bonuses
    df.loc[df["dist_to_low_pct"] <= params.near_support_max_pct, "score"] += 0.10
    df.loc[df["dist_to_high_pct"] <= params.near_resist_max_pct, "score"] += 0.05

    # basic guard-rails
    lo, hi = params.rsi_bounds_15m
    df = df[(df["rsi14_15m"].between(lo, hi, inclusive="both")) | df["rsi14_15m"].isna()]

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def explain_trending_row(row: pd.Series, params: Optional[TrendScanParams] = None) -> list[str]:
    """Return human-readable reasons for selection, grounded in row values."""
    if params is None:
        params = TrendScanParams()

    reasons: List[str] = []
    if pd.notna(row.get("pct4h")) and row["pct4h"] >= params.min_pct_4h:
        reasons.append(f"4h momentum +{row['pct4h']:.2f}%")
    if pd.notna(row.get("vol_z24h")) and row["vol_z24h"] >= params.min_vol_z_24h:
        reasons.append(f"24h volume spike (z={row['vol_z24h']:.2f})")
    rsi = row.get("rsi14_15m")
    lo, hi = params.rsi_bounds_15m
    if pd.notna(rsi) and lo <= rsi <= hi:
        reasons.append(f"15m RSI in range ({rsi:.1f})")
    sent = row.get("sentiment_score")
    if pd.notna(sent) and sent >= params.min_sentiment:
        reasons.append(f"positive sentiment ({sent:+.2f})")
    dlow = row.get("dist_to_low_pct")
    if pd.notna(dlow) and dlow <= params.near_support_max_pct:
        reasons.append(f"near support ({dlow:.2f}% from low)")
    dhigh = row.get("dist_to_high_pct")
    if pd.notna(dhigh) and dhigh <= params.near_resist_max_pct:
        reasons.append(f"near resistance ({dhigh:.2f}% from high)")
    return reasons
