# utiles/trending.py
# -----------------------------------------------------------
# Trend scanner + human-readable reasons for why a symbol
# was added. Works with your indicators engine.
# -----------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any
import math
import numpy as np
import pandas as pd

# Use your indicator engine
try:
    from utiles.indicators import compute_indicators
except Exception:  # local dev fallback
    from .indicators import compute_indicators  # type: ignore

# Optional sentiment hook — safe if missing
def _get_sentiment_score(symbol: str) -> float:
    try:
        from utiles.sentiment import get_sentiment_for_symbol
        score, _label = get_sentiment_for_symbol(symbol)
        return float(score)
    except Exception:
        return 0.0


# ---------- Parameters / thresholds ----------
@dataclass
class TrendScanParams:
    timeframe: str = "15m"
    limit: int = 240
    # thresholds
    min_pct_4h: float = 0.5            # >= +0.5% over ~4h
    min_volz_24h: float = 0.8          # 24h volume anomaly (z-score)
    rsi15m_min: float = 35.0
    rsi15m_max: float = 65.0
    min_sentiment: float = 0.05
    near_support_max_pct: float = 2.5
    near_resist_max_pct: float = 1.5

    def bars_for_4h(self) -> int:
        return {"5m": 48, "15m": 16, "30m": 8, "1h": 4}.get(self.timeframe, 16)

    def bars_for_24h(self) -> int:
        return {"5m": 288, "15m": 96, "30m": 48, "1h": 24}.get(self.timeframe, 96)


# ---------- Internal helpers ----------
def _pct_change_over_n(df: pd.DataFrame, n: int) -> float:
    if df.shape[0] <= n:
        return float("nan")
    c0 = float(df["close"].iloc[-n-1])
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
    """z-score of rolling 24h volume sums over a trailing window."""
    if df.shape[0] < bars_24h + window:
        return float("nan")
    sums = df["volume"].rolling(window=bars_24h, min_periods=bars_24h).sum()
    return _zscore_last(sums.dropna(), window=window, ddof=1)


# ---------- Human-readable reasons ----------
def explain_trending_row(row: pd.Series, p: TrendScanParams) -> List[str]:
    """
    Build reasons for why a row (symbol) is considered 'trending'.
    Expects columns: pct4h, vol_z24h, rsi14_15m, sentiment_score,
                     dist_to_low_pct, dist_to_high_pct.
    """
    reasons: List[str] = []
    if pd.notna(row.get("pct4h")) and row["pct4h"] >= p.min_pct_4h:
        reasons.append(f"4h momentum +{row['pct4h']:.2f}%")
    if pd.notna(row.get("vol_z24h")) and row["vol_z24h"] >= p.min_volz_24h:
        reasons.append(f"24h volume spike (z={row['vol_z24h']:.2f})")
    rsi = row.get("rsi14_15m")
    if pd.notna(rsi) and p.rsi15m_min <= rsi <= p.rsi15m_max:
        reasons.append(f"15m RSI in range ({rsi:.1f})")
    sent = row.get("sentiment_score")
    if pd.notna(sent) and sent >= p.min_sentiment:
        reasons.append(f"positive sentiment ({sent:+.2f})")
    dlow = row.get("dist_to_low_pct")
    if pd.notna(dlow) and dlow <= p.near_support_max_pct:
        reasons.append(f"near support ({dlow:.2f}% from low)")
    dhigh = row.get("dist_to_high_pct")
    if pd.notna(dhigh) and dhigh <= p.near_resist_max_pct:
        reasons.append(f"near resistance ({dhigh:.2f}% from high)")
    return reasons


# ---------- Main scan ----------
def scan_trending(
    exchange,
    universe: Optional[Iterable[str]] = None,
    params: Optional[TrendScanParams] = None,
) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      symbol, last, pct4h, vol_z24h, rsi14_15m, sentiment_score,
      dist_to_low_pct, dist_to_high_pct, score
    """
    if params is None:
        params = TrendScanParams()

    if universe is None:
        universe = [s for s in exchange.symbols if s.endswith("/USDT")]
    universe = list(universe)

    rows: List[Dict[str, Any]] = []
    for sym in universe:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=params.timeframe, limit=params.limit)
            if not ohlcv or len(ohlcv) < 60:
                continue
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            ind = compute_indicators(df)

            bars_4h = params.bars_for_4h()
            bars_24h = params.bars_for_24h()

            pct4h = _pct_change_over_n(df, n=bars_4h)
            volz24 = _compute_24h_vol_z(df, bars_24h=bars_24h, window=20)

            dist_low_pct  = float(ind["dist_to_range_low"].iloc[-1])   if "dist_to_range_low"  in ind else float("nan")
            dist_high_pct = float(ind["dist_to_range_high"].iloc[-1])  if "dist_to_range_high" in ind else float("nan")
            rsi_15m = float(ind["rsi_14"].iloc[-1]) if "rsi_14" in ind else float("nan")

            rows.append({
                "symbol": sym,
                "last": float(df["close"].iloc[-1]),
                "pct4h": pct4h,
                "vol_z24h": volz24,
                "rsi14_15m": rsi_15m,
                "sentiment_score": _get_sentiment_score(sym),
                "dist_to_low_pct": dist_low_pct,
                "dist_to_high_pct": dist_high_pct,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","last","pct4h","vol_z24h","rsi14_15m","sentiment_score",
            "dist_to_low_pct","dist_to_high_pct","score"
        ])

    df = pd.DataFrame(rows)

    # Blend score
    def _z(col: pd.Series) -> pd.Series:
        sd = col.std()
        return (col - col.mean()) / (sd if sd and not math.isnan(sd) else 1.0)

    df["score"] = 0.4*_z(df["vol_z24h"].fillna(0)) + 0.4*_z(df["pct4h"].fillna(0)) + 0.2*_z(df["sentiment_score"].fillna(0))
    df.loc[df["dist_to_low_pct"]  <= params.near_support_max_pct, "score"] += 0.10
    df.loc[df["dist_to_high_pct"] <= params.near_resist_max_pct, "score"] += 0.05

    # Keep symbols meeting at least one threshold
    keep_mask = (
        (df["pct4h"] >= params.min_pct_4h) |
        (df["vol_z24h"] >= params.min_volz_24h) |
        ((df["rsi14_15m"] >= params.rsi15m_min) & (df["rsi14_15m"] <= params.rsi15m_max))
    )
    return df.loc[keep_mask].sort_values("score", ascending=False).reset_index(drop=True)


# Exported names
__all__ = ["scan_trending", "explain_trending_row", "TrendScanParams"]
