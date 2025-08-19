---

# 🧠 utils/indicators.py

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # df must have columns: high, low, close
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0)
    return (volume - mean) / (std.replace(0, np.nan))

def ema_trend(close: pd.Series, span: int = 50, eps: float = 0.0) -> str:
    e = ema(close, span)
    slope = e.diff().iloc[-5:].mean()
    if slope > eps: return "up"
    if slope < -eps: return "down"
    return "sideways"

def recent_range(close: pd.Series, window: int = 40) -> tuple[float, float]:
    window = min(window, len(close))
    _max = float(close.tail(window).max())
    _min = float(close.tail(window).min())
    return _min, _max

def swing_pattern(close: pd.Series, atr_series: pd.Series, mult: float = 1.5) -> str:
    # crude zigzag based on ATR threshold
    if len(close) < 5: return "range"
    thr = (atr_series.iloc[-1] if not atr_series.empty else (close.std() * 0.5))
    thr *= mult
    pivots = [close.iloc[0]]
    last = pivots[-1]
    for v in close.iloc[1:]:
        if abs(v - last) >= thr:
            pivots.append(v); last = v
    if len(pivots) < 3:
        return "range"
    # Check last two moves
    if pivots[-2] < pivots[-1] and min(pivots[-3], pivots[-2]) < pivots[-2]:
        return "HL>HH"
    if pivots[-2] > pivots[-1] and max(pivots[-3], pivots[-2]) > pivots[-2]:
        return "LH>LL"
    return "range"

def nearest_sr_from_pivots(close: pd.Series, window: int = 40) -> tuple[float, float]:
    s, r = recent_range(close, window)
    return s, r
