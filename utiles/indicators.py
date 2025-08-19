# utiles/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict

# ----------------- Core helpers -----------------

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bbands(s: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    basis = s.rolling(length, min_periods=length).mean()
    dev = s.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return upper, basis, lower

def _atr_from_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def atr(*args, **kwargs) -> pd.Series:
    """
    Flexible ATR:
      atr(df, length=14)                # df has high/low/close
      atr(high, low, close, length=14)  # separate series
    """
    length = int(kwargs.get("length", 14))

    # Form 1: single DataFrame
    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        return _atr_from_series(df["high"], df["low"], df["close"], length=length)

    # Form 2: 3 Series
    if len(args) >= 3:
        h, l, c = [pd.Series(x) for x in args[:3]]
        return _atr_from_series(h, l, c, length=length)

    raise TypeError("atr() expects either (df, length=) or (high, low, close, length=).")

def rolling_extrema(s: pd.Series, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    highest = s.rolling(lookback, min_periods=1).max()
    lowest = s.rolling(lookback, min_periods=1).min()
    return highest, lowest

def distance_to_levels(price: pd.Series, high_level: pd.Series, low_level: pd.Series) -> Dict[str, pd.Series]:
    return {
        "dist_to_high": (high_level - price) / price,
        "dist_to_low": (price - low_level) / price,
    }

def volume_zscore(v: pd.Series, length: int = 20) -> pd.Series:
    """
    Rolling Z-score of volume. Values > ~2 often signal volume spikes.
    """
    mean = v.rolling(length, min_periods=length).mean()
    std = v.rolling(length, min_periods=length).std()
    z = (v - mean) / std.replace(0, np.nan)
    return z.fillna(0)

# ----------------- High-level compute -----------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame (copy) with common indicators added.
    Assumes df has columns: ts, open, high, low, close, volume
    """
    out = df.copy()

    # EMAs & RSI
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)

    # MACD
    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Bollinger Bands
    bb_u, bb_b, bb_l = bbands(out["close"], 20, 2.0)
    out["bb_upper"] = bb_u
    out["bb_basis"] = bb_b
    out["bb_lower"] = bb_l

    # ATR
    out["atr_14"] = atr(out["high"], out["low"], out["close"], length=14)

    # Range & distances
    rh, rl = rolling_extrema(out["close"], lookback=20)
    dists = distance_to_levels(out["close"], rh, rl)
    out["dist_to_range_high"] = dists["dist_to_high"]
    out["dist_to_range_low"] = dists["dist_to_low"]

    # Volume z-score
    out["vol_z"] = volume_zscore(out["volume"], length=20)

    return out
