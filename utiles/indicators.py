# utiles/indicators.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------- Basic Indicators -----------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def ema(series: pd.Series, length: int = 20) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ----------------- Extras (optional) -----------------
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist

def bbands(close: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower

def rolling_extrema(series: pd.Series, lookback: int = 20):
    rh = series.rolling(lookback, min_periods=1).max()
    rl = series.rolling(lookback, min_periods=1).min()
    return rh, rl

def distance_to_levels(close: pd.Series, rh: pd.Series, rl: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "dist_to_high": (rh - close) / close,
        "dist_to_low" : (close - rl) / close
    })

# ----------------- Volume Z-Score (the missing one) -----------------
def volume_zscore(v: pd.Series, length: int = 20) -> pd.Series:
    """
    Rolling z-score of volume. Values > ~2 can indicate spikes.
    """
    mean = v.rolling(length, min_periods=length).mean()
    std = v.rolling(length, min_periods=length).std()
    z = (v - mean) / std.replace(0, np.nan)
    return z.fillna(0)

# ----------------- Helper to bundle many indicators -----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["atr14"] = atr(out, 14)
    macd_line, macd_sig, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    bb_u, bb_m, bb_l = bbands(out["close"], 20, 2)
    out["bb_upper"] = bb_u
    out["bb_mid"]   = bb_m
    out["bb_lower"] = bb_l
    rh, rl = rolling_extrema(out["close"], 20)
    dists = distance_to_levels(out["close"], rh, rl)
    out["dist_to_high"] = dists["dist_to_high"]
    out["dist_to_low"]  = dists["dist_to_low"]
    out["vol_z20"] = volume_zscore(out["volume"], 20)
    return out

# (optional) explicit export list
__all__ = [
    "rsi","ema","atr","macd","bbands",
    "rolling_extrema","distance_to_levels",
    "volume_zscore","compute_indicators"
]
