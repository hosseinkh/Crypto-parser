# utiles/indicators.py
# Minimal, clean indicator helpers (no exotic deps)

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- basic MAs ----------
def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length, min_periods=length).mean()


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


# ---------- RSI ----------
def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=s.index)
    loss = pd.Series(loss, index=s.index)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")


# ---------- ATR ----------
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """df must have columns: 'high','low','close'"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


# ---------- MACD ----------
def macd(
    s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------- Bollinger Bands ----------
def bbands(
    s: pd.Series, length: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = sma(s, length)
    dev = s.rolling(length, min_periods=length).std()
    upper = basis + num_std * dev
    lower = basis - num_std * dev
    return upper, basis, lower


# ---------- “structure” helpers ----------
def rolling_extrema(
    s: pd.Series, lookback: int = 20
) -> tuple[pd.Series, pd.Series]:
    rh = s.rolling(lookback, min_periods=1).max()
    rl = s.rolling(lookback, min_periods=1).min()
    return rh, rl


def distance_to_levels(
    price: pd.Series, high_lvls: pd.Series, low_lvls: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dist_to_high": (high_lvls - price) / price,
            "dist_to_low": (price - low_lvls) / price,
        },
        index=price.index,
    )


# ---------- one-stop compute for a candle DF ----------
def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    ema_trend: int = 50,
    rsi_len: int = 14,
    atr_len: int = 14,
    bb_len: int = 20,
) -> pd.DataFrame:
    """
    Expects columns: 'open','high','low','close','volume'
    Returns original df + indicator columns.
    """
    out = df.copy()

    out[f"ema_{ema_fast}"] = ema(out["close"], ema_fast)
    out[f"ema_{ema_slow}"] = ema(out["close"], ema_slow)
    out[f"ema_{ema_trend}"] = ema(out["close"], ema_trend)

    out[f"rsi_{rsi_len}"] = rsi(out["close"], rsi_len)
    out[f"atr_{atr_len}"] = atr(out, atr_len)

    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    bb_u, bb_b, bb_l = bbands(out["close"], bb_len)
    out["bb_upper"] = bb_u
    out["bb_basis"] = bb_b
    out["bb_lower"] = bb_l

    rh, rl = rolling_extrema(out["close"], lookback=20)
    dists = distance_to_levels(out["close"], rh, rl)
    out["dist_to_range_high"] = dists["dist_to_high"]
    out["dist_to_range_low"] = dists["dist_to_low"]

    return out

# ---------- Volume Z-Score ----------
def volume_zscore(v: pd.Series, length: int = 20) -> pd.Series:
    """
    Rolling z-score of volume. Values > ~2 often signal volume spikes.
    """
    mean = v.rolling(length, min_periods=length).mean()
    std = v.rolling(length, min_periods=length).std()
    z = (v - mean) / std.replace(0, np.nan)
    return z.fillna(0)

