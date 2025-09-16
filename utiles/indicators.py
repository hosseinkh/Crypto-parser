# utiles/indicators.py
# --------------------------------------------------------------------
# Unified indicator engine for snapshot/export:
# - Works on a single TF OHLCV DataFrame (timestamp, open, high, low, close, volume)
# - Computes: EMA20/50/200, RSI14, MACD (hist), BB(20,2), ATR14,
#             volume z-score (rolling window), dist_to_range_high/low
# - Safe on short series (graceful NaNs), vectorized, no TA-lib required
# --------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------- Helper math ----------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def _std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=1)

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    # Wilder’s RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    # Wilder smoothing (EMA with alpha=1/period)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - macd_signal
    return hist

def _bb(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, window)
    sd = _std(close, window)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return lower, mid, upper

def _zscore_last(values: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    """Rolling z-score of the *last* value in each window (vectorized)."""
    m = values.rolling(window=window, min_periods=window).mean()
    s = values.rolling(window=window, min_periods=window).std(ddof=ddof)
    z = (values - m) / s.replace(0, np.nan)
    # For insufficient window, yield NaN; caller can fill with 0.0 if desired
    return z


# ---------- Public API ----------

@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    atr_period: int = 14
    ema_spans: Tuple[int, int, int] = (20, 50, 200)
    bb_window: int = 20
    bb_n_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_z_window: int = 20  # how many candles to compute z-score on
    # range distance lookbacks (in candles)
    range_low_lookback: int = 10    # support
    range_high_lookback: int = 15   # resistance

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame is missing required columns: {missing}")
    # enforce numeric types where expected
    for col in ["open", "high", "low", "close", "volume"]:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

def compute_indicators(
    df: pd.DataFrame,
    config: Optional[IndicatorConfig] = None,
    fill_partial: bool = True
) -> pd.DataFrame:
    """
    Compute all indicators required by the v3.3 playbook on a single timeframe.

    Input df columns:
      - timestamp (ms or ns), open, high, low, close, volume

    Returns:
      Original columns +:
        ema_20, ema_50, ema_200,
        rsi_14,
        macd_hist,
        bb_lower, bb_mid, bb_upper,
        atr_14,
        vol_z, vol_mu, vol_sigma,
        range_low, range_high,
        dist_to_range_low (%), dist_to_range_high (%)
    """
    if config is None:
        config = IndicatorConfig()

    df = df.copy()
    validate_ohlcv(df)

    # --- EMAs
    e20, e50, e200 = config.ema_spans
    df["ema_20"] = _ema(df["close"], e20)
    df["ema_50"] = _ema(df["close"], e50)
    df["ema_200"] = _ema(df["close"], e200)

    # --- RSI
    df["rsi_14"] = _rsi(df["close"], config.rsi_period)

    # --- MACD histogram
    df["macd_hist"] = _macd_hist(
        df["close"],
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
    )

    # --- Bollinger Bands
    bb_low, bb_mid, bb_up = _bb(df["close"], config.bb_window, config.bb_n_std)
    df["bb_lower"] = bb_low
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_up

    # --- ATR
    df["atr_14"] = _atr(df, config.atr_period)

    # --- Volume z-score (rolling, last value)
    z = _zscore_last(df["volume"], window=config.vol_z_window, ddof=1)
    df["vol_z"] = z
    # optional mu/sigma for debugging/inspection
    df["vol_mu"] = df["volume"].rolling(window=config.vol_z_window, min_periods=config.vol_z_window).mean()
    df["vol_sigma"] = df["volume"].rolling(window=config.vol_z_window, min_periods=config.vol_z_window).std(ddof=1)

    # --- Range levels and distances
    # range_low = absolute low of last N candles; range_high = absolute high of last M candles
    rl = df["low"].rolling(window=config.range_low_lookback, min_periods=config.range_low_lookback).min()
    rh = df["high"].rolling(window=config.range_high_lookback, min_periods=config.range_high_lookback).max()
    df["range_low"] = rl
    df["range_high"] = rh

    # Distance in percent from last close to the range bounds
    last_close = df["close"]
    df["dist_to_range_low"] = (last_close - rl) / last_close * 100.0
    df["dist_to_range_high"] = (rh - last_close) / last_close * 100.0

    # --- Optional filling for early rows (so snapshots don’t expose NaNs)
    if fill_partial:
        # Replace initial NaNs where reasonable:
        for col in [
            "ema_20", "ema_50", "ema_200",
            "rsi_14", "macd_hist",
            "bb_lower", "bb_mid", "bb_upper",
            "atr_14", "vol_z", "vol_mu", "vol_sigma",
            "range_low", "range_high",
            "dist_to_range_low", "dist_to_range_high",
        ]:
            # For z-score specifically, we *prefer* to leave NaN until we have enough window;
            # but for snapshot convenience, fill remaining NaNs with 0.0 (neutral) at the tail rows only.
            if col == "vol_z":
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(method="bfill").fillna(method="ffill")

    return df


# ---------- Convenience: last-row summary for snapshots ----------

def summarize_last_row(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract the latest indicator values in a compact dict for JSON export.

    Returns keys matching the playbook’s data expectations where possible.
    """
    if df.empty:
        return {}

    row = df.iloc[-1]
    # Defensive: .get with default to avoid KeyError if caller removed columns
    g = row.get

    return {
        "ema20": float(g("ema_20", np.nan)),
        "ema50": float(g("ema_50", np.nan)),
        "ema200": float(g("ema_200", np.nan)),
        "rsi14": float(g("rsi_14", np.nan)),
        "macd_hist": float(g("macd_hist", np.nan)),
        "bb_lower": float(g("bb_lower", np.nan)),
        "bb_upper": float(g("bb_upper", np.nan)),
        "atr14": float(g("atr_14", np.nan)),
        "vol_z": float(g("vol_z", np.nan)),
        "dist_to_range_low": float(g("dist_to_range_low", np.nan)),
        "dist_to_range_high": float(g("dist_to_range_high", np.nan)),
        "range_low": float(g("range_low", np.nan)),
        "range_high": float(g("range_high", np.nan)),
    }
