# utiles/snapshot.py
import numpy as np
import pandas as pd
from datetime import timezone
from . import indicators as ind   # <-- important: module import, not star-import

# ---- helpers ----
def df_to_last_candles(df: pd.DataFrame, max_rows: int = 50):
    """
    Return the last `max_rows` candles in a compact list form:
    [{"t": iso, "o":..., "h":..., "l":..., "c":..., "v":...}, ...]
    """
    rows = df.tail(max_rows)
    # Expect df columns: time(UTC ms or iso), open, high, low, close, volume
    # Normalize timestamp to ISO
    if np.issubdtype(rows["time"].dtype, np.number):
        times = pd.to_datetime(rows["time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        times = pd.to_datetime(rows["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out = []
    for t, o, h, l, c, v in zip(times, rows["open"], rows["high"], rows["low"], rows["close"], rows["volume"]):
        out.append({"t": str(t), "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)})
    return out


def summarize_frame(df: pd.DataFrame) -> dict:
    """
    Compute a small summary dict for the frame (uses indicators.py).
    Assumes df columns: open, high, low, close, volume
    """
    out = {}

    # Core indicators (safe defaults if not enough history)
    rsi = ind.rsi(df["close"]).iloc[-1]
    ema_fast = ind.ema(df["close"], 21).iloc[-1]
    ema_slow = ind.ema(df["close"], 50).iloc[-1]
    atr_val = ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1]
    vol_z = ind.volume_zscore(df["volume"], 20).iloc[-1]

    out["price"] = float(df["close"].iloc[-1])
    out["rsi"] = float(np.nan_to_num(rsi, nan=50.0))
    out["ema_fast"] = float(np.nan_to_num(ema_fast, nan=out["price"]))
    out["ema_slow"] = float(np.nan_to_num(ema_slow, nan=out["price"]))
    out["atr"] = float(np.nan_to_num(atr_val, nan=0.0))
    out["vol_z"] = float(np.nan_to_num(vol_z, nan=0.0))

    # Simple trend tag
    if out["ema_fast"] > out["ema_slow"]:
        out["trend"] = "up"
    elif out["ema_fast"] < out["ema_slow"]:
        out["trend"] = "down"
    else:
        out["trend"] = "flat"

    return out


# ---- main entry expected by streamlit_app.py ----
def build_tf_block(ex, symbol: str, timeframe: str, limit: int, now_iso: str) -> dict:
    """
    Fetch OHLCV for (symbol, timeframe, limit), compute indicators and return a
    timeframe block the UI/JSON expects.
    `ex` is the ccxt exchange instance.
    """
    # 1) fetch data
    df = ex.fetch_ohlcv_df(symbol=symbol, timeframe=timeframe, limit=limit)
    # Safety: ensure columns
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {missing}")

    # 2) indicators + compact candles
    summary = summarize_frame(df)
    candles = df_to_last_candles(df, max_rows=min(limit, 120))

    # 3) frame dict
    frame = {
        "tf": timeframe,
        "n_candles": int(len(df)),
        "last_candles": candles,
        "indicators": {
            "rsi": summary["rsi"],
            "ema_fast": summary["ema_fast"],
            "ema_slow": summary["ema_slow"],
            "atr": summary["atr"],
            "vol_z": summary["vol_z"],
        },
        "structure": {
            "trend": summary["trend"],
            # placeholders you can enrich later:
            "local_sr": None,
            "breakout": None,
            "divergence": None,
        },
        "notes": [],
    }
    return frame
