from __future__ import annotations
import ccxt
import pandas as pd
from datetime import datetime, timezone

TIMEFRAME_MAP = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def make_exchange() -> ccxt.bitget:
    ex = ccxt.bitget()
    ex.load_markets()
    return ex

def normalize_symbol(symbol: str, quote="USDT") -> str:
    # Expected input "FET/USDT" or "FET"
    if "/" in symbol:
        return symbol.upper()
    return f"{symbol.upper()}/{quote}"

def fetch_ohlcv_df(ex: ccxt.bitget, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME_MAP[tf], limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# ---- Convenience wrapper used by snapshot.py ----
def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV from Bitget without passing an exchange object.
    Returns columns: time, open, high, low, close, volume (UTC).
    """
    ex = make_exchange()
    sym = normalize_symbol(symbol)
    tf = TIMEFRAME_MAP.get(timeframe, timeframe)  # allow "15m", "1h", "4h"

    ohlcv = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    # Bitget/ccxt timestamp is ms — convert to UTC datetime then back to ISO only in snapshot
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df
