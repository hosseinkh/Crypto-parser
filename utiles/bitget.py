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
