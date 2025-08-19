# utiles/bitget.py
from __future__ import annotations

from typing import Final
import ccxt
import pandas as pd
from datetime import datetime, timezone

# Map our UI timeframes to CCXT
TIMEFRAME_MAP: Final[dict[str, str]] = {
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
}

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def make_exchange() -> ccxt.bitget:
    ex = ccxt.bitget()
    ex.load_markets()
    return ex

def normalize_symbol(symbol: str, quote: str = "USDT") -> str:
    # Accept "FET/USDT" or "FET"
    if "/" in symbol:
        return symbol.upper()
    return f"{symbol.upper()}/{quote}"

# NOTE: make 'limit' positional-only using '/' so kwargs like limit=... will raise
def fetch_ohlcv_df(ex: ccxt.bitget, symbol: str, tf: str, limit, /) -> pd.DataFrame:
    """
    Fetch OHLCV and return as a sorted DataFrame with columns:
    ts (datetime), open, high, low, close, volume
    """
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}")

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME_MAP[tf], limit=int(limit))
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    # Convert ms -> UTC datetime, sort ascending
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df
