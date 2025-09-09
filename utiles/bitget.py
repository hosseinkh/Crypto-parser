# utiles/bitget.py
from __future__ import annotations

from typing import Final
import ccxt
import pandas as pd
from datetime import datetime, timezone

# Map UI timeframes to CCXT timeframes
TIMEFRAME_MAP: Final[dict[str, str]] = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}

def now_utc_iso() -> str:
    """UTC timestamp for snapshot stamping, ISO-8601 Zulu."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def make_exchange() -> ccxt.bitget:
    """Bitget spot client with sane defaults."""
    ex = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return ex

def normalize_symbol(symbol: str, quote: str = "USDT") -> str:
    """Accept 'FET/USDT' or 'FET' and normalize to 'FET/USDT'."""
    return symbol.upper() if "/" in symbol else f"{symbol.upper()}/{quote}"

# IMPORTANT: 'limit' is positional-only. Do NOT call with limit=...
def fetch_ohlcv_df(ex: ccxt.bitget, symbol: str, tf: str, limit, /) -> pd.DataFrame:
    """Fetch OHLCV into a clean, time-sorted DataFrame (UTC)."""
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}")

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME_MAP[tf], limit=int(limit))
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df
