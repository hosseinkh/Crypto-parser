# utiles/bitget.py
from __future__ import annotations
from typing import Dict, List
from datetime import datetime, timezone
import requests
import pandas as pd

REQUEST_TIMEOUT = 12

TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day",
}

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _get(url: str, params: Dict | None = None) -> dict:
    r = requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def list_symbols_bitget(quote: str = "USDT") -> List[str]:
    """Return spot symbols quoted in `quote`, e.g. ['BTC/USDT','ETH/USDT', ...]"""
    url = "https://api.bitget.com/api/spot/v1/public/products"
    data = _get(url)
    items = data.get("data", []) if isinstance(data, dict) else []
    out: List[str] = []
    for it in items:
        if it.get("quoteAsset") == quote and it.get("status") == "online":
            base = it.get("baseAsset")
            if base:
                out.append(f"{base}/{quote}")
    return sorted(list(dict.fromkeys(out)))

def klines_bitget(symbol: str, timeframe: str, limit: int = 240) -> pd.DataFrame:
    """Bitget OHLCV: columns [ts(ms), open, high, low, close, volume]"""
    tf = TIMEFRAME_MAP.get(timeframe, "15min")
    inst_id = symbol.replace("/", "")
    url = "https://api.bitget.com/api/spot/v1/market/candles"
    data = _get(url, {"symbol": inst_id, "period": tf, "limit": str(limit)})
    cols = ["ts", "open", "high", "low", "close", "volume"]
    rows = []
    for r in data:
        rows.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df.sort_values("ts", inplace=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)

def ticker_bitget(symbol: str) -> dict:
    """Return last/bid/ask for one symbol."""
    inst_id = symbol.replace("/", "")
    url = "https://api.bitget.com/api/spot/v1/market/ticker"
    data = _get(url, {"symbol": inst_id})
    d = data.get("data", {})
    return {
        "last": float(d.get("last", "nan")),
        "bid": float(d.get("bestBid", "nan")),
        "ask": float(d.get("bestAsk", "nan")),
        "ts": int(d.get("ts", 0)),
    }

