# utiles/bitget.py
# -----------------------------------------------------------
# ccxt wrapper exposing:
#   • make_exchange()  → returns wrapper with .symbols and fetch_ohlcv()
#   • get_exchange()   → alias for backward compatibility
#   • ticker_bitget()  → lightweight one-off public ticker (used by ticks.py)
#   • now_utc_iso()    → small helper (used elsewhere)
# -----------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Dict, Any
import ccxt


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ExchangeWrapper:
    """Wrap a ccxt exchange; expose .symbols (active SPOT/USDT) and fetch_ohlcv()."""
    def __init__(self, ex: ccxt.Exchange):
        self.raw = ex
        try:
            self.raw.load_markets(reload=False)
        except Exception:
            self.raw.load_markets(reload=True)

        syms: List[str] = []
        for m in self.raw.markets.values():
            # SPOT, active, quoted in USDT → “BTC/USDT”, etc.
            if m.get("spot") and m.get("active") and m.get("quote") == "USDT":
                s = m.get("symbol")
                if s:
                    syms.append(s)
        self.symbols = sorted(set(syms))

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 240):
        return self.raw.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def make_exchange(name: str = "bitget") -> _ExchangeWrapper:
    """
    Build an exchange and wrap it. Supports: bitget, binance, bybit.
    Uses public endpoints (API keys optional via env).
    """
    name = (name or "bitget").lower()
    mapping = {"bitget": ccxt.bitget, "binance": ccxt.binance, "bybit": ccxt.bybit}
    if name not in mapping:
        raise ValueError(f"Unsupported exchange '{name}'. Supported: {', '.join(mapping)}")
    cls = mapping[name]

    api_key = os.getenv("API_KEY") or os.getenv(f"{name.upper()}_API_KEY")
    secret  = os.getenv("API_SECRET") or os.getenv(f"{name.upper()}_API_SECRET")
    password = os.getenv("API_PASSWORD") or os.getenv(f"{name.upper()}_API_PASSWORD")

    kwargs = {"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}}
    if api_key and secret:
        kwargs.update({"apiKey": api_key, "secret": secret})
    if password:
        kwargs.update({"password": password})

    ex = cls(kwargs)
    return _ExchangeWrapper(ex)


def get_exchange(name: str = "bitget") -> _ExchangeWrapper:
    """Backward-compat alias."""
    return make_exchange(name)


def ticker_bitget(symbol: str) -> Dict[str, Any]:
    """
    Lightweight public ticker. Returns {'last','bid','ask','ts'} or {'error':...}.
    """
    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})
    try:
        t = ex.fetch_ticker(symbol)
    except Exception as e:
        return {"error": str(e)}
    return {
        "last": t.get("last") or t.get("close"),
        "bid": t.get("bid"),
        "ask": t.get("ask"),
        "ts": t.get("timestamp") or t.get("datetime"),
    }


__all__ = ["make_exchange", "get_exchange", "ticker_bitget", "now_utc_iso", "_ExchangeWrapper"]
