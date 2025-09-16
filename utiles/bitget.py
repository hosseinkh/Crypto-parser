# utiles/bitget.py
# -----------------------------------------------------------
# ccxt wrapper exposing:
#  - .symbols (active SPOT /USDT)
#  - fetch_ohlcv passthrough
#  - make_exchange() constructor
# -----------------------------------------------------------

from __future__ import annotations

import os
from typing import List
import ccxt


class _ExchangeWrapper:
    """Wraps a ccxt exchange, exposing .symbols and fetch_ohlcv()."""

    def __init__(self, ex: ccxt.Exchange):
        self.raw = ex
        # Load markets safely
        try:
            self.raw.load_markets(reload=False)
        except Exception:
            self.raw.load_markets(reload=True)

        # Build clean SPOT/USDT symbol list
        syms: List[str] = []
        for m in self.raw.markets.values():
            if m.get("spot") and m.get("active") and m.get("quote") == "USDT":
                sym = m.get("symbol")
                if sym:
                    syms.append(sym)
        self.symbols = sorted(set(syms))

    # passthrough used by the app
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 240):
        return self.raw.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def make_exchange(name: str = "bitget") -> _ExchangeWrapper:
    """
    Create and return an exchange wrapped with the helpers above.
    Supported names: "bitget", "binance", "bybit".
    Uses public endpoints; if API keys exist in env, ccxt will use them.
    """
    name = (name or "bitget").lower()
    mapping = {
        "bitget": ccxt.bitget,
        "binance": ccxt.binance,
        "bybit": ccxt.bybit,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported exchange '{name}'. Supported: {', '.join(mapping)}")

    cls = mapping[name]

    # Optional keys (not required)
    api_key = os.getenv("API_KEY") or os.getenv(f"{name.upper()}_API_KEY")
    secret = os.getenv("API_SECRET") or os.getenv(f"{name.upper()}_API_SECRET")
    password = os.getenv("API_PASSWORD") or os.getenv(f"{name.upper()}_API_PASSWORD")

    kwargs = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    }
    if api_key and secret:
        kwargs.update({"apiKey": api_key, "secret": secret})
    if password:
        kwargs.update({"password": password})

    ex = cls(kwargs)
    return _ExchangeWrapper(ex)


# Backward-compat alias
def get_exchange(name: str = "bitget") -> _ExchangeWrapper:
    return make_exchange(name)


__all__ = ["make_exchange", "get_exchange", "_ExchangeWrapper"]
