# utiles/bitget.py
# -----------------------------------------------------------
# Tiny wrapper around ccxt to create an exchange instance with:
#  - rate-limit enabled
#  - unified list of spot USDT symbols available as .symbols
#  - safe market loading
#  - optional API keys from env (not required for public data)
# -----------------------------------------------------------

from __future__ import annotations

import os
from typing import List
import ccxt


class _ExchangeWrapper:
    """
    Wraps a ccxt exchange to expose:
      - .raw -> underlying ccxt instance
      - .symbols -> list[str] of available SPOT symbols (e.g., "BTC/USDT")
      - fetch_ohlcv passthrough
    """
    def __init__(self, ex: ccxt.Exchange):
        self.raw = ex
        # Ensure markets are loaded
        try:
            self.raw.load_markets(reload=False)
        except Exception:
            self.raw.load_markets(reload=True)

        # Build a clean SPOT /USDT symbol list
        syms: List[str] = []
        for m in self.raw.markets.values():
            if m.get("spot") and m.get("active") and m.get("quote") == "USDT":
                # ccxt unifies id/symbol; use the human-readable symbol
                sym = m.get("symbol")
                if sym:
                    syms.append(sym)
        # de-dup & sort
        self.symbols = sorted(set(syms))

    # passthrough methods you already use
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 240):
        return self.raw.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def make_exchange(name: str = "bitget") -> _ExchangeWrapper:
    """
    Create and return an exchange wrapped with the helpers above.
    Supported names: "bitget", "binance", "bybit" (extend as needed).
    Public endpoints only; if API keys are present in env, ccxt will use them.
    """
    name = (name or "bitget").lower()

    # Map a few common names to ccxt classes
    mapping = {
        "bitget": ccxt.bitget,
        "binance": ccxt.binance,
        "bybit": ccxt.bybit,
        # Add more if you like
    }
    if name not in mapping:
        raise ValueError(f"Unsupported exchange '{name}'. Supported: {', '.join(mapping)}")

    cls = mapping[name]

    # Optional keys from env (not necessary for public market data)
    # BITGET_* or BINANCE_* etc; we pass whatever is present.
    api_key = os.getenv("API_KEY") or os.getenv(f"{name.upper()}_API_KEY")
    secret  = os.getenv("API_SECRET") or os.getenv(f"{name.upper()}_API_SECRET")
    password = os.getenv("API_PASSWORD") or os.getenv(f"{name.upper()}_API_PASSWORD")  # for Bitget/OKX-like

    kwargs = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "spot",  # ensure spot markets
        }
    }
    if api_key and secret:
        kwargs.update({"apiKey": api_key, "secret": secret})
    if password:
        kwargs.update({"password": password})

    ex = cls(kwargs)
    return _ExchangeWrapper(ex)


# For backward compatibility if some code imports get_exchange
def get_exchange(name: str = "bitget") -> _ExchangeWrapper:
    return make_exchange(name)


__all__ = ["make_exchange", "get_exchange", "_ExchangeWrapper"]
