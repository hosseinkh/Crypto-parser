# utiles/bitget.py
from __future__ import annotations
import time
from typing import Dict, List, Any, Optional, Tuple

try:
    import ccxt
except Exception as e:
    raise RuntimeError("ccxt is required. pip install ccxt") from e


# ---- PUBLIC API (used by other modules) --------------------------------------

def make_exchange(ex_name: Optional[str] = None, rate_limit: bool = True):
    """
    Create and return a ccxt exchange instance (Bitget by default)
    with sane defaults for public data access.
    """
    name = (ex_name or "bitget").lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Unsupported exchange: {name}")

    exchange = getattr(ccxt, name)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv_safe(ex, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
    """
    Robust OHLCV fetch with small retry to survive transient errors/rate limits.
    """
    if limit <= 0:
        limit = 200
    for attempt in range(3):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)  # [ts, o,h,l,c,v]
        except Exception:
            time.sleep(0.5 + attempt * 0.7)
    # last try (if it raises, let it bubble up so we can see the real error)
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def fetch_ticker_safe(ex, symbol: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            return ex.fetch_ticker(symbol)
        except Exception:
            time.sleep(0.5 + attempt * 0.7)
    return ex.fetch_ticker(symbol)


def list_spot_usdt_symbols(ex) -> List[str]:
    """
    Return a clean list of spot symbols ending with /USDT and tradable.
    """
    out = []
    for s, m in ex.markets.items():
        try:
            if m.get("spot") and s.endswith("/USDT") and not m.get("inactive", False):
                out.append(s)
        except Exception:
            continue
    out.sort()
    return out
