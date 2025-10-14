# utiles/bitget.py
from __future__ import annotations
import time
from typing import Dict, List, Any, Optional

try:
    import ccxt
except Exception as e:
    raise RuntimeError("ccxt is required. pip install ccxt") from e


# ---- helpers (internal) ------------------------------------------------------

def _timeframe_ms(tf: str) -> int:
    table = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if tf not in table:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return table[tf]


def _floor_ts_to_tf(ts_ms: int, tf: str) -> int:
    step = _timeframe_ms(tf)
    return (ts_ms // step) * step


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
        "enableRateLimit": True if rate_limit else False,
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv_safe(
    ex,
    symbol: str,
    timeframe: str,
    limit: int,
    since: Optional[int] = None,
    *,
    closed_only: bool = True,
    now_ts: Optional[int] = None,
) -> List[List[Any]]:
    """
    Robust OHLCV fetch with small retry to survive transient errors/rate limits.
    Adds *closed_only* and *now_ts* so callers can enforce using only CLOSED candles
    and align multiple timeframes to the SAME UTC close boundary.

    Args:
        ex: ccxt exchange instance
        symbol: e.g. 'SOL/USDT'
        timeframe: e.g. '15m', '1h', '4h'
        limit: number of candles to fetch (<=0 coerced to 200)
        since: optional starting timestamp (ms)
        closed_only: if True (default), drop the currently forming candle
        now_ts: optional "clock" (ms) to determine the latest closed boundary consistently
                across different timeframes in the same snapshot

    Returns:
        List of [timestamp, open, high, low, close, volume]
    """
    if limit <= 0:
        limit = 200

    last_err = None
    for attempt in range(3):
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            if not rows:
                return rows

            if closed_only:
                # Determine the latest closed-candle boundary using a shared clock.
                now_ms = int(now_ts if now_ts is not None else time.time() * 1000)
                last_close_boundary = _floor_ts_to_tf(now_ms, timeframe)
                # The candle whose timestamp == last_close_boundary is the still-forming candle.
                # If our last returned candle has that timestamp (or greater), drop it.
                if rows and rows[-1][0] >= last_close_boundary:
                    rows = rows[:-1]

            return rows
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.7)

    # final attempt: bubble the real error if it still fails
    if last_err:
        raise last_err
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)


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
