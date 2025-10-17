# utiles/bitget.py
from __future__ import annotations
import time
import math
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
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "12h": 43_200_000,
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


def _normalize_row(row: Any) -> Optional[List[float]]:
    """
    Normalize a single OHLCV row to [ts, open, high, low, close, volume].
    Returns None if the row is malformed.
    """
    if isinstance(row, (list, tuple)) and len(row) >= 5:
        ts = row[0]
        o = row[1] if len(row) > 1 else None
        h = row[2] if len(row) > 2 else None
        l = row[3] if len(row) > 3 else None
        c = row[4] if len(row) > 4 else None
        v = row[5] if len(row) > 5 else 0.0
    elif isinstance(row, dict):
        ts = row.get("timestamp") or row.get(0)
        o = row.get("open")      or row.get(1)
        h = row.get("high")      or row.get(2)
        l = row.get("low")       or row.get(3)
        c = row.get("close")     or row.get(4)
        v = row.get("volume")    or row.get(5) or 0.0
    else:
        return None

    vals = [ts, o, h, l, c, v]
    if any(x is None for x in vals):
        return None

    try:
        ts = int(ts)
        o = float(o); h = float(h); l = float(l); c = float(c); v = float(v)
    except Exception:
        return None

    # Drop NaN/Inf
    for x in (o, h, l, c, v):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None

    # Ensure H/L envelope valid in case vendor flipped them
    hi = max(h, l)
    lo = min(h, l)
    h, l = hi, lo

    return [ts, o, h, l, c, v]


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
    Robust OHLCV fetch that normalizes rows to:
        [timestamp(ms), open, high, low, close, volume]
    - Retries on transient errors
    - Sorts ascending
    - Drops malformed rows / NaNs
    - Fixes any flipped high/low
    - Optionally drops the still-forming last candle using a shared clock
    """
    if limit <= 0:
        limit = 200

    last_err = None
    for attempt in range(3):
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            if not raw:
                return []

            # Normalize and filter
            out: List[List[float]] = []
            for row in raw:
                norm = _normalize_row(row)
                if norm is not None:
                    out.append(norm)

            if not out:
                return []

            # Sort ascending by timestamp (some exchanges return descending)
            out.sort(key=lambda r: r[0])

            # Drop last row if not closed (use shared now_ts for cross-TF alignment)
            if closed_only and len(out) > 0:
                tf_ms = _timeframe_ms(timeframe)
                # If now_ts not provided, fall back to wall clock
                clock = int(now_ts if now_ts is not None else time.time() * 1000)
                last_closed_boundary = _floor_ts_to_tf(clock, timeframe)
                # If the last bar's timestamp equals the current interval start, it's still forming.
                if out[-1][0] >= last_closed_boundary:
                    out = out[:-1]

            # Trim to limit (after dropping open bar)
            if limit and len(out) > limit:
                out = out[-limit:]

            return out
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.7)

    # Final fallback: raise the last error if we couldn't return
    if last_err:
        raise last_err

    # As a last resort (shouldn't reach here)
    return []


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
