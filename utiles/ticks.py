# utiles/ticks.py
from __future__ import annotations

from typing import Any, Dict, List
from utiles.bitget import make_exchange, now_utc_iso

"""
Attach a 'tick' block to each symbol in your packed snapshot just before you
write it to JSON. This captures the *live* execution price at the time you
click 'Download snapshot', while your indicators still rely on CLOSED candles.
"""

def _compute_spread_bps(bid: float, ask: float, last: float) -> float | None:
    try:
        if bid and ask and last:
            return round((ask - bid) / last * 10000.0, 3)
    except Exception:
        pass
    return None

def augment_with_ticks(packed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds to the top-level:
      - snapshot_time_utc (ISO-8601, Z)
    Adds per-symbol:
      - tick: { last_trade_price, best_bid, best_ask, spread_bps }
    """
    try:
        # Your snapshot likely uses 'symbols' (but we also support 'data')
        symbols: List[Dict[str, Any]] = packed.get("symbols") or packed.get("data") or []
        # Stamp snapshot time once
        packed.setdefault("snapshot_time_utc", now_utc_iso())

        ex = make_exchange()  # ccxt Bitget client

        for sym_block in symbols:
            symbol = sym_block.get("symbol") or sym_block.get("name") or sym_block.get("pair")
            if not symbol:
                continue
            try:
                tkr = ex.fetch_ticker(symbol)
                bid = float(tkr.get("bid") or 0.0)
                ask = float(tkr.get("ask") or 0.0)
                last = float(tkr.get("last") or tkr.get("close") or 0.0)
                sym_block["tick"] = {
                    "last_trade_price": last,
                    "best_bid": bid,
                    "best_ask": ask,
                    "spread_bps": _compute_spread_bps(bid, ask, last),
                }
            except Exception as e:
                # Keep the snapshot robust even if one ticker fails
                sym_block["tick"] = {"error": str(e)}
    except Exception as e:
        packed.setdefault("_augment_error", str(e))
    return packed

def augment_many_with_ticks(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batch version for chunked snapshot export (list of packed dicts).
    """
    try:
        ex = make_exchange()
        out = []
        for p in payloads:
            # stamp time per payload to be explicit (optional: could stamp once)
            p.setdefault("snapshot_time_utc", now_utc_iso())
            symbols: List[Dict[str, Any]] = p.get("symbols") or p.get("data") or []
            for sym_block in symbols:
                symbol = sym_block.get("symbol") or sym_block.get("name") or sym_block.get("pair")
                if not symbol:
                    continue
                try:
                    tkr = ex.fetch_ticker(symbol)
                    bid = float(tkr.get("bid") or 0.0)
                    ask = float(tkr.get("ask") or 0.0)
                    last = float(tkr.get("last") or tkr.get("close") or 0.0)
                    sym_block["tick"] = {
                        "last_trade_price": last,
                        "best_bid": bid,
                        "best_ask": ask,
                        "spread_bps": _compute_spread_bps(bid, ask, last),
                    }
                except Exception as e:
                    sym_block["tick"] = {"error": str(e)}
            out.append(p)
        return out
    except Exception as e:
        # Fall back to returning original payloads with an error note
        for p in payloads:
            p.setdefault("_augment_error", str(e))
        return payloads
