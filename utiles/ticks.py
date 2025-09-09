# utiles/ticks.py
from __future__ import annotations

from typing import Any, Dict, List
from utiles.bitget import make_exchange, now_utc_iso

def _spread_bps(bid: float, ask: float, last: float) -> float | None:
    try:
        if bid and ask and last:
            return round((ask - bid) / last * 10000.0, 3)
    except Exception:
        pass
    return None

def augment_with_ticks(packed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach a 'tick' block to each symbol in *one* packed snapshot:
      tick = { last_trade_price, best_bid, best_ask, spread_bps }
    Also stamps top-level 'snapshot_time_utc' (ISO-8601, Z).
    """
    try:
        symbols = packed.get("symbols") or packed.get("data") or []
        packed.setdefault("snapshot_time_utc", now_utc_iso())

        ex = make_exchange()
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
                    "spread_bps": _spread_bps(bid, ask, last),
                }
            except Exception as e:
                sym_block["tick"] = {"error": str(e)}
    except Exception as e:
        packed.setdefault("_augment_error", str(e))
    return packed

def augment_many_with_ticks(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach a 'tick' block to each symbol in a *list* of packed snapshots
    (used for chunked export).
    """
    try:
        ex = make_exchange()
        out: List[Dict[str, Any]] = []
        for p in payloads:
            p.setdefault("snapshot_time_utc", now_utc_iso())
            symbols = p.get("symbols") or p.get("data") or []
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
                        "spread_bps": _spread_bps(bid, ask, last),
                    }
                except Exception as e:
                    sym_block["tick"] = {"error": str(e)}
            out.append(p)
        return out
    except Exception as e:
        for p in payloads:
            p.setdefault("_augment_error", str(e))
        return payloads

