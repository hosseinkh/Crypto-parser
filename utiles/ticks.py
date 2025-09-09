# utiles/ticks.py
from __future__ import annotations
from typing import Any, Dict, List
from utiles.bitget import ticker_bitget, now_utc_iso

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
      tick = { last_trade_price, best_bid, best_ask, spread_bps, tick_ts }
    Also stamps top-level 'snapshot_time_utc'.
    """
    try:
        symbols = packed.get("symbols") or packed.get("data") or []
        packed["snapshot_time_utc"] = now_utc_iso()  # stamp once

        for sym_block in symbols:
            symbol = sym_block.get("symbol") or sym_block.get("name") or sym_block.get("pair")
            if not symbol:
                continue
            try:
                t = ticker_bitget(symbol)
                last = float(t.get("last", float("nan")))
                bid = float(t.get("bid", float("nan")))
                ask = float(t.get("ask", float("nan")))
                sym_block["tick"] = {
                    "last_trade_price": last,
                    "best_bid": bid,
                    "best_ask": ask,
                    "spread_bps": _spread_bps(bid, ask, last),
                    "tick_ts": int(t.get("ts", 0)),
                }
            except Exception as e:
                sym_block["tick"] = {"error": str(e)}
    except Exception as e:
        packed.setdefault("_augment_error", str(e))
    return packed

def augment_many_with_ticks(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach 'tick' to each symbol in a *list* of packed snapshots (chunked export).
    """
    try:
        for p in payloads:
            p["snapshot_time_utc"] = now_utc_iso()
            symbols = p.get("symbols") or p.get("data") or []
            for sym_block in symbols:
                symbol = sym_block.get("symbol") or sym_block.get("name") or sym_block.get("pair")
                if not symbol:
                    continue
                try:
                    t = ticker_bitget(symbol)
                    last = float(t.get("last", float("nan")))
                    bid = float(t.get("bid", float("nan")))
                    ask = float(t.get("ask", float("nan")))
                    sym_block["tick"] = {
                        "last_trade_price": last,
                        "best_bid": bid,
                        "best_ask": ask,
                        "spread_bps": _spread_bps(bid, ask, last),
                        "tick_ts": int(t.get("ts", 0)),
                    }
                except Exception as e:
                    sym_block["tick"] = {"error": str(e)}
        return payloads
    except Exception as e:
        for p in payloads:
            p.setdefault("_augment_error", str(e))
        return payloads
