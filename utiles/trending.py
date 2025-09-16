# utiles/trending.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .bitget import make_exchange, list_spot_usdt_symbols, fetch_ohlcv_safe
from .snapshot import rsi as _rsi, volume_zscore as _vz
from .sentiment import get_sentiment_for_symbol

@dataclass
class TrendScanParams:
    exchange_name: str = "bitget"
    timeframe: str = "15m"
    limit: int = 150
    top_n: int = 10
    min_vol_z: float = 0.8
    rsi_range: Tuple[float, float] = (45.0, 68.0)   # avoid extremes
    include_sentiment: bool = True                  # ✅ now enabled by default

def _tech_row(symbol: str, closes: List[float], vols: List[float]) -> Dict[str, Any]:
    rsi = _rsi(closes, 14)
    vz = _vz([[0,0,0,0,c,v] for c, v in zip(closes, vols)], 50)
    score = 0.0
    reasons = []
    # RSI contribution
    if not (rsi != rsi):  # not NaN
        if 45 <= rsi <= 68:
            score += 1.0; reasons.append(f"RSI15m {rsi:.1f} in [45,68]")
        elif rsi < 45:
            reasons.append(f"RSI15m {rsi:.1f} low (watch bounce)")
        else:
            reasons.append(f"RSI15m {rsi:.1f} high (watch pullback)")
    # Volume z-score contribution
    if not (vz != vz):
        score += max(0.0, min(1.0, (vz - 0.5)))     # gently reward high vz
        reasons.append(f"vol_z {vz:.2f}")
    return {"score": score, "reasons": reasons, "rsi": rsi, "vol_z": vz}

def scan_trending(params: TrendScanParams) -> List[Dict[str, Any]]:
    ex = make_exchange(params.exchange_name)
    syms = list_spot_usdt_symbols(ex)
    rows: List[Dict[str, Any]] = []

    for s in syms:
        try:
            ohlcv = fetch_ohlcv_safe(ex, s, params.timeframe, params.limit)
            closes = [x[4] for x in ohlcv]
            vols   = [x[5] for x in ohlcv]
            tech = _tech_row(s, closes, vols)
            if tech["vol_z"] >= params.min_vol_z:
                row = {
                    "symbol": s,
                    "score": float(tech["score"]),
                    "reasons": tech["reasons"],
                    "rsi": tech["rsi"],
                    "vol_z": tech["vol_z"],
                }
                # ✅ Optional sentiment contribution
                if params.include_sentiment:
                    sent_score, sent_details = get_sentiment_for_symbol(s)
                    # map [-1..+1] to [0..1] bump (~0..0.6)
                    bump = max(0.0, sent_score + 0.2) * 0.6
                    row["score"] += bump
                    row["sentiment_score"] = sent_score
                    row["sentiment_details"] = sent_details
                    row["reasons"].append(f"sent {sent_score:+.2f}")
                rows.append(row)
        except Exception:
            continue

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[: params.top_n]

def explain_trending_row(row: Dict[str, Any]) -> str:
    parts = [f"{row['symbol']}"]
    if "vol_z" in row and row["vol_z"] == row["vol_z"]:
        parts.append(f"vol_z={row['vol_z']:.2f}")
    if "rsi" in row and row["rsi"] == row["rsi"]:
        parts.append(f"RSI={row['rsi']:.1f}")
    if "sentiment_score" in row:
        parts.append(f"sent={row['sentiment_score']:+.2f}")
    if row.get("reasons"):
        parts.append(" | " + "; ".join(row["reasons"]))
    return " ".join(parts)
