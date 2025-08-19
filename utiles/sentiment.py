from __future__ import annotations
import requests

def get_fear_greed(default: int = 50) -> int:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        v = int(r.json()["data"][0]["value"])
        return v
    except Exception:
        return default

def get_coingecko_global():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global")
        j = r.json()
        btc_dom = float(j["data"]["market_cap_percentage"]["btc"])
        total_mcap = float(j["data"]["total_market_cap"]["usd"])
        total_24h = float(j["data"]["market_cap_change_percentage_24h_usd"]) / 100.0
        return {"btc_dominance_pct": btc_dom, "total_mcap_24h_pct": total_24h}
    except Exception:
        return {"btc_dominance_pct": None, "total_mcap_24h_pct": None}

def derive_risk_regime(fng: int | None, total3_24h_pct: float | None, btc_dominance_pct: float | None) -> str:
    try:
        if fng is not None and total3_24h_pct is not None and btc_dominance_pct is not None:
            if fng >= 55 and total3_24h_pct >= 0.01:
                return "risk-on"
            if fng <= 35 or total3_24h_pct <= -0.01:
                return "risk-off"
    except Exception:
        pass
    return "neutral"
