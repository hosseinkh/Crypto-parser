# utiles/sentiment.py
from __future__ import annotations
import os, time, json
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

import urllib.request
import urllib.parse

# -----------------------------
# Design:
#  - Returns (score, details) where score in [-1, +1]
#  - Uses multiple providers when available; otherwise returns neutral (0.0)
#  - Safe fallbacks: never raises to callers
#  - Lightweight caching to avoid hammering APIs
# -----------------------------

# --- Helpers -------------------------------------------------

def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 8) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _symbol_to_slug(symbol: str) -> str:
    # Best-effort symbol → slug mapping (override via env if needed)
    base = symbol.split("/")[0].lower()
    # quick overrides (extend as needed)
    overrides = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "ada": "cardano",
        "dot": "polkadot",
        "ltc": "litecoin",
        "xrp": "ripple",
        "xlm": "stellar",
        "link": "chainlink",
        "sol": "solana",
        "trx": "tron",
        "bnb": "binancecoin",
        "jup": "jupiter-exchange",
        "stx": "stacks",
        "tia": "celestia",
        "rif": "rif-token",
        "chess": "tranchess",
        "syrup": "pancake-bunny",
        "atm": "atletico-madrid-fan-token",
        "sxt": "swyftx",   # placeholder; adjust if needed
        "blast": "blast",  # placeholder
    }
    return overrides.get(base, base)

# --- Providers -----------------------------------------------

def _fear_greed_market_score() -> Tuple[float, Dict[str, Any]]:
    """
    alternative.me Fear & Greed index (0..100). Map to [-0.5, +0.5].
    """
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    data = _http_get(url)
    if not data or "data" not in data or not data["data"]:
        return 0.0, {"provider": "fear_greed", "status": "unavailable"}
    try:
        v = float(data["data"][0]["value"])
    except Exception:
        return 0.0, {"provider": "fear_greed", "status": "bad_data"}
    # 0 bearish .. 100 bullish → shift to [-0.5..+0.5]
    score = (v / 100.0) - 0.5
    return score, {"provider": "fear_greed", "value": v, "mapped_score": score}

def _cryptopanic_asset_score(symbol: str) -> Tuple[float, Dict[str, Any]]:
    """
    CryptoPanic headlines sentiment (requires CRYPTOPANIC_TOKEN).
    Score heuristic in [-0.5, +0.5].
    """
    token = os.getenv("CRYPTOPANIC_TOKEN")
    if not token:
        return 0.0, {"provider": "cryptopanic", "status": "disabled"}
    # Query by symbol (CryptoPanic supports 'currencies' param with comma list)
    qsym = symbol.split("/")[0]
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={urllib.parse.quote(token)}&currencies={urllib.parse.quote(qsym)}&public=true"
    data = _http_get(url)
    if not data or "results" not in data:
        return 0.0, {"provider": "cryptopanic", "status": "unavailable"}
    pos = neg = 0
    for r in data["results"][:50]:
        s = (r.get("vote", {}) or {}).get("value") or 0
        # crude heuristic using tags if present
        tags = r.get("tags") or []
        title = (r.get("title") or "").lower()
        if any(t in tags for t in ("positive", "bullish")) or "surge" in title or "rally" in title:
            pos += 1
        if any(t in tags for t in ("negative", "bearish")) or "hack" in title or "dump" in title:
            neg += 1
    total = max(1, pos + neg)
    raw = (pos - neg) / total  # [-1..+1]
    score = raw * 0.5          # cap to [-0.5..+0.5]
    return score, {"provider": "cryptopanic", "pos": pos, "neg": neg, "mapped_score": score}

def _coingecko_asset_score(symbol: str) -> Tuple[float, Dict[str, Any]]:
    """
    Coingecko 'sentiment_votes_up_percentage' if available for asset; map to [-0.5,+0.5].
    No key required; public endpoint with rate limits.
    """
    slug = _symbol_to_slug(symbol)
    url = f"https://api.coingecko.com/api/v3/coins/{urllib.parse.quote(slug)}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=false&sparkline=false"
    data = _http_get(url)
    if not data:
        return 0.0, {"provider": "coingecko", "status": "unavailable"}
    try:
        up = (data.get("sentiment_votes_up_percentage") or 50.0)
        down = (data.get("sentiment_votes_down_percentage") or 50.0)
        total = max(1.0, up + down)
        raw = (up - down) / total  # [-1..+1] approx
        score = (raw) * 0.5
        return score, {"provider": "coingecko", "up%": up, "down%": down, "mapped_score": score}
    except Exception:
        return 0.0, {"provider": "coingecko", "status": "bad_data"}

# --- Public API ----------------------------------------------

@lru_cache(maxsize=4096)
def _get_market_sent_cached(ts_bucket: int) -> Tuple[float, Dict[str, Any]]:
    # Cache bucket ~300s
    return _fear_greed_market_score()

@lru_cache(maxsize=4096)
def _get_asset_sent_cached(symbol: str, ts_bucket: int) -> Tuple[float, Dict[str, Any]]:
    # Try asset-level sources; combine when available
    s1, d1 = _cryptopanic_asset_score(symbol)
    s2, d2 = _coingecko_asset_score(symbol)
    # Weighted combine (favor asset over market)
    # If both missing → 0
    if d1.get("status") == "disabled" and d2.get("status") in ("unavailable", "bad_data"):
        return 0.0, {"provider": "asset_combo", "components": [d1, d2]}
    # Average of available sources
    parts = []
    if d1.get("status") not in ("disabled", "unavailable", "bad_data"):
        parts.append(s1)
    if d2.get("status") not in ("unavailable", "bad_data"):
        parts.append(s2)
    score = sum(parts) / len(parts) if parts else 0.0
    return score, {"provider": "asset_combo", "components": [d1, d2], "mapped_score": score}

def get_sentiment_for_symbol(symbol: str) -> Tuple[float, Dict[str, Any]]:
    """
    Return (score, details) where score ∈ [-1, +1].
    details includes 'market', 'asset', and final 'combined'.
    """
    # time bucket for cache (300s)
    bucket = int(time.time() // 300)
    try:
        m_sc, m_det = _get_market_sent_cached(bucket)
    except Exception:
        m_sc, m_det = 0.0, {"provider": "fear_greed", "status": "error"}

    try:
        a_sc, a_det = _get_asset_sent_cached(symbol, bucket)
    except Exception:
        a_sc, a_det = 0.0, {"provider": "asset_combo", "status": "error"}

    # Combine: 40% market + 60% asset
    combined = 0.4 * m_sc + 0.6 * a_sc
    details = {
        "market": m_det,
        "asset": a_det,
        "combined_score": combined
    }
    # Clamp
    combined = max(-1.0, min(1.0, combined))
    return combined, details
