# utiles/sentiment.py
# --------------------------------------------------------------------
# Free-news sentiment via Cryptopanic + VADER.
# - Uses CRYPTOPANIC_KEY from environment or Streamlit secrets.
# - Caches results briefly to avoid rate-limits.
# - Outputs {"score": [-1..1], "label": negative|neutral|positive}
# --------------------------------------------------------------------

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import requests

try:
    # VADER is lightweight and free: pip install vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception as e:
    raise RuntimeError("vaderSentiment is required. Install with: pip install vaderSentiment") from e


# --- simple in-memory cache for 2 minutes per symbol --- #
_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}
_CACHE_TTL = 120.0  # seconds


def _get_api_key() -> str:
    # Try Streamlit secrets first, fallback to env
    key = os.environ.get("CRYPTOPANIC_KEY")
    if key:
        return key
    # Optional: you can add streamlit import to read secrets if desired:
    try:
        import streamlit as st  # type: ignore
        if "CRYPTOPANIC_KEY" in st.secrets:
            return st.secrets["CRYPTOPANIC_KEY"]  # type: ignore
    except Exception:
        pass
    return ""  # public endpoint still works with limits, but key recommended


def _fetch_cryptopanic_headlines(symbol: str, limit: int = 30) -> List[str]:
    """
    Fetch recent headlines from Cryptopanic tagged for the coin.
    Returns list of titles (strings). If API key missing, still tries public endpoint.
    """
    key = _get_api_key()
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "currencies": symbol.split("/")[0].replace("USDT", ""),
        "kind": "news",
        "public": "true",
        "regions": "en",
        "filter": "hot",
        "limit": min(limit, 50),
    }
    if key:
        params["auth_token"] = key

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        titles = []
        for item in results:
            title = item.get("title", "")
            if title:
                titles.append(title)
        return titles
    except Exception:
        return []


def _vader_score_texts(texts: List[str]) -> float:
    """
    Compute average compound VADER score across provided texts.
    Returns float in [-1, 1].
    """
    if not texts:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    s = 0.0
    n = 0
    for t in texts:
        if not t or not isinstance(t, str):
            continue
        score = analyzer.polarity_scores(t).get("compound", 0.0)
        s += score
        n += 1
    return 0.0 if n == 0 else max(min(s / n, 1.0), -1.0)


def _label_from_score(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def get_sentiment_for_symbol(symbol: str) -> Dict[str, float | str]:
    """
    Public API: returns {"score": float in [-1,1], "label": str}
    Caches for short period to avoid API spam.
    """
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and (now - cached[0] <= _CACHE_TTL):
        sc = cached[1]["score"]
        return {"score": sc, "label": _label_from_score(sc)}

    titles = _fetch_cryptopanic_headlines(symbol, limit=40)
    score = _vader_score_texts(titles)
    _CACHE[symbol] = (now, {"score": score})
    return {"score": score, "label": _label_from_score(score)}
