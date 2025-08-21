# utiles/sentiment.py
from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- helpers -------------------------------------------------

_analyzer = SentimentIntensityAnalyzer()

def _symbol_base(symbol: str) -> str:
    """
    Extract base coin ticker from a pair like 'INJ/USDT' -> 'INJ'.
    """
    s = symbol.upper().strip()
    return s.split("/")[0] if "/" in s else s

def _normalize_score_01(compound: float) -> float:
    """
    VADER compound is in [-1, 1]. Map to [0, 1].
    """
    x = (compound + 1.0) / 2.0
    return max(0.0, min(1.0, x))

# ---- global sentiment: Fear & Greed (free) -------------------

def get_fear_greed() -> Dict[str, Any]:
    """
    Fetch latest Crypto Fear & Greed Index from alternative.me.
    No API key required.
    Returns: {"value": int 0..100, "label": str, "source": "alternative.me"}
    On failure, returns nulls.
    """
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        item = (data.get("data") or [{}])[0]
        value = int(item.get("value")) if item.get("value") is not None else None
        label = item.get("value_classification")
        return {
            "value": value,
            "label": label,
            "source": "alternative.me",
        }
    except Exception:
        return {"value": None, "label": None, "source": "alternative.me"}

# ---- per-crypto sentiment via CryptoPanic (free key) --------

def _get_cryptopanic_key() -> Optional[str]:
    """
    Resolve CRYPTOPANIC_API_KEY from (in order):
    - Streamlit secrets (if running in Streamlit)
    - Environment variable
    """
    try:
        import streamlit as st  # type: ignore
        if "CRYPTOPANIC_API_KEY" in st.secrets:
            return st.secrets["CRYPTOPANIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("CRYPTOPANIC_API_KEY")

def fetch_news_titles_for(symbol: str, max_items: int = 30) -> List[str]:
    """
    Fetch recent crypto news titles for the base ticker using CryptoPanic.
    Requires a free API key; without it, returns [].
    """
    key = _get_cryptopanic_key()
    if not key:
        return []

    base = _symbol_base(symbol)
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": key,
        "currencies": base,   # e.g., BTC
        "kind": "news",
        "public": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results") or []
        titles = []
        for item in results[:max_items]:
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                titles.append(title.strip())
        return titles
    except Exception:
        return []

def score_headlines_vader(titles: List[str]) -> Optional[float]:
    """
    Run VADER on titles and return average normalized score in [0,1].
    Returns None if no titles.
    """
    if not titles:
        return None
    scores = []
    for t in titles:
        comp = _analyzer.polarity_scores(t).get("compound", 0.0)
        scores.append(_normalize_score_01(comp))
    if not scores:
        return None
    return float(sum(scores) / len(scores))

def per_crypto_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Build a compact sentiment dict for a coin, using news headlines only (for now).
    twitter_score and community_score are left as None (placeholders).
    """
    titles = fetch_news_titles_for(symbol, max_items=30)
    news_score = score_headlines_vader(titles)

    overall = news_score  # for now; later we can aggregate multiple sources
    return {
        "twitter_score": None,
        "news_score": news_score,
        "community_score": None,
        "overall": overall,
        "source": "CryptoPanic+VADER" if news_score is not None else "none",
    }
