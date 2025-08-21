# utiles/sentiment.py
from __future__ import annotations

import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

import requests

# VADER for headline text sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: Streamlit secrets if available (safe to import in Cloud)
try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

_ANALYZER = SentimentIntensityAnalyzer()


# ------------------------
# helpers
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _secret(name: str) -> Optional[str]:
    if _HAS_ST:
        try:
            v = st.secrets.get(name)
            if v:
                return str(v)
        except Exception:
            pass
    return os.environ.get(name)

def _summary_from_score(score: float) -> str:
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral/mixed"

def _score_titles_vader(titles: List[str]) -> Optional[float]:
    if not titles:
        return None
    vals = []
    for t in titles:
        comp = _ANALYZER.polarity_scores(t or "").get("compound", 0.0)  # -1..+1
        vals.append(comp)
    if not vals:
        return None
    return float(sum(vals) / len(vals))  # still -1..+1


# ------------------------
# CryptoPanic (per-coin)
# ------------------------
def _cryptopanic_titles_for_base(base: str, max_items: int = 30) -> List[dict]:
    """Return list of {title, url} for a base coin (e.g., 'INJ')."""
    key = _secret("CRYPTOPANIC_KEY")
    if not key:
        return []

    try:
        r = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={
                "auth_token": key,
                "currencies": base,   # e.g., BTC
                "kind": "news",
                "public": "true",
                "page": 1,
            },
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for item in (data.get("results") or [])[:max_items]:
            title = (item.get("title") or "").strip()
            url = item.get("url") or item.get("source", {}).get("url")
            if title:
                out.append({"title": title, "url": url})
        return out
    except Exception:
        return []


def fetch_cryptopanic_per_symbol(bases: List[str]) -> Dict[str, dict]:
    """
    Build a per-symbol sentiment dict by scoring CryptoPanic headlines with VADER.
    Returns keys as BASE (no /USDT) for compactness; your packer can map as needed.
    """
    results: Dict[str, dict] = {}
    fetched_at = _now_utc_iso()

    for base in bases:
        samples = _cryptopanic_titles_for_base(base, max_items=30)
        titles = [s["title"] for s in samples]
        score = _score_titles_vader(titles)  # -1..+1 or None
        if score is None:
            block = {
                "score": None,
                "summary": "no data",
                "articles_count": 0,
                "examples": [],
                "confidence": 0.0,
                "source": "cryptopanic",
                "fetched_at_utc": fetched_at,
                "window": "recent",
            }
        else:
            count = len(titles)
            # simple confidence: more articles -> more confidence (cap at 1)
            confidence = min(1.0, count / 20.0)
            block = {
                "score": round(score, 3),
                "summary": _summary_from_score(score),
                "articles_count": count,
                "examples": samples[:5],  # first 5 titles/urls for audit
                "confidence": round(confidence, 2),
                "source": "cryptopanic",
                "fetched_at_utc": fetched_at,
                "window": "recent",
            }
        results[base] = block
    return results


# ------------------------
# NewsAPI (general market)
# ------------------------
def fetch_newsapi_general() -> dict:
    """
    Pull recent crypto headlines from NewsAPI and produce a coarse general sentiment.
    Uses keyword heuristic + VADER average for better robustness.
    """
    key = _secret("NEWSAPI_KEY")
    fetched_at = _now_utc_iso()
    if not key:
        return {
            "score": None,
            "summary": "no key",
            "articles_considered": 0,
            "examples": [],
            "source": "newsapi",
            "fetched_at_utc": fetched_at,
            "window": "recent",
        }

    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "(crypto OR cryptocurrency OR bitcoin OR ethereum)",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": key,
            },
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles") or []
        titles = [a.get("title") or "" for a in articles]
        # Combine VADER average with a light keyword heuristic (just as a stabilizer)
        vader = _score_titles_vader(titles) or 0.0

        bulls = ("surge", "rally", "breakout", "record", "up", "bull", "partnership", "approval", "etf")
        bears = ("falls", "down", "drop", "hack", "exploit", "lawsuit", "ban", "bear", "liquidation")
        pos = sum(1 for t in titles if any(w in t.lower() for w in bulls))
        neg = sum(1 for t in titles if any(w in t.lower() for w in bears))
        total = max(1, pos + neg)
        kw_score = (pos - neg) / total  # -1..+1

        score = (vader * 0.7) + (kw_score * 0.3)

        return {
            "score": round(score, 3),
            "summary": _summary_from_score(score),
            "articles_considered": len(titles),
            "examples": [{"title": t} for t in titles[:5]],
            "source": "newsapi",
            "fetched_at_utc": fetched_at,
            "window": "recent",
        }
    except Exception as e:
        return {
            "score": None,
            "summary": f"error: {e}",
            "articles_considered": 0,
            "examples": [],
            "source": "newsapi",
            "fetched_at_utc": fetched_at,
            "window": "recent",
        }


# ------------------------
# Public: bundle builder
# ------------------------
def build_sentiment_bundle(bases: List[str]) -> dict:
    """
    Compose both general and per-symbol sentiment blocks.
    Returns:
    {
      "general": {...},               # newsapi-based, with examples
      "per_symbol": { "INJ": {...}, "TRX": {...}, ... }
    }
    """
    general = fetch_newsapi_general()
    per_symbol = fetch_cryptopanic_per_symbol(bases)
    return {"general": general, "per_symbol": per_symbol}
