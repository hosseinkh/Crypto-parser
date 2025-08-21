# utiles/sentiment.py
from __future__ import annotations

import os
import requests
from typing import Dict, List, Optional
import streamlit as st

# ---------- helpers ----------
def _secret(name: str) -> Optional[str]:
    # Try Streamlit secrets first, then environment (for local dev)
    try:
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name)

def _summarize_score(score: float) -> str:
    if score > 0.25:
        return "bullish"
    if score < -0.25:
        return "bearish"
    return "neutral/mixed"

# ---------- CryptoPanic per-symbol sentiment ----------
def fetch_cryptopanic_per_symbol(bases: List[str]) -> Optional[Dict[str, dict]]:
    """
    Builds a per-symbol sentiment dict using CryptoPanic headlines.
    We approximate a score with (positive_votes - negative_votes) / max(1, total_votes).
    """
    key = _secret("CRYPTOPANIC_KEY")
    if not key:
        return None

    out: Dict[str, dict] = {}
    for base in bases:
        try:
            # CryptoPanic requires 'auth_token' query param
            r = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={
                    "auth_token": key,
                    "currencies": base,     # e.g. "BTC"
                    "kind": "news",
                    "public": "true",
                    "page": 1,
                },
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()

            pos = neg = 0
            for item in data.get("results", []):
                votes = item.get("votes") or item.get("vote") or {}
                pos += int(votes.get("positive", 0))
                neg += int(votes.get("negative", 0))

            total = max(1, pos + neg)
            score = (pos - neg) / total
            out[f"{base}/USDT"] = {
                "score": round(score, 3),
                "summary": _summarize_score(score),
                "articles_count": len(data.get("results", [])),
                "source": "cryptopanic",
            }
        except Exception as e:
            out[f"{base}/USDT"] = {
                "score": None,
                "summary": f"error: {e}",
                "source": "cryptopanic",
            }
    return out

# ---------- NewsAPI general market sentiment ----------
def fetch_newsapi_general() -> Optional[dict]:
    """
    Pulls recent crypto headlines from NewsAPI and produces a coarse sentiment.
    (Keyword heuristic on titles; simple but works as a signal.)
    """
    key = _secret("NEWSAPI_KEY")
    if not key:
        return None

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

        titles = [a.get("title", "") or "" for a in data.get("articles", [])]

        # very lightweight keywords heuristic
        bulls = ("surge", "rally", "breakout", "record", "up", "bull", "partnership", "approval", "ETF")
        bears = ("falls", "down", "drop", "hack", "exploit", "lawsuit", "ban", "bear", "liquidation")

        pos = sum(1 for t in titles if any(w in t.lower() for w in bulls))
        neg = sum(1 for t in titles if any(w in t.lower() for w in bears))
        total = max(1, pos + neg)
        score = (pos - neg) / total

        return {
            "score": round(score, 3),
            "summary": _summarize_score(score),
            "sample_titles": titles[:5],
            "articles_considered": len(titles),
            "source": "newsapi",
        }
    except Exception as e:
        return {"score": None, "summary": f"error: {e}", "source": "newsapi"}

# ---------- Bundle ----------
def build_sentiment_bundle(bases: List[str]) -> dict:
    """
    Compose both general and per-symbol sentiment blocks.
    """
    general = fetch_newsapi_general()
    per_symbol = fetch_cryptopanic_per_symbol(bases)
    return {"general": general, "per_symbol": per_symbol}
