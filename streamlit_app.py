# streamlit_app.py
# -----------------------------------------------------------
# Always-include favorites (incl. GALA, XLM) + trend scan with reasons
# -----------------------------------------------------------

from __future__ import annotations
import os, sys, importlib
from typing import List
import streamlit as st
import pandas as pd
from dataclasses import dataclass

# --- Package setup ---
ROOT_DIR = os.path.dirname(__file__)
UTIL_DIR = os.path.join(ROOT_DIR, "utiles")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.makedirs(UTIL_DIR, exist_ok=True)
init_path = os.path.join(UTIL_DIR, "__init__.py")
if not os.path.exists(init_path):
    open(init_path, "a").close()

# --- Exchange helper ---
try:
    from utiles.bitget import make_exchange
except Exception as e:
    st.error(f"Failed to import utiles.bitget.make_exchange: {e}")
    st.stop()

# --- Trending module (module import, then getattr) ---
try:
    trending_mod = importlib.import_module("utiles.trending")
except Exception as e:
    st.error(f"Failed to import utiles.trending: {e}")
    st.stop()

scan_trending = getattr(trending_mod, "scan_trending", None)
explain_trending_row = getattr(trending_mod, "explain_trending_row", None)
TrendScanParams = getattr(trending_mod, "TrendScanParams", None)

# Fallbacks so app keeps running even if symbols missing
if TrendScanParams is None:
    @dataclass
    class TrendScanParams:
        min_pct_4h: float = 0.5
        min_volz_24h: float = 0.8
        rsi15m_min: float = 35.0
        rsi15m_max: float = 65.0
        min_sentiment: float = 0.05
        near_support_max_pct: float = 2.5
        near_resist_max_pct: float = 1.5

if explain_trending_row is None:
    def explain_trending_row(row: pd.Series, p: TrendScanParams) -> List[str]:
        reasons: List[str] = []
        if pd.notna(row.get("pct4h")) and row["pct4h"] >= p.min_pct_4h:
            reasons.append(f"4h momentum +{row['pct4h']:.2f}%")
        if pd.notna(row.get("vol_z24h")) and row["vol_z24h"] >= p.min_volz_24h:
            reasons.append(f"24h volume spike (z={row['vol_z24h']:.2f})")
        rsi = row.get("rsi14_15m")
        if pd.notna(rsi) and p.rsi15m_min <= rsi <= p.rsi15m_max:
            reasons.append(f"15m RSI in range ({rsi:.1f})")
        sent = row.get("sentiment_score")
        if pd.notna(sent) and sent >= p.min_sentiment:
            reasons.append(f"positive sentiment ({sent:+.2f})")
        dlow = row.get("dist_to_low_pct")
        if pd.notna(dlow) and dlow <= p.near_support_max_pct:
            reasons.append(f"near support ({dlow:.2f}% from low)")
        dhigh = row.get("dist_to_high_pct")
        if pd.notna(dhigh) and dhigh <= p.near_resist_max_pct:
            reasons.append(f"near resistance ({dhigh:.2f}% from high)")
        return reasons

if scan_trending is None:
    st.error("utiles.trending is missing: scan_trending")
    st.stop()

# --- Favorites ---
ALWAYS_INCLUDE = {"GALA/USDT", "XLM/USDT"}
DEFAULT_FAVORITES = {
    "BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","XRP/USDT",
    "LINK/USDT","AVAX/USDT","TRX/USDT","DOT/USDT","LTC/USDT",
    "INJ/USDT","GRT/USDT","CRO/USDT","XTZ/USDT","FET/USDT",
}

def _init_watchlist(exchange):
    available = {s for s in exchange.symbols if s.endswith("/USDT")}
    base = (DEFAULT_FAVORITES | ALWAYS_INCLUDE) & available
    if "working_symbols" not in st.session_state or not st.session_state["working_symbols"]:
        st.session_state["working_symbols"] = sorted(base)

# --- UI ---
st.set_page_config(page_title="Crypto Parser", layout="wide")
st.title("Crypto Parser — Watchlist & Trending")

ex = make_exchange("bitget")
_init_watchlist(ex)

colL, colR = st.columns([2, 1])
with colL:
    st.subheader("📌 Current list")
    st.write(", ".join(st.session_state["working_symbols"]))

with colR:
    st.subheader("➕ Add manually")
    universe = sorted([s for s in ex.symbols if s.endswith("/USDT")])
    add_symbol = st.selectbox("Add symbol", [""] + universe, index=0)
    if add_symbol and add_symbol not in st.session_state["working_symbols"]:
        st.session_state["working_symbols"].append(add_symbol)
        st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
        st.success(f"Added {add_symbol}")

st.markdown("---")
st.subheader("🔥 Find trending & explain why")

params = TrendScanParams()

if st.button("Scan market and add with reasons"):
    df = scan_trending(ex, params=params)

    if df.empty:
        st.info("No trending candidates met the thresholds.")
    else:
        st.dataframe(df[[
            "symbol","score","pct4h","vol_z24h","sentiment_score",
            "rsi14_15m","dist_to_low_pct","dist_to_high_pct"
        ]].head(30))

        added = []
        for _, row in df.iterrows():
            sym = row["symbol"]
            if sym not in st.session_state["working_symbols"]:
                st.session_state["working_symbols"].append(sym)
                reasons = explain_trending_row(row, params)
                added.append((sym, reasons))

        st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
        if added:
            st.success("New symbols added:")
            for sym, reasons in added:
                st.write(f"• **{sym}** — " + "; ".join(reasons))
        else:
            st.info("No new symbols were added (already present or no reasons).")
