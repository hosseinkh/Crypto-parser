# streamlit_app.py
# -----------------------------------------------------------
# Minimal, robust Streamlit app that:
# - Ensures utiles/ is a package
# - Loads trending module silently (no ImportError noise)
# - Always includes Favorites + GALA + XLM
# - Adds trending coins and shows human-readable reasons
# -----------------------------------------------------------

from __future__ import annotations

import os, sys, importlib
from typing import List, Set
import streamlit as st
import pandas as pd

# --- Make sure 'utiles' is importable -----------------------------------------
ROOT_DIR = os.path.dirname(__file__)
UTIL_DIR = os.path.join(ROOT_DIR, "utiles")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.makedirs(UTIL_DIR, exist_ok=True)
init_path = os.path.join(UTIL_DIR, "__init__.py")
if not os.path.exists(init_path):
    open(init_path, "a").close()  # create empty __init__.py

# --- Exchange via ccxt (fallback if your own wrapper isn't present) -----------
try:
    from utiles.bitget import make_exchange  # your helper if it exists
except Exception:
    import ccxt  # type: ignore

    def make_exchange(ex_name: str = "bitget"):
        ex = getattr(ccxt, ex_name)()
        ex.enableRateLimit = True
        ex.load_markets()
        return ex

# --- Import trending module silently ------------------------------------------
try:
    trending_mod = importlib.import_module("utiles.trending")
except Exception as e:
    st.error(f"Failed to import utiles.trending: {e}")
    st.stop()

scan_trending = getattr(trending_mod, "scan_trending", None)
explain_trending_row = getattr(trending_mod, "explain_trending_row", None)
TrendScanParams = getattr(trending_mod, "TrendScanParams", None)

_missing = [n for n,v in {
    "scan_trending": scan_trending,
    "explain_trending_row": explain_trending_row,
    "TrendScanParams": TrendScanParams
}.items() if v is None]
if _missing:
    st.error("utiles.trending is missing: " + ", ".join(_missing))
    st.stop()

# --- Defaults / Favorites ------------------------------------------------------
ALWAYS_INCLUDE: Set[str] = {"GALA/USDT", "XLM/USDT"}

DEFAULT_FAVORITES: Set[str] = {
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT",
    "LINK/USDT", "AVAX/USDT", "TRX/USDT", "DOT/USDT", "LTC/USDT",
    "INJ/USDT", "GRT/USDT", "CRO/USDT", "XTZ/USDT", "FET/USDT"
}

def _init_watchlist(exchange):
    available = {s for s in exchange.symbols if s.endswith("/USDT")}
    base = sorted((DEFAULT_FAVORITES | ALWAYS_INCLUDE) & available)
    if "working_symbols" not in st.session_state or not st.session_state["working_symbols"]:
        st.session_state["working_symbols"] = base

# --- App UI -------------------------------------------------------------------
st.set_page_config(page_title="Crypto Scanner", layout="wide")
st.title("📈 Crypto Scanner (favorites + trending)")

# Exchange picker (you can hardcode 'bitget' if you prefer)
ex_name = st.sidebar.selectbox("Exchange", ["bitget", "binance", "okx"], index=0)
ex = make_exchange(ex_name)
_init_watchlist(ex)

# Show where trending module was loaded from (debug)
st.caption(f"trending loaded from: {trending_mod.__file__}")

# --- Type-ahead add symbols ---------------------------------------------------
st.subheader("Add symbols (type-ahead)")
all_symbols = sorted([s for s in ex.symbols if s.endswith("/USDT")])
typed = st.text_input("Search symbols (e.g., 'INJ', 'DOGE')", "")
suggestions = [s for s in all_symbols if typed.upper() in s.upper()][:30]
col1, col2 = st.columns([3,1])
with col1:
    sel = st.selectbox("Suggestions", suggestions) if suggestions else None
with col2:
    if st.button("➕ Add selected") and sel:
        st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"] + [sel]))

st.write("📌 Current list:", ", ".join(st.session_state["working_symbols"]))

# --- Trending screener --------------------------------------------------------
st.subheader("Trending screener")
params = TrendScanParams()

if st.button("🔎 Find trending & add with reasons"):
    df = scan_trending(ex, params=params)  # scans entire USDT universe by default

    if df.empty:
        st.info("No trending candidates returned.")
    else:
        # Show the top table
        show_cols = ["symbol","score","pct4h","vol_z24h","sentiment_score","rsi14_15m","dist_to_low_pct","dist_to_high_pct"]
        st.dataframe(df[show_cols].head(30), use_container_width=True)

        # Add + show reasons
        added: List[str] = []
        for _, row in df.head(30).iterrows():
            sym = row["symbol"]
            if sym not in st.session_state["working_symbols"]:
                reasons = explain_trending_row(row, params)
                if reasons:
                    st.session_state["working_symbols"].append(sym)
                    added.append(f"➕ Added {sym}: " + "; ".join(reasons))

        st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))

        if added:
            for msg in added:
                st.success(msg)
        else:
            st.info("No new symbols added (either already present or thresholds not met).")

# --- Footer -------------------------------------------------------------------
st.markdown("---")
st.write("Tip: favorites (incl. **GALA/USDT** and **XLM/USDT**) are always included on first load.")
