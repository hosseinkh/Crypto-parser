# streamlit_app.py
# --------------------------------------------------------------------
# Streamlit UI:
#  - Initializes default watchlist with Favorites + GALA + XLM
#  - Type-ahead add
#  - Scan Trending -> adds coins and shows reasons for each
#  - Build snapshots (15m/1h/4h) using compute_indicators (vol_z included)
# --------------------------------------------------------------------

from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import ccxt

from utiles.indicators import compute_indicators, summarize_last_row
from utiles.trending import scan_trending, explain_trending_row, TrendScanParams


# ----------------------- Exchange helpers -----------------------------

@st.cache_resource
def make_exchange(ex_name: str = "bitget"):
    ex_class = getattr(ccxt, ex_name)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    return ex

def safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int = 240):
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        st.warning(f"Failed OHLCV {symbol} {timeframe}: {e}")
        return []


# -------------------- Defaults / Favorites ----------------------------

ALWAYS_INCLUDE = {"GALA/USDT", "XLM/USDT"}

DEFAULT_FAVORITES = {
    "BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","XRP/USDT",
    "LINK/USDT","AVAX/USDT","TRX/USDT","DOT/USDT","LTC/USDT",
}

def _build_default_universe(ex_name: str) -> List[str]:
    ex = make_exchange(ex_name)
    try:
        avail = {s for s in ex.symbols if s.endswith("/USDT")}
    except Exception:
        avail = set()
    base = (DEFAULT_FAVORITES | ALWAYS_INCLUDE) & avail
    return sorted(base) if base else sorted(DEFAULT_FAVORITES | ALWAYS_INCLUDE)


def _init_session_defaults(ex_name: str):
    if "working_symbols" not in st.session_state:
        st.session_state.working_symbols = _build_default_universe(ex_name)


# --------------------- Snapshot construction --------------------------

def build_tf_block(ex, symbol: str, tf: str, limit: int = 240) -> Dict[str, Any]:
    ohlcv = safe_fetch_ohlcv(ex, symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    if df.empty or df.shape[0] < 60:
        return {"tf": tf, "error": "insufficient data"}
    ind = compute_indicators(df)
    last = {
        "t": int(df["timestamp"].iloc[-1]),
        "o": float(df["open"].iloc[-1]),
        "h": float(df["high"].iloc[-1]),
        "l": float(df["low"].iloc[-1]),
        "c": float(df["close"].iloc[-1]),
        "v": float(df["volume"].iloc[-1]),
    }
    return {"tf": tf, "last": last, "indicators": summarize_last_row(ind)}


def build_snapshot(ex_name: str, symbols: List[str], limit: int = 240) -> Dict[str, Any]:
    ex = make_exchange(ex_name)
    rows, errors = [], []
    for sym in symbols:
        try:
            tf_blocks = {}
            for tf in ["15m", "1h", "4h"]:
                tf_blocks[tf] = build_tf_block(ex, sym, tf, limit)
            rows.append({
                "symbol": sym,
                "timeframes": tf_blocks,
                "sentiment": {"score": 0.0, "label": "neutral"}  # placeholder
            })
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})

    return {
        "meta": {
            "exchange": ex_name,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "timeframes": ["15m","1h","4h"],
            "limit_per_tf": limit,
            "symbols_count": len(rows)
        },
        "rows": rows,
        "errors": errors
    }


# ----------------------------- UI ------------------------------------

def main():
    st.set_page_config(page_title="Crypto Snapshot & Screener", layout="wide")
    st.title("📊 Crypto Snapshot & Trending Screener")

    ex_name = st.sidebar.selectbox("Exchange", ["bitget","binance"], index=0)
    _init_session_defaults(ex_name)
    ex = make_exchange(ex_name)

    # --- Type-ahead add
    st.sidebar.subheader("Add symbols")
    try:
        all_syms = [s for s in ex.symbols if s.endswith("/USDT")]
    except Exception:
        all_syms = []
    typed = st.sidebar.text_input("Type (e.g., INJ, DOGE)", "")
    suggestions = [s for s in all_syms if typed.upper() in s.upper()][:30]
    if suggestions:
        sel = st.sidebar.selectbox("Suggestions", suggestions)
        if st.sidebar.button("➕ Add selected"):
            st.session_state.working_symbols = sorted(set(st.session_state.working_symbols) | {sel})

    st.sidebar.markdown("**Current list:**<br>" + ", ".join(st.session_state.working_symbols), unsafe_allow_html=True)

    # --- Trending + reasons
    st.header("Trending")
    params = TrendScanParams()
    if st.button("🔎 Scan trending (15m/1h) and add with reasons"):
        df = scan_trending(ex, universe=None, params=params)
        st.dataframe(df.head(25))

        added = []
        for _, row in df.head(25).iterrows():
            sym = row["symbol"]
            if sym not in st.session_state.working_symbols:
                reasons = explain_trending_row(row, params)
                if reasons:
                    st.session_state.working_symbols.append(sym)
                    added.append((sym, reasons))
        st.session_state.working_symbols = sorted(set(st.session_state.working_symbols))

        if added:
            for sym, reasons in added:
                st.success(f"➕ Added {sym}: " + "; ".join(reasons))
        else:
            st.info("No new symbols were added (already present or no candidates met thresholds).")

    # --- Snapshot section
    st.header("Snapshot")
    cols = st.columns([2,1])
    with cols[0]:
        st.write("Symbols:", ", ".join(st.session_state.working_symbols))
    with cols[1]:
        if st.button("📦 Build snapshot"):
            snap = build_snapshot(ex_name, st.session_state.working_symbols, limit=240)
            st.json(snap)

if __name__ == "__main__":
    main()
