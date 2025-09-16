# streamlit_app.py
# --------------------------------------------------------------------
# Streamlit app for multi-timeframe crypto snapshots + optional trending
# --------------------------------------------------------------------

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# --- local modules ---
from utiles.snapshot import build_snapshot, make_exchange
from utiles.sentiment import get_sentiment_for_symbol

# trending is optional; app still runs if you haven't added it yet
try:
    from utiles.trending import scan_trending, TrendScanParams
    TRENDING_AVAILABLE = True
except Exception:
    TRENDING_AVAILABLE = False


# ---------------- UI Helpers ---------------- #

def _default_symbols(ex_name: str) -> List[str]:
    # safe defaults
    return [
        "BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT",
        "DOT/USDT","LINK/USDT","TRX/USDT","XRP/USDT",
        "BCH/USDT","BNB/USDT"
    ]

def _exchange_symbols_cached(ex_name: str) -> List[str]:
    if "ALL_EX_SYMBOLS" not in st.session_state:
        ex = make_exchange(ex_name)
        try:
            syms = [s for s in ex.load_markets().keys() if s.endswith("/USDT")]
        except Exception:
            syms = _default_symbols(ex_name)
        st.session_state.ALL_EX_SYMBOLS = sorted(syms)
    return st.session_state.ALL_EX_SYMBOLS


def _init_session_defaults(ex_name: str):
    if "exchange" not in st.session_state:
        st.session_state.exchange = ex_name
    if "working_symbols" not in st.session_state:
        st.session_state.working_symbols = _default_symbols(ex_name)
    if "last_snapshot" not in st.session_state:
        st.session_state.last_snapshot = None


# ---------------- App Layout ---------------- #

st.set_page_config(page_title="Crypto Snapshot", layout="wide")
st.title("📊 Crypto Multi-TF Snapshot")

# Sidebar — settings
with st.sidebar:
    ex_name = st.selectbox("Exchange", ["bitget","binance","bybit","okx"], index=0)
    _init_session_defaults(ex_name)

    st.subheader("Symbols")
    # type-ahead add
    all_symbols = _exchange_symbols_cached(ex_name)
    typed = st.text_input("Type to search (e.g. 'INJ', 'DOGE')", "")
    suggestions = [s for s in all_symbols if typed.upper() in s.upper()][:30]
    if suggestions:
        sel = st.selectbox("Suggestions", suggestions)
        if st.button("➕ Add selected"):
            st.session_state.working_symbols = sorted(
                list(set(st.session_state.working_symbols + [sel]))
            )
    # remove symbol(s)
    if st.session_state.working_symbols:
        rem_sel = st.multiselect("Remove symbol(s)", st.session_state.working_symbols)
        if st.button("🗑️ Remove"):
            st.session_state.working_symbols = [s for s in st.session_state.working_symbols if s not in rem_sel]

    st.write("**Current list:**", ", ".join(st.session_state.working_symbols))

    st.divider()
    st.subheader("Trending (optional)")
    if TRENDING_AVAILABLE:
        with st.expander("Scan & add trending"):
            min_pct_4h = st.number_input("Min +4h %", value=0.5, step=0.1)
            min_vol_z = st.number_input("Min 24h vol z", value=0.8, step=0.1)
            rsi_lo = st.number_input("RSI 15m min", value=40.0, step=1.0)
            rsi_hi = st.number_input("RSI 15m max", value=65.0, step=1.0)
            near_sup_bonus = st.checkbox("Bonus if ≤2.5% from support", value=True)
            topN = st.number_input("Add top N", value=10, step=1, min_value=1, max_value=50)

            if st.button("🔍 Scan trending"):
                params = TrendScanParams(
                    min_pct_4h=min_pct_4h,
                    min_vol_z_24h=min_vol_z,
                    rsi_bounds_15m=(rsi_lo, rsi_hi),
                    bonus_near_support=0.1 if near_sup_bonus else 0.0
                )
                df, passing = scan_trending(ex_name, universe=all_symbols, params=params)
                st.write("**Trending candidates (top 25)**")
                st.dataframe(df[["symbol","score","pct4h","vol_z24h","sentiment_score","rsi14","dist_to_low_pct"]].head(25))
                if st.button(f"➕ Add top {int(topN)}"):
                    top_syms = df.head(int(topN))["symbol"].tolist()
                    st.session_state.working_symbols = sorted(list(set(st.session_state.working_symbols + top_syms)))
    else:
        st.info("Trending module not found. You can still build snapshots. Add `utiles/trending.py` later to enable scans.")

    st.divider()
    st.subheader("Snapshot options")
    tfs = st.multiselect("Timeframes", ["15m","1h","4h"], default=["15m","1h","4h"])
    limit = st.number_input("Candles per TF", value=240, min_value=100, max_value=500, step=20)

    if st.button("📦 Build Snapshot"):
        snap = build_snapshot(
            ex_name=ex_name,
            symbols=st.session_state.working_symbols,
            timeframes=tfs,
            limit=int(limit),
            sentiment_func=get_sentiment_for_symbol,
        )
        st.session_state.last_snapshot = snap

# Main panel — snapshot display
snap = st.session_state.last_snapshot
if snap:
    st.caption(f"Generated at: {snap['meta']['generated_at']} | {snap['meta']['exchange']} | TFs: {', '.join(snap['meta']['timeframes'])}")

    # Flatten a compact table for display
    rows = []
    for r in snap["rows"]:
        sym = r["symbol"]
        last = r["last"]
        sent = r["sentiment"]["score"]
        row: Dict[str, Any] = {"symbol": sym, "last": last, "sentiment": np.round(sent, 3)}
        # bring selected TF indicators up (15m only to keep table compact)
        tf15 = r["timeframes"].get("15m")
        if isinstance(tf15, dict) and "indicators" in tf15:
            ind = tf15["indicators"]
            row.update({
                "rsi14_15m": np.round(ind.get("rsi14", np.nan), 2),
                "atr14_15m": np.round(ind.get("atr14", np.nan), 6),
                "vol_z_15m": np.round(ind.get("vol_z", np.nan), 2),
                "dist_low_%": np.round(ind.get("dist_to_range_low", np.nan), 2),
                "dist_high_%": np.round(ind.get("dist_to_range_high", np.nan), 2),
            })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    st.dataframe(df)

    # Download snapshot
    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Save snapshot to file"):
            filename = f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            st.success(f"Saved to {filename}")

    with colB:
        st.download_button(
            "⬇️ Download snapshot JSON",
            data=json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
            mime="application/json",
        )

else:
    st.info("Build a snapshot from the sidebar to see results here.")
