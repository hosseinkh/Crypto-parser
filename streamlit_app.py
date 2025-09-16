# streamlit_app.py
from __future__ import annotations
import json
import io
import streamlit as st
from typing import List, Dict, Any
from utiles.snapshot import SnapshotParams, build_snapshot_v41, DEFAULT_TFS
from utiles.trending import scan_trending, explain_trending_row, TrendScanParams

st.set_page_config(page_title="Crypto parser", page_icon="📊", layout="centered")

# -------------------------- Sidebar settings ----------------------------------
st.sidebar.title("Settings")
exchange_name = st.sidebar.selectbox("Exchange", ["bitget"], index=0)
default_list = st.sidebar.text_area(
    "Favorites (comma separated symbols, /USDT)",
    "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT,TRX/USDT,LINK/USDT,DOT/USDT,LTC/USDT,XRP/USDT"
)
favorites = [s.strip() for s in default_list.split(",") if s.strip()]

st.title("🔍 Scan market and add Top-N with reasons")
col_a, col_b = st.columns([1,1])

with col_a:
    top_n = st.number_input("Top-N trendy", value=10, min_value=1, max_value=50, step=1)
with col_b:
    min_vz = st.slider("Min vol_z", 0.0, 3.0, 0.8, 0.05)

if st.button("Scan market and add Top-N with reasons"):
    tparams = TrendScanParams(exchange_name=exchange_name, top_n=top_n, min_vol_z=min_vz)
    rows = scan_trending(tparams)
    if not rows:
        st.warning("No trendy symbols found (vol_z too strict?).")
    else:
        st.success(f"Top {len(rows)} trending:")
        added: List[str] = []
        for r in rows:
            st.write("• " + explain_trending_row(r))
            if r["symbol"] not in favorites:
                favorites.append(r["symbol"])
                added.append(r["symbol"])
        if added:
            st.info("Added symbols: " + ", ".join(added))
        else:
            st.info("No new symbols added (already in favorites).")

st.markdown("---")

# -------------------------- Build Snapshot (v4.1) -----------------------------
st.header("📸 Build snapshot (v4.1)")

# Choose timeframes with defaults 15m, 1h, 4h
ALL_TFS = ["1m","5m","15m","30m","1h","2h","4h","6h","12h","1d"]
default_idx = [ALL_TFS.index("15m"), ALL_TFS.index("1h"), ALL_TFS.index("4h")]
timeframes = st.multiselect("Timeframes", ALL_TFS, default=[ALL_TFS[i] for i in default_idx])
candles_limit = st.number_input("Candles limit (per TF)", value=240, min_value=100, max_value=2000, step=10)

if st.button("Build snapshot now"):
    sparams = SnapshotParams.with_defaults(
        timeframes=timeframes or DEFAULT_TFS,
        candles_limit=candles_limit,
        exchange_name=exchange_name,
        favorites=favorites,
        meta={"ui_timeframes": timeframes or DEFAULT_TFS, "ui_favorites": favorites},
    )
    snapshot = build_snapshot_v41(sparams)
    st.success(f"Built snapshot for {len(snapshot['items'])} symbols on {snapshot['timeframes']}")
    # show small preview
    st.json({ "version": snapshot["version"], "timeframes": snapshot["timeframes"], "candles_limit": snapshot["candles_limit"] })
    # download
    buffer = io.StringIO()
    buffer.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), indent=2))
    st.download_button(
        "Download snapshot.json",
        data=buffer.getvalue().encode("utf-8"),
        file_name=f"snapshot_{snapshot['version']}.json",
        mime="application/json",
    )
