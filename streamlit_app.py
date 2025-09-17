# streamlit_app.py
from __future__ import annotations
import json, io
import streamlit as st
from typing import List

from utiles.snapshot import SnapshotParams, build_snapshot_v41, DEFAULT_TFS
from utiles.trending import scan_trending, explain_trending_row, TrendScanParams
from utiles.bitget import make_exchange

st.set_page_config(page_title="Crypto parser", page_icon="📊", layout="centered")

# -------------------------- Sidebar: base universe -----------------------------
st.sidebar.title("Settings")
exchange_name = st.sidebar.selectbox("Exchange", ["bitget"], index=0)

default_list = st.sidebar.text_area(
    "Base list (comma separated, /USDT)",
    "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT,TRX/USDT,LINK/USDT,DOT/USDT,LTC/USDT,XRP/USDT, XTZ/USDT, PENGU/USDT, CTO/USDT,INJ/USDT, GRT/USDT, AVAX/USDT, FET/USDT, ADA/USDT,XRP/USDT")
base_list = [s.strip().upper() for s in default_list.split(",") if s.strip()]

ALWAYS_INCLUDE = ["GALA/USDT", "XLM/USDT"]

if "working_symbols" not in st.session_state:
    st.session_state["working_symbols"] = sorted(set(base_list + ALWAYS_INCLUDE))

def show_working_list():
    wl = st.session_state["working_symbols"]
    st.write(", ".join(wl) if wl else "—")

# -------------------------- Manual add/remove ----------------------------------
st.title("📌 Current working list")
col_a, col_b = st.columns([3, 2])
with col_a:
    show_working_list()

with col_b:
    ex = make_exchange(exchange_name)
    all_syms = sorted([s for s in getattr(ex, "symbols", []) if s.endswith("/USDT")]) or []
    manual = st.selectbox("➕ Add symbol (/USDT)", [""] + all_syms, index=0, key="add_sym_box")
    if manual:
        if manual not in st.session_state["working_symbols"]:
            st.session_state["working_symbols"].append(manual)
            st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
            st.success(f"Added {manual}")
        else:
            st.info(f"{manual} already in list.")
    if st.session_state["working_symbols"]:
        rem = st.selectbox("➖ Remove symbol", [""] + st.session_state["working_symbols"], index=0, key="rem_sym_box")
        if rem:
            st.session_state["working_symbols"] = [s for s in st.session_state["working_symbols"] if s != rem]
            st.warning(f"Removed {rem}")

st.markdown("---")

# -------------------------- Trending scan -------------------------------------
st.header("🔍 Scan market and add Top-N with reasons")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    top_n = st.number_input("Top-N trendy", value=10, min_value=1, max_value=50, step=1)
with col2:
    min_vz = st.slider("Min vol_z", 0.0, 3.0, 0.8, 0.05)
with col3:
    tf_scan = st.selectbox("Scan timeframe", ["15m", "1h", "4h"], index=0)
with col4:
    include_sent_scan = st.checkbox("Include sentiment (scan)", value=True)

if st.button("Scan market and add Top-N with reasons"):
    tparams = TrendScanParams(
        exchange_name=exchange_name,
        top_n=top_n,
        min_vol_z=min_vz,
        timeframe=tf_scan,
        include_sentiment=include_sent_scan
    )
    rows = scan_trending(tparams)
    if not rows:
        st.warning("No trendy symbols found (try lowering Min vol_z).")
    else:
        st.success(f"Top {len(rows)} trending on {tf_scan}:")
        newly_added: List[str] = []
        for r in rows:
            st.write("• " + explain_trending_row(r))
            sym = r["symbol"].upper()
            if sym not in st.session_state["working_symbols"]:
                st.session_state["working_symbols"].append(sym)
                newly_added.append(sym)
        if newly_added:
            st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
            st.info("Added to working list: " + ", ".join(newly_added))
        else:
            st.info("No new symbols added (already present).")

st.markdown("---")

# -------------------------- Build Snapshot (v4.1) -----------------------------
st.header("📸 Build snapshot (v4.1)")

cap_tfs = []
try:
    if getattr(ex, "timeframes", None):
        cap_tfs = list(ex.timeframes.keys())
except Exception:
    pass
FALLBACK_TFS = ["1m","5m","15m","30m","1h","2h","4h","6h","12h","1d"]
ALL_TFS = sorted(set(cap_tfs or FALLBACK_TFS),
                 key=lambda x: (x[-1], float(x[:-1]) if x[:-1].isdigit() else 0))

timeframes = st.multiselect(
    "Timeframes (multi-select)",
    ALL_TFS,
    default=[tf for tf in ["15m", "1h", "4h"] if tf in ALL_TFS],
    key="tfs_v41_multiselect",
)
st.caption(f"Available TFs: {', '.join(ALL_TFS)} | Selected: {', '.join(timeframes or DEFAULT_TFS)}")

candles_limit = st.number_input("Candles per TF", value=240, min_value=100, max_value=2000, step=10)
include_sent_snapshot = st.checkbox("Include sentiment (snapshot)", value=True)

if st.button("Build snapshot now", key="build_snapshot_btn"):
    working_universe = st.session_state["working_symbols"]
    if not working_universe:
        st.error("Working list is empty. Add symbols or run the trending scan first.")
    else:
        sparams = SnapshotParams.with_defaults(
            timeframes=timeframes or DEFAULT_TFS,
            candles_limit=candles_limit,
            exchange_name=exchange_name,
            universe=working_universe,
            include_sentiment=include_sent_snapshot,   # ✅
            meta={"ui_timeframes": timeframes or DEFAULT_TFS, "ui_symbols": working_universe},
        )
        snapshot = build_snapshot_v41(sparams)

        st.success(f"Built snapshot for {len(snapshot['items'])} symbols on {snapshot['timeframes']}")
        st.json({
            "version": snapshot["version"],
            "timeframes": snapshot["timeframes"],
            "candles_limit": snapshot["candles_limit"],
            "exchange": snapshot["exchange"],
            "count": len(snapshot["items"]),
        })

        buf = io.StringIO()
        buf.write(json.dumps(snapshot, ensure_ascii=False, indent=2))
        st.download_button(
            "⬇️ Download snapshot.json",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"snapshot_{snapshot['version']}.json",
            mime="application/json",
        )
