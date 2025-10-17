# streamlit_app.py
from __future__ import annotations
import json, io
import streamlit as st
from typing import List, Dict, Any

# ── BEGIN: strict package import bootstrap (fixes Streamlit Cloud ImportError) ──
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # .../crypto-parser
REPO_DIR = APP_DIR.parent                          # repo root

# Ensure Python can locate the 'utiles' package regardless of working dir
for p in (str(APP_DIR), str(REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import ONLY via package so relative imports inside utiles/*.py keep working
from utiles.snapshot import SnapshotParams, build_snapshot_v41, DEFAULT_TFS
from utiles.trending import scan_trending, explain_trending_row, TrendScanParams
from utiles.bitget import make_exchange
# ── END: strict package import bootstrap ──

st.set_page_config(page_title="Crypto parser", page_icon="📊", layout="centered")

# -------------------------- Sidebar: base universe -----------------------------
st.sidebar.title("Settings")
exchange_name = st.sidebar.selectbox("Exchange", ["bitget"], index=0)

default_list = st.sidebar.text_area(
    "Base list (comma separated, /USDT)",
    "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT,TRX/USDT,LINK/USDT,DOT/USDT,LTC/USDT,XRP/USDT, XTZ/USDT, PENGU/USDT, CTO/USDT,INJ/USDT, GRT/USDT, AVAX/USDT, FET/USDT, ADA/USDT,XRP/USDT,DOGE/USDT,MATIC/USDT,SHIB/USDT,ATOM/USDT,NEAR/USDT,APT/USDT,ARB/USDT,OP/USDT,ICP/USDT,ETC/USDT"
)
base_list = [s.strip().upper() for s in default_list.split(",") if s.strip()]

ALWAYS_INCLUDE = ["GALA/USDT", "XLM/USDT"]

# NEW: Active-trade mode (adds 5m for L1/L0 logic) + optional positions map
active_trade_mode = st.sidebar.checkbox("Active-trade mode (adds 5m TF)", value=False)
st.sidebar.caption("Adds 5m so micro-divergence (L1) & early partial (L0) can work.")

positions_text = st.sidebar.text_area(
    "Positions (optional: one per line, e.g.  SOL/USDT=132.5 )",
    value="",
    help="If provided, snapshot computes profit_pct & ATR gain multiple for early partials."
)

def _parse_positions(text: str) -> Dict[str, Dict[str, Any]]:
    pos: Dict[str, Dict[str, Any]] = {}
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or "=" not in ln:
            continue
        sym, px = ln.split("=", 1)
        sym = sym.strip().upper()
        try:
            price = float(px.strip())
        except Exception:
            continue
        pos[sym] = {"entry_price": price}
    return pos

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
st.markdown(
    "<div style='display:inline-block;padding:4px 8px;border-radius:8px;"
    "background:#e8f5e9;color:#1b5e20;font-weight:600;'>Closed candles only ✓</div>"
    "<span style='margin-left:8px;color:#555;'>Aligned across TFs (15m/1h/4h)</span>",
    unsafe_allow_html=True
)

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

# Default selection keeps your original, but we may add 5m in active mode
default_sel = [tf for tf in ["15m", "1h", "4h"] if tf in ALL_TFS]
timeframes = st.multiselect(
    "Timeframes (multi-select)",
    ALL_TFS,
    default=default_sel,
    key="tfs_v41_multiselect",
)
st.caption(f"Available TFs: {', '.join(ALL_TFS)} | Selected: {', '.join(timeframes or DEFAULT_TFS)}")

# If active-trade mode, ensure 5m is included (for L1 micro-div & smoother L0)
if active_trade_mode and "5m" in ALL_TFS and "5m" not in timeframes:
    timeframes = ["5m"] + (timeframes or default_sel)

candles_limit = st.number_input("Candles per TF", value=240, min_value=100, max_value=2000, step=10)
include_sent_snapshot = st.checkbox("Include sentiment (snapshot)", value=True)

if st.button("Build snapshot now", key="build_snapshot_btn"):
    working_universe = st.session_state["working_symbols"]
    if not working_universe:
        st.error("Working list is empty. Add symbols or run the trending scan first.")
    else:
        positions_map = _parse_positions(positions_text)

        sparams = SnapshotParams.with_defaults(
            timeframes=timeframes or DEFAULT_TFS,
            candles_limit=candles_limit,
            exchange_name=exchange_name,
            universe=working_universe,
            include_sentiment=include_sent_snapshot,
            meta={
                "ui_timeframes": timeframes or DEFAULT_TFS,
                "ui_symbols": working_universe,
                # pass positions if provided → enables profit_pct & L0 early-partial calc
                "positions": positions_map if positions_map else {},
            },
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
