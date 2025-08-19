# streamlit_app.py
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ---- our utils (folder name: utiles) ----
from utiles.bitget import make_exchange, normalize_symbol, now_utc_iso
from utiles import snapshot as snap

# =========================
# Page config & session
# =========================
st.set_page_config(page_title="Bitget Crypto Scanner – JSON Snapshot MVP", layout="wide")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "symbols" not in st.session_state:
    st.session_state.symbols = ["BTC/USDT", "ETH/USDT", "INJ/USDT", "GRT/USDT", "CRO/USDT", "TRX/USDT"]
if "timeframes" not in st.session_state:
    st.session_state.timeframes = ["15m", "1h", "4h"]

# =========================
# UI – controls
# =========================
st.title("🔎 Crypto Scanner (Bitget) – JSON Snapshot MVP")
st.caption("No keys. No servers. Mobile-friendly. (UTC timestamps)")

with st.sidebar:
    st.header("Settings")

    # Symbols
    symbols = st.multiselect(
        "Symbols (BASE/USDT)",
        options=st.session_state.symbols,
        default=["INJ/USDT", "GRT/USDT", "CRO/USDT", "TRX/USDT"],
        help="Type to add more symbols like LINK/USDT, AVAX/USDT, FET/USDT…",
    )

    # Timeframes
    tfs = st.multiselect(
        "Timeframes",
        options=st.session_state.timeframes,
        default=["15m", "1h", "4h"],
    )

    st.markdown("---")
    candles_15m = st.number_input("15m candles", min_value=10, max_value=500, value=30, step=5)
    candles_1h  = st.number_input("1h candles",  min_value=10, max_value=500, value=50, step=5)
    candles_4h  = st.number_input("4h candles",  min_value=10, max_value=500, value=80, step=5)

    tf_limits = {"15m": int(candles_15m), "1h": int(candles_1h), "4h": int(candles_4h)}

st.write("")  # spacing

# =========================
# Scan helper
# =========================
def run_scan(selected_symbols: list[str], selected_tfs: list[str], tf_limits: dict[str, int]):
    """
    Build per-(symbol, timeframe) blocks using utils.snapshot.build_tf_block.
    Returns: list of dicts: {symbol, tf, block}
    """
    ex = make_exchange()
    rows: list[dict] = []

    for sym in selected_symbols:
        sym_norm = normalize_symbol(sym, "USDT")
        for tf in selected_tfs:
            limit = int(tf_limits.get(tf, 50))
            try:
                block = snap.build_tf_block(ex, sym_norm, tf, limit=limit)
                rows.append({"symbol": sym_norm, "tf": tf, "block": block})
            except Exception as e:
                st.warning(f"{sym_norm} ({tf}) fetch error: {e}")
    return rows


def pack_snapshot(rows: list[dict]) -> dict:
    """
    Group per-symbol and pack a single JSON snapshot ready to download.
    """
    by_symbol: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        tf = r["tf"]
        blk = r["block"]
        if sym not in by_symbol:
            by_symbol[sym] = {"symbol": sym, "timeframes": {}}
        by_symbol[sym]["timeframes"][tf] = blk

    packed = {
        "generated_at_utc": now_utc_iso(),
        "symbols": list(by_symbol.values()),
        "meta": {"source": "bitget", "note": "LLM-friendly JSON snapshot"},
    }
    return packed


# =========================
# Run scan
# =========================
if st.button("🚀 Scan Now"):
    if not symbols or not tfs:
        st.error("Please select at least one symbol and one timeframe.")
    else:
        with st.spinner("Scanning…"):
            st.session_state.scan_results = run_scan(symbols, tfs, tf_limits)

# =========================
# Results
# =========================
st.subheader("Results")

rows = st.session_state.scan_results
if not rows:
    st.info("Click **Scan Now** to fetch data.")
else:
    # quick table summary
    table = []
    for r in rows:
        ind = r["block"]["indicators"]
        struct = r["block"]["structure"]
        table.append(
            {
                "symbol": r["symbol"],
                "tf": r["tf"],
                "close": round(ind.get("close", float("nan")), 8),
                "rsi14": round(ind.get("rsi14", float("nan")), 2),
                "atr14": round(ind.get("atr14", float("nan")), 8),
                "trend": struct.get("trend", "?"),
                "dist_to_high": round(ind.get("dist_to_high", float("nan")), 4),
                "dist_to_low": round(ind.get("dist_to_low", float("nan")), 4),
                "vol_z": round(ind.get("vol_z", float("nan")), 2),
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True)

    # pack & download
    packed = pack_snapshot(rows)
    as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON snapshot",
        data=as_bytes,
        file_name=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        mime="application/json",
    )

    # pretty viewer
    with st.expander("🔍 View JSON"):
        st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
