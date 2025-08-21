# streamlit_app.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
import streamlit as st

# our utils
from utiles.bitget import make_exchange, normalize_symbol, now_utc_iso
from utiles import snapshot as snap


# -----------------------------
# Page & session state
# -----------------------------
st.set_page_config(page_title="Crypto Scanner (Bitget) – JSON Snapshot MVP", layout="wide")

if "scan_results" not in st.session_state:
    st.session_state.scan_results: List[Dict] = []

# Build full symbol list (spot /USDT)
if "all_symbols" not in st.session_state:
    ex = make_exchange()
    ex.load_markets()
    candidates = []
    for m in ex.markets.values():
        sym = m.get("symbol", "")
        if "/" in sym and sym.endswith("/USDT") and (":" not in sym) and m.get("active", True):
            candidates.append(sym)
    curated = [
        "BTC/USDT","ETH/USDT","INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT",
        "AVAX/USDT","LINK/USDT","XRP/USDT","FET/USDT","SOL/USDT","ADA/USDT",
    ]
    st.session_state.all_symbols = sorted(set(candidates) | set(curated))

if "timeframes" not in st.session_state:
    st.session_state.timeframes = ["15m", "1h", "4h"]


# -----------------------------
# Sidebar controls
# -----------------------------
st.title("🔎 Crypto Scanner (Bitget) – JSON Snapshot MVP")
st.caption("No keys. No servers. Mobile-friendly. (UTC timestamps)")

with st.sidebar:
    st.header("Settings")

    default_selection = [
        s for s in ["INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT","BTC/USDT","ETH/USDT","XRP/USDT","SOL/USDT"
                   ,"ADA/USDT","FET/USDT","AVAX/USDT","LINK/USDT","PENGU/USDT","XTZ/USDT"]
        if s in st.session_state.all_symbols
    ] or ["BTC/USDT","ETH/USDT"]

    symbols = st.multiselect(
        "Symbols (BASE/USDT)",
        options=st.session_state.all_symbols,
        default=default_selection,
        help="Type to add more symbols like LINK/USDT, AVAX/USDT, FET/USDT…",
    )

    custom = st.text_input("Add custom (e.g., FET/USDT)")
    if custom:
        cs = custom.upper().strip()
        if cs.endswith("/USDT"):
            if cs not in st.session_state.all_symbols:
                st.session_state.all_symbols.append(cs)
                st.session_state.all_symbols.sort()
            if cs not in symbols:
                symbols.append(cs)

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


# -----------------------------
# Scan helpers
# -----------------------------
def run_scan(selected_symbols: list[str], selected_tfs: list[str], tf_limits: dict[str, int]):
    ex = make_exchange()
    rows: List[Dict] = []
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
    Also attaches sentiment (general + per symbol).
    """
    from utiles.sentiment import build_sentiment_bundle

    by_symbol: dict[str, dict] = {}
    bases = set()

    for r in rows:
        sym = r["symbol"]           # e.g., "INJ/USDT"
        base = sym.split("/")[0]    # "INJ"
        bases.add(base)

        tf = r["tf"]
        blk = r["block"]
        if sym not in by_symbol:
            by_symbol[sym] = {"symbol": sym, "timeframes": {}}
        by_symbol[sym]["timeframes"][tf] = blk

    # Build sentiment once per snapshot
    sentiment = build_sentiment_bundle(sorted(bases))

    packed = {
        "generated_at_utc": now_utc_iso(),
        "sentiment": sentiment,                    # <— now included
        "symbols": list(by_symbol.values()),
        "meta": {
            "source": "bitget",
            "note": "LLM-friendly JSON snapshot",
            "how_to_read": (
                "Use 'sentiment.general' for broad market tone, "
                "'sentiment.per_symbol[SYMBOL/USDT]' for coin-level tone. "
                "'symbols[].timeframes[tf]' holds candles + indicators."
            ),
        },
    }
    return packed


# -----------------------------
# Run scan
# -----------------------------
if st.button("🚀 Scan Now"):
    if not symbols or not tfs:
        st.error("Please select at least one symbol and one timeframe.")
    else:
        with st.spinner("Scanning…"):
            st.session_state.scan_results = run_scan(symbols, tfs, tf_limits)


# -----------------------------
# Results
# -----------------------------
st.subheader("Results")

rows = st.session_state.scan_results
if not rows:
    st.info("Click **Scan Now** to fetch data.")
else:
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
    df = pd.DataFrame(table)

    # Some browsers/extensions can break the modern dataframe component.
    # Fall back to static table if that happens.
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)

    packed = pack_snapshot(rows)
    as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON snapshot",
        data=as_bytes,
        file_name=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        mime="application/json",
    )
    with st.expander("🔍 View JSON"):
        st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
