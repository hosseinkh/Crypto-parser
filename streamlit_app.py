# streamlit_app.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
import streamlit as st

# ---- our utils (folder name: utiles) ----
from utiles.bitget import make_exchange, normalize_symbol, now_utc_iso
from utiles import snapshot as snap
from utiles.sentiment import build_sentiment_bundle  # used for per-file sentiment

# =========================
# Page config & session
# =========================
st.set_page_config(page_title="Crypto Scanner (Bitget) – JSON Snapshot MVP", layout="wide")

if "scan_results" not in st.session_state:
    st.session_state.scan_results: List[Dict] = []

# Build a full symbol universe once (USDT spot-looking symbols)
if "all_symbols" not in st.session_state:
    ex_for_list = make_exchange()
    ex_for_list.load_markets()
    candidates: List[str] = []
    for m in ex_for_list.markets.values():
        sym = m.get("symbol", "")
        # keep simple SPOT-looking symbols like "BTC/USDT" (avoid futures like "BTC/USDT:USDT")
        if "/" in sym and sym.endswith("/USDT") and (":" not in sym) and m.get("active", True):
            candidates.append(sym)

    # Ensure your favorites exist
    curated = [
        "BTC/USDT","ETH/USDT","INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT",
        "AVAX/USDT","LINK/USDT","XRP/USDT","FET/USDT","SOL/USDT","ADA/USDT",
        "PENGU/USDT","XTZ/USDT"
    ]
    st.session_state.all_symbols = sorted(set(candidates) | set(curated))

if "timeframes" not in st.session_state:
    st.session_state.timeframes = ["15m", "1h", "4h"]

# =========================
# UI – controls
# =========================
st.title("🔎 Crypto Scanner (Bitget) – JSON Snapshot MVP")
st.caption("No keys. No servers. Mobile-friendly. (UTC timestamps)")

with st.sidebar:
    st.header("Settings")

    # Default preselection (safe subset)
    default_selection = [
        s for s in ["INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT"]
        if s in st.session_state.all_symbols
    ] or ["BTC/USDT","ETH/USDT"]

    # Symbols – uses the full universe
    symbols = st.multiselect(
        "Symbols (BASE/USDT)",
        options=st.session_state.all_symbols,
        default=default_selection,
        help="Type to add more symbols like LINK/USDT, AVAX/USDT, FET/USDT…",
    )

    # Quick add a custom symbol
    custom = st.text_input("Add custom (e.g., FET/USDT)")
    if custom:
        cs = custom.upper().strip()
        if cs.endswith("/USDT"):
            if cs not in st.session_state.all_symbols:
                st.session_state.all_symbols.append(cs)
                st.session_state.all_symbols.sort()
            if cs not in symbols:
                symbols.append(cs)

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

    st.markdown("---")
    split_output = st.checkbox("Split output into multiple JSON files", value=True)
    max_per_file = st.number_input(
        "Max symbols per file",
        min_value=1, max_value=50, value=5, step=1,
        help="How many symbols should be packed into each JSON file."
    )

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
                # snapshot.build_tf_block supports 'limit=' kwarg
                block = snap.build_tf_block(ex, sym_norm, tf, limit=limit)
                rows.append({"symbol": sym_norm, "tf": tf, "block": block})
            except Exception as e:
                st.warning(f"{sym_norm} ({tf}) fetch error: {e}")
    return rows

def _chunk(lst: List[dict], n: int) -> List[List[dict]]:
    """Yield chunks of size n from list."""
    if n <= 0:
        return [lst]
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def pack_snapshot_from_rows(rows: List[dict]) -> dict:
    """
    Group per-symbol and pack a single JSON snapshot (no chunking).
    Also attaches sentiment (general + per symbol) for ONLY the bases present in `rows`.
    """
    by_symbol: dict[str, dict] = {}
    bases = set()

    for r in rows:
        sym = r["symbol"]            # e.g., "INJ/USDT"
        base = sym.split("/")[0]     # "INJ"
        bases.add(base)

        tf = r["tf"]
        blk = r["block"]
        if sym not in by_symbol:
            by_symbol[sym] = {"symbol": sym, "timeframes": {}}
        by_symbol[sym]["timeframes"][tf] = blk

    # Build sentiment once per file (for the bases included)
    sentiment = build_sentiment_bundle(sorted(bases))

    packed = {
        "generated_at_utc": now_utc_iso(),
        "sentiment": sentiment,
        "symbols": list(by_symbol.values()),
        "meta": {
            "source": "bitget",
            "note": "LLM-friendly JSON snapshot",
            "how_to_read": (
                "Use 'sentiment.general' for broad market tone, "
                "'sentiment.per_symbol[BASE]' for coin-level tone. "
                "'symbols[].timeframes[tf]' holds candles + indicators."
            ),
        },
    }
    return packed

def pack_snapshot_chunked(rows: List[dict], max_symbols_per_file: int) -> List[dict]:
    """
    Split the scan results into multiple JSON payloads, each with up to N symbols.
    Sentiment is computed per-chunk so each output is standalone.
    """
    # First, group rows by symbol to avoid splitting the same symbol across files
    symbols_map: Dict[str, List[dict]] = {}
    for r in rows:
        symbols_map.setdefault(r["symbol"], []).append(r)

    # Rebuild a list of "symbol bundles"
    symbol_bundles = []
    for sym, rlist in symbols_map.items():
        symbol_bundles.append(rlist)

    # Chunk by symbol
    chunks: List[List[dict]] = []
    if max_symbols_per_file <= 0:
        chunks = [rows]
    else:
        # Make contiguous groups where each group contains whole symbols
        current: List[dict] = []
        seen_syms: set = set()
        for bundle in symbol_bundles:
            sym = bundle[0]["symbol"]
            if len(seen_syms) >= max_symbols_per_file:
                chunks.append(current)
                current = []
                seen_syms = set()
            current.extend(bundle)
            seen_syms.add(sym)
        if current:
            chunks.append(current)

    # Pack each chunk
    return [pack_snapshot_from_rows(chunk_rows) for chunk_rows in chunks]

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
    df = pd.DataFrame(table)

    # robust display
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)

    # === Download(s) ===
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if split_output:
        payloads = pack_snapshot_chunked(rows, int(max_per_file))
        st.write(f"Creating **{len(payloads)}** JSON files (max {int(max_per_file)} symbols per file).")
        for idx, packed in enumerate(payloads, start=1):
            as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            st.download_button(
                f"⬇️ Download JSON #{idx}",
                data=as_bytes,
                file_name=f"snapshot_{ts}_{idx:02d}.json",
                mime="application/json",
                key=f"dl_{idx}",
            )
        with st.expander("🔍 Preview first JSON"):
            st.code(json.dumps(payloads[0], indent=2, ensure_ascii=False), language="json")
    else:
        packed = pack_snapshot_from_rows(rows)
        as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        st.download_button(
            "⬇️ Download single JSON snapshot",
            data=as_bytes,
            file_name=f"snapshot_{ts}.json",
            mime="application/json",
        )
        with st.expander("🔍 View JSON"):
            st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
