# streamlit_app.py
import json
import pandas as pd
import streamlit as st

from utiles.bitget import make_exchange, normalize_symbol, now_utc_iso
from utiles.snapshot import build_tf_block

st.set_page_config(page_title="Bitget Crypto Scanner", layout="wide")
st.title("🔎 Crypto Scanner (Bitget) – JSON Snapshot MVP")

# ---------------- Settings UI ----------------
symbols = st.multiselect(
    "Symbols (BASE/USDT)",
    options=["BTC/USDT","ETH/USDT","SOL/USDT","INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT","FET/USDT","AVAX/USDT","LINK/USDT"],
    default=["INJ/USDT","GRT/USDT","CRO/USDT","TRX/USDT"]
)

tfs = st.multiselect("Timeframes", options=["15m","1h","4h"], default=["15m","1h","4h"])
c15 = st.number_input("15m candles", min_value=10, max_value=300, value=30, step=1)
c1h = st.number_input("1h candles",  min_value=10, max_value=500, value=50, step=1)
c4h = st.number_input("4h candles",  min_value=10, max_value=800, value=80, step=1)

tf_counts = {"15m": int(c15), "1h": int(c1h), "4h": int(c4h)}
run = st.button("🚀 Scan Now")

if run:
    ex = make_exchange()
    results = []
    for sym in symbols:
        norm = normalize_symbol(sym)
        sym_out = {"symbol": norm, "built_at": now_utc_iso(), "timeframes": {}}

        for tf in tfs:
            try:
                lc = tf_counts[tf]
                # IMPORTANT: positional only (no 'limit=' anywhere)
                block, _ = build_tf_block(ex, norm, tf, lc)
                sym_out["timeframes"][tf] = block
            except Exception as e:
                st.warning(f"{norm} ({tf}) fetch error: {e}")
        results.append(sym_out)

    st.subheader("JSON")
    st.code(json.dumps(results, indent=2), language="json")
