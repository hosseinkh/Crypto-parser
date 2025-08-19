import json, time
from datetime import datetime, timezone
from dateutil import tz
import numpy as np
import pandas as pd
import streamlit as st

from utils.bitget import make_exchange, normalize_symbol, fetch_ohlcv_df, now_utc_iso
from utils import indicators as ind
from utils import filters as filt
from utils import snapshot as snap
from utils import trade_log as tlog
from utils import sentiment as sent

st.set_page_config(page_title="Bitget Crypto Scanner", layout="wide")

# ---------- Session State ----------
if "ex" not in st.session_state:
    st.session_state.ex = None
if "trades" not in st.session_state:
    st.session_state.trades = tlog.empty_df()
if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = None

# ---------- Header ----------
st.title("🔎 Crypto Scanner (Bitget) – JSON Snapshot MVP")
st.caption("No keys. No servers. Mobile-friendly. (UTC timestamps)")

# ---------- Controls ----------
with st.sidebar:
    st.subheader("Settings")
    default_symbols = ["TRX/USDT","INJ/USDT","GRT/USDT","CRO/USDT","FET/USDT","AVAX/USDT","LINK/USDT","XTZ/USDT","BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT"]
    symbols = st.multiselect("Symbols (BASE/USDT)", default_symbols, default=default_symbols[:6])
    tfs = st.multiselect("Timeframes", ["15m","1h","4h"], default=["15m","1h","4h"])
    candles_15m = st.number_input("15m candles", 10, 200, 30, step=5)
    candles_1h  = st.number_input("1h candles", 10, 300, 50, step=5)
    candles_4h  = st.number_input("4h candles", 10, 400, 80, step=5)

    st.markdown("---")
    st.subheader("Filters")
    use_pullback = st.checkbox("Pullback in uptrend", value=True)
    use_oversold = st.checkbox("Oversold near support", value=True)
    use_breakout = st.checkbox("Breakout with volume", value=True)

    st.markdown("---")
    st.subheader("Sentiment (optional)")
    use_sentiment = st.checkbox("Fetch global sentiment (Fear&Greed, dominance)", value=False)

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh while open (every 5 min)", value=False)

# ---------- Exchange init ----------
if st.session_state.ex is None:
    try:
        st.session_state.ex = make_exchange()
    except Exception as e:
        st.error(f"Could not initialize Bitget via CCXT: {e}")

ex = st.session_state.ex

# ---------- Scan Now ----------
colA, colB = st.columns([1,1])
with colA:
    do_scan = st.button("🚀 Scan Now", use_container_width=True)
with colB:
    if auto_refresh:
        st.info("Auto-refresh is ON. The app will rescan every ~5 minutes while open.")

def build_global_sentiment():
    if not use_sentiment:
        return {
            "source_window": "6h",
            "fear_greed": None,
            "btc_dominance_pct": None,
            "total_mcap_24h_pct": None,
            "total3_24h_pct": None,
            "avg_funding_pct": None,
            "btc_oi_6h_pct": None,
            "news_bullish_ratio": None,
            "risk_regime": "neutral",
            "notes": "Sentiment disabled."
        }
    fng = sent.get_fear_greed()
    cg = sent.get_coingecko_global()
    btc_dom = cg.get("btc_dominance_pct")
    total_mcap_24h_pct = cg.get("total_mcap_24h_pct")
    # TOTAL3 approx: we skip in v1; leave None or estimate later
    total3_24h_pct = None
    regime = sent.derive_risk_regime(fng, total3_24h_pct, btc_dom)
    return {
        "source_window": "6h",
        "fear_greed": fng,
        "btc_dominance_pct": btc_dom,
        "total_mcap_24h_pct": total_mcap_24h_pct,
        "total3_24h_pct": total3_24h_pct,
        "avg_funding_pct": None,
        "btc_oi_6h_pct": None,
        "news_bullish_ratio": None,
        "risk_regime": regime,
        "notes": "Basic global sentiment."
    }

def scan_once():
    if ex is None:
        st.error("Exchange not initialized.")
        return

    generated_at = now_utc_iso()
    tf_candles = {"15m": candles_15m, "1h": candles_1h, "4h": candles_4h}
    tf_candles = {k:v for k,v in tf_candles.items() if k in tfs}

    coins = []
    shortlist_rows = []

    for sym in symbols:
        sym_norm = normalize_symbol(sym)
        price = None
        tf_blocks = {}
        ok_any = False

        for tf, limit in tf_candles.items():
            try:
                df = fetch_ohlcv_df(ex, sym_norm, tf, limit=max(limit, 60))
                if df.empty or len(df) < 20:
                    continue
                price = float(df["close"].iloc[-1])
                block = snap.build_tf_block(df, price, tf, lc_count=limit)
                tf_blocks[tf] = block
                ok_any = True
            except Exception as e:
                st.warning(f"{sym_norm} ({tf}) fetch error: {e}")
                continue

        if not ok_any:
            continue

        coin_obj = {
            "symbol": sym_norm,
            "base": sym_norm.split("/")[0],
            "quote": sym_norm.split("/")[1],
            "price": price,
            "timeframes": tf_blocks,
            "sentiment": None,  # per-coin funding/headlines optional in v1
            "llm_notes": ""
        }
        coin_obj = snap.attach_screen_flags(coin_obj, filt)

        # compute which filters fired on chosen TF
        flags = coin_obj["screen_flags"]
        fired = []
        if use_pullback and flags["pullback_in_uptrend"]: fired.append("pullback")
        if use_oversold and flags["oversold_near_support"]: fired.append("oversold")
        if use_breakout and flags["breakout_with_volume"]: fired.append("breakout")

        if fired:
            shortlist_rows.append({
                "symbol": sym_norm,
                "price": price,
                "flags": ", ".join(fired),
                "tf_used": next((tf for tf in ["15m","1h","4h"] if tf in tf_blocks), None),
                "rsi": round(tf_blocks[next((tf for tf in ["15m","1h","4h"] if tf in tf_blocks))]["indicators"]["rsi"], 1),
                "trend": tf_blocks[next((tf for tf in ["15m","1h","4h"] if tf in tf_blocks))]["structure"]["trend"],
            })

        coins.append(coin_obj)

    snapshot = {
        "schema_version": "1.0.0",
        "generated_at": generated_at,
        "exchange": "bitget",
        "quote_currency": "USDT",
        "universe": symbols,
        "timeframes": list(tf_candles.keys()),
        "global_sentiment": build_global_sentiment(),
        "coins": coins
    }
    st.session_state.last_snapshot = snapshot

    # UI: shortlist
    st.subheader("Shortlist (filters matched)")
    if shortlist_rows:
        st.dataframe(pd.DataFrame(shortlist_rows))
    else:
        st.info("No symbols matched the selected filters.")

    # UI: snapshot JSON (preview + download)
    st.subheader("Snapshot JSON")
    st.json(snapshot, expanded=False)
    st.download_button("⬇️ Download snapshot JSON", data=json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"snapshot_{generated_at.replace(':','-')}.json", mime="application/json")

    # Copy-friendly compact JSON
    with st.expander("Copy compact JSON for ChatGPT"):
        st.code(json.dumps(snapshot, separators=(',',':')), language="json")

# Run scan
if do_scan:
    scan_once()

# Optional auto-refresh while open
if auto_refresh:
    st.experimental_rerun()

st.markdown("---")
# ---------- Trade Log (your lessons) ----------
st.header("🧾 Trade Log (your lessons)")

# Upload existing CSV
upload = st.file_uploader("Upload existing trade_log.csv (optional)", type=["csv"])
if upload is not None:
    try:
        st.session_state.trades = tlog.load_csv(upload.read())
        st.success("Trade log loaded.")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

with st.form("new_trade_form", clear_on_submit=True):
    st.subheader("Add / Update Trade")
    col1, col2, col3 = st.columns(3)
    with col1:
        time_open = st.text_input("Time open (UTC, e.g. 2025-08-19T10:30:00Z)", value="")
        symbol = st.text_input("Symbol (e.g., FET/USDT)", value="")
        side = st.selectbox("Side", ["long","short"])
    with col2:
        entry_plan_entry = st.text_input("Planned entry", value="")
        entry_plan_sl    = st.text_input("Planned SL", value="")
        entry_plan_tp    = st.text_input("Planned TP", value="")
        size             = st.text_input("Size (quote)", value="")
    with col3:
        time_fill = st.text_input("Fill time (UTC) (optional)", value="")
        price_fill = st.text_input("Fill price (optional)", value="")
        setup_tag = st.text_input("Setup tag (e.g., pullback, oversold, breakout)", value="")
    notes = st.text_area("Notes (reason for entry, context)")

    snapshot_id = ""
    if st.session_state.last_snapshot:
        snapshot_id = f"snapshot_{st.session_state.last_snapshot['generated_at']}"

    submitted = st.form_submit_button("➕ Add to Log")
    if submitted:
        trade_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}_{symbol.replace('/','')}"
        st.session_state.trades = tlog.new_trade(
            st.session_state.trades,
            trade_id=trade_id,
            snapshot_id=snapshot_id,
            time_open=time_open, symbol=symbol, side=side,
            entry_plan_entry=entry_plan_entry, entry_plan_sl=entry_plan_sl, entry_plan_tp=entry_plan_tp, size=size,
            time_fill=time_fill, price_fill=price_fill,
            time_exit="", price_exit="", status="open",
            pnl_usd="", hold_minutes="", setup_tag=setup_tag, notes=notes
        )
        st.success("Trade added.")

st.subheader("Current Trade Log")
if not st.session_state.trades.empty:
    st.dataframe(st.session_state.trades, use_container_width=True)
    st.download_button("⬇️ Download trade_log.csv", data=tlog.to_csv_bytes(st.session_state.trades),
                       file_name="trade_log.csv", mime="text/csv")
else:
    st.info("No trades logged yet.")

# Update exits
with st.expander("Close / Update a Trade"):
    if st.session_state.trades.empty:
        st.write("No trades to update.")
    else:
        idx = st.number_input("Row index to update", min_value=0, max_value=len(st.session_state.trades)-1, value=0)
        colu1, colu2, colu3 = st.columns(3)
        with colu1:
            time_exit = st.text_input("Exit time (UTC)", value="")
            price_exit = st.text_input("Exit price", value="")
        with colu2:
            status = st.selectbox("Status", ["tp_hit","sl_hit","manual_exit","cancelled","expired"], index=2)
            pnl_usd = st.text_input("PnL USD (optional)", value="")
        with colu3:
            hold_minutes = st.text_input("Hold minutes (optional)", value="")
        if st.button("💾 Save update"):
            df = st.session_state.trades.copy()
            if time_exit: df.loc[idx, "time_exit"] = time_exit
            if price_exit: df.loc[idx, "price_exit"] = price_exit
            if status: df.loc[idx, "status"] = status
            if pnl_usd: df.loc[idx, "pnl_usd"] = pnl_usd
            if hold_minutes: df.loc[idx, "hold_minutes"] = hold_minutes
            st.session_state.trades = df
            st.success("Trade updated.")
