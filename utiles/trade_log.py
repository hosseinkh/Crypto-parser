# ---- Trade Log (simple, file-based) ----
from __future__ import annotations
import io, json
import pandas as pd
import streamlit as st
from typing import Dict, Any

st.markdown("---")
st.header("🧾 Trade Log")

# -----------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------
REQUIRED_COLS = [
    "timestamp_open", "symbol", "side", "entry",
    "stop", "take_profit", "size", "notes",
    "timestamp_close", "exit", "pnl",
]

OPTIONAL_COLS = [
    # Early Partial (L0) + Exit state + BTC context
    "early_partial_taken",           # bool
    "early_partial_pct",             # float (e.g., 0.35 for 35% partial)
    "early_partial_ts",              # ISO time when partial was taken
    "early_partial_metrics",         # json string with snapshot-derived numbers
    "exit_matrix_state",             # GREEN/YELLOW/ORANGE/RED at the moment you updated
    "btc_bear_bias",                 # bool at the moment you updated (from snapshot.market.btc_anchor_bias.bear)
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS + OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = "" if c not in {"early_partial_taken", "early_partial_pct", "btc_bear_bias"} else (False if c in {"early_partial_taken", "btc_bear_bias"} else 0.0)
    # Coerce types for new fields
    if "early_partial_taken" in df.columns:
        df["early_partial_taken"] = df["early_partial_taken"].astype(bool, errors="ignore")
    if "early_partial_pct" in df.columns:
        try:
            df["early_partial_pct"] = pd.to_numeric(df["early_partial_pct"], errors="coerce")
        except Exception:
            pass
    if "btc_bear_bias" in df.columns:
        df["btc_bear_bias"] = df["btc_bear_bias"].astype(bool, errors="ignore")
    return df

def load_snapshot_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {}

# -----------------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------------
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=REQUIRED_COLS)

st.session_state.trades = ensure_columns(st.session_state.trades)

# -----------------------------------------------------------------------------------
# View / Update
# -----------------------------------------------------------------------------------
with st.expander("View / Update Trade Log", expanded=True):
    up = st.file_uploader("Upload existing trade_log.csv (optional)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.session_state.trades = ensure_columns(df)
            st.success("Trade log loaded.")
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")

    st.dataframe(st.session_state.trades, use_container_width=True)

    st.subheader("Add a trade")
    c1, c2, c3 = st.columns(3)
    with c1:
        sym = st.text_input("Symbol", value="BTC/USDT")
        side = st.selectbox("Side", ["LONG", "SHORT"])
        size = st.number_input("Size", min_value=0.0, value=100.0, step=10.0)
    with c2:
        entry = st.number_input("Entry", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        stop = st.number_input("Stop", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        take = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    with c3:
        open_ts = st.text_input("Open time (UTC ISO)", value="")
        notes = st.text_area("Notes", value="", height=80)

    if st.button("Add to log"):
        st.session_state.trades.loc[len(st.session_state.trades)] = [
            open_ts, sym, side, entry, stop, take, size, notes, "", "", "",
            False, 0.0, "", "", "", False
        ]
        st.success("Added.")

    st.download_button(
        "⬇️ Download trade_log.csv",
        data=st.session_state.trades.to_csv(index=False).encode("utf-8"),
        file_name="trade_log.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------------
# New: Early Partial / Exit State Recorder (minimal & optional)
# -----------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚡ Record Early Partial (L0) / Exit Matrix State")

# Choose a row to update
row_count = len(st.session_state.trades)
if row_count == 0:
    st.info("Add a trade first to record early partials or exit state.")
else:
    idx = st.number_input("Row index to update", min_value=0, max_value=max(0, row_count - 1), value=0, step=1)
    current_row = st.session_state.trades.iloc[idx] if row_count > 0 else None
    if current_row is not None:
        st.caption(f"Selected: #{idx} {current_row.get('symbol','')} {current_row.get('side','')} opened {current_row.get('timestamp_open','')}")

    colA, colB = st.columns(2)

    with colA:
        st.write("**Mark early partial (optional)**")
        ep_taken = st.checkbox("Early partial taken", value=bool(current_row.get("early_partial_taken", False)))
        ep_pct = st.slider("Partial size (%)", min_value=0.0, max_value=100.0, value=float(current_row.get("early_partial_pct") or 30.0), step=1.0)
        ep_ts = st.text_input("Early partial time (UTC ISO)", value=str(current_row.get("early_partial_ts") or ""))

    with colB:
        st.write("**Exit matrix & BTC context (optional)**")
        exit_state = st.selectbox(
            "Exit matrix state",
            ["", "GREEN", "YELLOW", "ORANGE", "RED"],
            index=0 if not current_row.get("exit_matrix_state") else ["","GREEN","YELLOW","ORANGE","RED"].index(str(current_row.get("exit_matrix_state"))),
        )
        btc_bear = st.checkbox("BTC bear bias", value=bool(current_row.get("btc_bear_bias", False)))

    # Optional: load snapshot.json to auto-fill early-partial metrics
    snap_col1, snap_col2 = st.columns([2,1])
    with snap_col1:
        st.write("**(Optional) Load snapshot.json to capture metrics**")
        snap_file = st.file_uploader("Upload snapshot.json", type=["json"], key="snap_uploader")
    with snap_col2:
        symbol_for_metrics = st.text_input("Symbol (match snapshot key)", value=str(current_row.get("symbol") or ""))

    ep_metrics_out = None
    if snap_file is not None and symbol_for_metrics:
        try:
            snap_text = snap_file.read().decode("utf-8")
            snap = load_snapshot_json(snap_text)
            # Navigate: items[symbol]
            item = (snap.get("items") or {}).get(symbol_for_metrics)
            if item:
                derived = (item.get("derived") or {})
                early_partial_block = derived.get("early_partial", {})
                market = snap.get("market") or {}
                btc_anchor = market.get("btc_anchor_bias") or {}
                ep_metrics_out = {
                    "profit_pct": early_partial_block.get("profit_pct"),
                    "atr_gain_multiple_since_entry": early_partial_block.get("atr_gain_multiple_since_entry"),
                    "vpr10_drop_pct_from_peak_last3": early_partial_block.get("vpr10_drop_pct_from_peak_last3"),
                    "derived_early_partial_flag": (True if "enabled" in early_partial_block else None),
                }
                # If user didn’t tick BTC bear bias manually, offer snapshot suggestion
                if not btc_bear and isinstance(btc_anchor.get("bear"), bool):
                    btc_bear = btc_anchor.get("bear")
                st.success("Loaded early-partial metrics from snapshot.")
            else:
                st.warning("Symbol not found in snapshot.")
        except Exception as e:
            st.warning(f"Could not parse snapshot: {e}")

    if st.button("Update selected row"):
        st.session_state.trades.at[idx, "early_partial_taken"] = bool(ep_taken)
        st.session_state.trades.at[idx, "early_partial_pct"] = float(ep_pct)
        st.session_state.trades.at[idx, "early_partial_ts"] = ep_ts
        st.session_state.trades.at[idx, "exit_matrix_state"] = exit_state
        st.session_state.trades.at[idx, "btc_bear_bias"] = bool(btc_bear)
        if ep_metrics_out is not None:
            st.session_state.trades.at[idx, "early_partial_metrics"] = json.dumps(ep_metrics_out, ensure_ascii=False)
        st.success("Row updated.")
        st.dataframe(st.session_state.trades, use_container_width=True)
