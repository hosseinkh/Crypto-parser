# ---- Trade Log (simple, file-based) ----
import io
import pandas as pd
import streamlit as st

st.markdown("---")
st.header("🧾 Trade Log")

if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame(
        columns=[
            "timestamp_open", "symbol", "side", "entry",
            "stop", "take_profit", "size", "notes",
            "timestamp_close", "exit", "pnl"
        ]
    )

with st.expander("View / Update Trade Log", expanded=True):
    up = st.file_uploader("Upload existing trade_log.csv (optional)", type=["csv"])
    if up is not None:
        try:
            st.session_state.trades = pd.read_csv(up)
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
            open_ts, sym, side, entry, stop, take, size, notes, "", "", ""
        ]
        st.success("Added.")

    st.download_button(
        "⬇️ Download trade_log.csv",
        data=st.session_state.trades.to_csv(index=False).encode("utf-8"),
        file_name="trade_log.csv",
        mime="text/csv",
    )
