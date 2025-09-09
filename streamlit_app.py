# streamlit_app.py
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utiles.bitget import (
    list_symbols_bitget,
    klines_bitget,
    TIMEFRAME_MAP,
    now_utc_iso,
)
from utiles.ticks import augment_with_ticks, augment_many_with_ticks

APP_TITLE = "Professional Snapshot Builder"
DEFAULT_TIMEFRAME = "15m"
CANDLE_LIMIT = 240
MAX_SCAN_SYMBOLS = 120

# ===== Indicators =====
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    def sma(x, n):
        return pd.Series(x).rolling(n).mean()

    def rsi(x, n=14):
        x = pd.Series(x)
        delta = x.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(n).mean()
        ma_down = down.rolling(n).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out

    def atr(h, l, c, n=14):
        c_shift = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - c_shift).abs(), (l - c_shift).abs()], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    df["ma20"] = sma(close, 20)
    df["ma50"] = sma(close, 50)
    df["ma200"] = sma(close, 200)
    df["rsi14"] = rsi(close, 14)
    df["atr14"] = atr(high, low, close, 14)

    vol_ma = vol.rolling(50).mean()
    vol_sd = vol.rolling(50).std(ddof=0)
    df["vol_z"] = (vol - vol_ma) / vol_sd.replace(0, np.nan)
    return df

def last_price(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1]) if not df.empty else float("nan")

def trend_score(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    if df.empty or len(df) < 210:
        return 0.0, {}
    last = df.iloc[-1]; prev = df.iloc[-2]
    structure = 1.0 if (last["close"] > last["ma20"] > last["ma50"]) else 0.0
    slope_raw = 0.0
    if pd.notna(prev["ma20"]) and prev["ma20"] != 0:
        slope_raw = (last["ma20"] - prev["ma20"]) / prev["ma20"]
    slope20 = max(0.0, min(1.0, float(slope_raw) * 100))
    rsi = float(last["rsi14"]) if pd.notna(last["rsi14"]) else float("nan")
    if math.isnan(rsi):
        rsi_part = 0.0
    elif rsi < 45: rsi_part = 0.2
    elif 45 <= rsi <= 65: rsi_part = 1.0
    elif 65 < rsi <= 72: rsi_part = 0.6
    else: rsi_part = 0.2
    volz = float(last["vol_z"]) if pd.notna(last["vol_z"]) else float("nan")
    vol_part = 1.0 if (not math.isnan(volz) and volz > 0) else 0.4
    prem = 0.0
    if pd.notna(last["ma20"]) and last["ma20"] != 0:
        prem = (last["close"] - last["ma20"]) / last["ma20"]
    prem_part = 1.0 if 0 < prem < 0.02 else 0.5 if prem >= 0.02 else 0.2
    score = (0.30*structure + 0.20*slope20 + 0.25*rsi_part + 0.15*vol_part + 0.10*prem_part) * 100.0
    explain = {
        "structure": round(structure, 3),
        "slope20": round(slope20, 3),
        "rsi14": round(rsi, 2) if not math.isnan(rsi) else float("nan"),
        "vol_z": round(volz, 2) if not math.isnan(volz) else float("nan"),
        "premium_to_ma20": round(prem, 4),
    }
    return float(score), explain

def sentiment_boost(symbol: str) -> float:
    return 0.5

# ===== State =====
def init_state():
    if "symbols" not in st.session_state:
        st.session_state.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    if "universe" not in st.session_state:
        st.session_state.universe = []
    if "last_universe_refresh" not in st.session_state:
        st.session_state.last_universe_refresh = 0

# ===== UI Helpers =====
def sidebar_controls():
    st.sidebar.header("⚙️ Settings")
    quote = st.sidebar.selectbox("Quote", ["USDT", "USDC", "BTC"], index=0)
    timeframe = st.sidebar.selectbox("Timeframe (scan & snapshot)", ["15m", "1h", "4h"], index=0)
    max_scan = st.sidebar.slider("Max symbols to scan", 20, MAX_SCAN_SYMBOLS, 60, 10)
    top_k = st.sidebar.slider("Add top N trending", 5, 40, 15, 1)
    return quote, timeframe, max_scan, top_k

def refresh_universe(quote: str):
    with st.spinner("Fetching tradable symbols…"):
        uni = list_symbols_bitget(quote)
        st.session_state.universe = uni
        st.session_state.last_universe_refresh = time.time()

def ui_symbol_manager(quote: str):
    st.subheader("🧰 Symbol Manager")
    if not st.session_state.universe:
        refresh_universe(quote)
    c1, c2 = st.columns([3, 1])
    with c1:
        add_syms = st.multiselect(
            "Add symbols (type to search):",
            options=st.session_state.universe,
            default=[],
            help="Start typing, e.g., 'ARB/USDT'."
        )
    with c2:
        st.write("")
        if st.button("🔄 Refresh"):
            refresh_universe(quote)
    if st.button("➕ Add selected"):
        added = 0
        for s in add_syms:
            if s not in st.session_state.symbols:
                st.session_state.symbols.append(s); added += 1
        st.success(f"Added {added} symbols. Total: {len(st.session_state.symbols)}")
    if st.session_state.symbols:
        rem = st.multiselect("Remove symbols:", options=st.session_state.symbols, default=[])
        if st.button("🗑️ Remove selected"):
            st.session_state.symbols = [s for s in st.session_state.symbols if s not in rem]
            st.success("Removed.")
    st.caption(f"Current list: {', '.join(st.session_state.symbols)}")

def scan_trending(universe: List[str], timeframe: str, cap: int) -> pd.DataFrame:
    if not universe:
        return pd.DataFrame()
    uni = universe[:cap]
    rows = []
    prog = st.progress(0.0, text="Scanning market…")
    for i, sym in enumerate(uni, 1):
        try:
            df = klines_bitget(sym, timeframe, CANDLE_LIMIT)
            if df.empty:
                continue
            df = add_indicators(df)
            score_tech, explain = trend_score(df)
            sboost = sentiment_boost(sym)
            final = 0.8*score_tech + 20*sboost
            rows.append({
                "symbol": sym,
                "last": last_price(df),
                "score": round(final, 2),
                "tech_score": round(score_tech, 2),
                "sentiment": round(sboost, 2),
                **explain
            })
        except Exception:
            pass
        if i % 5 == 0 or i == len(uni):
            prog.progress(min(1.0, i/len(uni)))
    prog.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "tech_score"], ascending=False).reset_index(drop=True)

def ui_scan_and_add(timeframe: str, max_scan: int, top_k: int):
    st.subheader("📈 Scan Trending")
    st.caption("Ranks symbols (tech + sentiment stub). Auto-adds top results to your list if you want.")
    c1, c2 = st.columns([1, 1])
    with c1:
        run = st.button("🚀 Scan Trending Now")
    with c2:
        auto_add = st.checkbox("Auto-add top results", value=True)
    if run:
        df = scan_trending(st.session_state.universe, timeframe, max_scan)
        if df.empty:
            st.warning("No results. Try increasing Max symbols to scan or change timeframe."); return
        st.dataframe(df.head(top_k), use_container_width=True)
        if auto_add:
            candidates = list(df.head(top_k)["symbol"].values); added = 0
            for s in candidates:
                if s not in st.session_state.symbols:
                    st.session_state.symbols.append(s); added += 1
            st.success(f"Added {added} symbols to your list.")

# ===== Snapshot build & pack =====
def pack_snapshot_from_rows(rows: List[Dict]) -> Dict:
    return {
        "meta": {
            "built_at": now_utc_iso(),
            "timeframe": DEFAULT_TIMEFRAME,
            "note": "Closed-candle indicators; per-symbol 'tick' is attached at download time."
        },
        "symbols": rows,
    }

def build_snapshot(symbols: List[str], timeframe: str) -> tuple[Dict, pd.DataFrame]:
    table = []; rows_for_json = []
    if not symbols:
        return {"meta": {}, "symbols": []}, pd.DataFrame()
    prog = st.progress(0.0, text="Building snapshot…")
    for i, sym in enumerate(symbols, 1):
        try:
            df = klines_bitget(sym, timeframe, CANDLE_LIMIT)
            df = add_indicators(df)
            last = df.iloc[-1] if not df.empty else None
            rows_for_json.append({
                "symbol": sym,
                "tf": timeframe,
                "last_closed": {
                    "t": str(df["time"].iloc[-1]) if not df.empty else "",
                    "o": float(df["open"].iloc[-1]) if not df.empty else float("nan"),
                    "h": float(df["high"].iloc[-1]) if not df.empty else float("nan"),
                    "l": float(df["low"].iloc[-1]) if not df.empty else float("nan"),
                    "c": float(df["close"].iloc[-1]) if not df.empty else float("nan"),
                    "v": float(df["volume"].iloc[-1]) if not df.empty else float("nan"),
                },
                "indicators": {
                    "ma20": float(last["ma20"]) if last is not None else float("nan"),
                    "ma50": float(last["ma50"]) if last is not None else float("nan"),
                    "ma200": float(last["ma200"]) if last is not None else float("nan"),
                    "rsi14": float(last["rsi14"]) if last is not None else float("nan"),
                    "atr14": float(last["atr14"]) if last is not None else float("nan"),
                    "vol_z": float(last["vol_z"]) if last is not None else float("nan"),
                }
            })
            table.append({
                "symbol": sym,
                "last_close": float(df["close"].iloc[-1]) if not df.empty else float("nan"),
                "rsi14": round(float(last["rsi14"]), 2) if last is not None else float("nan"),
                "ma20": round(float(last["ma20"]), 6) if last is not None else float("nan"),
                "ma50": round(float(last["ma50"]), 6) if last is not None else float("nan"),
                "ma200": round(float(last["ma200"]), 6) if last is not None else float("nan"),
                "atr14": round(float(last["atr14"]), 6) if last is not None else float("nan"),
                "vol_z": round(float(last["vol_z"]), 2) if last is not None else float("nan"),
            })
        except Exception as e:
            table.append({"symbol": sym, "error": str(e)})
        prog.progress(min(1.0, i/len(symbols)))
    prog.empty()
    packed = pack_snapshot_from_rows(rows_for_json)
    return packed, pd.DataFrame(table)

# ===== App =====
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    init_state()
    st.title(APP_TITLE)
    st.caption("Type to add symbols • Scan trending • Build a JSON snapshot that includes the **last price at snapshot time** for each coin.")

    quote, timeframe, max_scan, top_k = sidebar_controls()
    ui_symbol_manager(quote); st.divider()
    ui_scan_and_add(timeframe, max_scan, top_k); st.divider()

    st.subheader("🧪 Build Snapshot")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        split_output = st.checkbox("Split output", value=False)
    with c2:
        max_per_file = st.number_input("Max symbols per file", min_value=10, max_value=500, value=120, step=10)
    with c3:
        build_btn = st.button("🧱 Build Snapshot", use_container_width=True)

    if build_btn:
        if not st.session_state.symbols:
            st.warning("Your symbol list is empty. Add symbols first."); return
        packed, df = build_snapshot(st.session_state.symbols, timeframe)

        # Preview
        try: st.dataframe(df, use_container_width=True)
        except Exception: st.table(df)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if split_output:
            # chunking
            parts = []; current = []
            for i, row in enumerate(packed["symbols"], 1):
                current.append(row)
                if len(current) >= int(max_per_file) or i == len(packed["symbols"]):
                    parts.append({"meta": packed["meta"], "symbols": current}); current = []
            # attach live ticks (captures last price at download time)
            parts = augment_many_with_ticks(parts)
            st.write(f"Creating **{len(parts)}** JSON files (max {int(max_per_file)} symbols per file).")
            for idx, part in enumerate(parts, start=1):
                as_bytes = json.dumps(part, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                st.download_button(
                    f"⬇️ Download JSON #{idx}", data=as_bytes,
                    file_name=f"snapshot_{ts}_{idx:02d}.json", mime="application/json", key=f"dl_{idx}",
                )
            with st.expander("🔍 Preview first JSON"):
                st.code(json.dumps(parts[0], indent=2, ensure_ascii=False), language="json")
        else:
            # single file; attach live ticks here too
            packed = augment_with_ticks(packed)
            as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            st.download_button(
                "⬇️ Download single JSON snapshot",
                data=as_bytes, file_name=f"snapshot_{ts}.json", mime="application/json", use_container_width=True,
            )
            with st.expander("🔍 View JSON"):
                st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")

if __name__ == "__main__":
    main()
