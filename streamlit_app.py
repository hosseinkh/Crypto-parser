# streamlit_app.py
# -----------------------------------------------------------
# Watchlist (favorites incl. GALA/XLM)
# Trending scan (adds only Top-N with chosen ranking)
# Build Snapshot v4.1 (summary + download, raw JSON in expander)
# -----------------------------------------------------------

from __future__ import annotations
import os, sys, importlib, json
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from dataclasses import dataclass

# --- Package bootstrap ---
ROOT_DIR = os.path.dirname(__file__)
UTIL_DIR  = os.path.join(ROOT_DIR, "utiles")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.makedirs(UTIL_DIR, exist_ok=True)
init_path = os.path.join(UTIL_DIR, "__init__.py")
if not os.path.exists(init_path):
    open(init_path, "a").close()

# --- Exchange helper (module import + safe fallback) ---
try:
    bitget_mod = importlib.import_module("utiles.bitget")
except Exception as e:
    st.error(f"Failed to import utiles.bitget: {e}")
    st.stop()

def _fallback_make_exchange(name: str = "bitget"):
    try:
        import ccxt  # type: ignore
        name = (name or "bitget").lower()
        cls = getattr(ccxt, name)
        ex = cls({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})
        class _Wrap:
            def __init__(self, ex):
                self.raw = ex
                try:
                    self.raw.load_markets(reload=False)
                except Exception:
                    self.raw.load_markets(reload=True)
                self.symbols = sorted(
                    set(m["symbol"] for m in self.raw.markets.values()
                        if m.get("spot") and m.get("active") and m.get("quote") == "USDT")
                )
            def fetch_ohlcv(self, *a, **k):
                return self.raw.fetch_ohlcv(*a, **k)
        return _Wrap(ex)
    except Exception as e:
        st.error(f"Fallback make_exchange failed: {e}")
        st.stop()

make_exchange = getattr(bitget_mod, "make_exchange", _fallback_make_exchange)

# --- Trending module ---
try:
    trending_mod = importlib.import_module("utiles.trending")
except Exception as e:
    st.error(f"Failed to import utiles.trending: {e}")
    st.stop()

scan_trending        = getattr(trending_mod, "scan_trending", None)
explain_trending_row = getattr(trending_mod, "explain_trending_row", None)
TrendScanParams      = getattr(trending_mod, "TrendScanParams", None)

if TrendScanParams is None:
    @dataclass
    class TrendScanParams:
        min_pct_4h: float = 0.5
        min_volz_24h: float = 0.8
        rsi15m_min: float = 35.0
        rsi15m_max: float = 65.0
        min_sentiment: float = 0.05
        near_support_max_pct: float = 2.5
        near_resist_max_pct: float = 1.5

if explain_trending_row is None:
    def explain_trending_row(row: pd.Series, p: TrendScanParams) -> List[str]:
        reasons: List[str] = []
        if pd.notna(row.get("pct4h")) and row["pct4h"] >= p.min_pct_4h:
            reasons.append(f"4h momentum +{row['pct4h']:.2f}%")
        if pd.notna(row.get("vol_z24h")) and row["vol_z24h"] >= p.min_volz_24h:
            reasons.append(f"24h volume spike (z={row['vol_z24h']:.2f})")
        rsi = row.get("rsi14_15m")
        if pd.notna(rsi) and p.rsi15m_min <= rsi <= p.rsi15m_max:
            reasons.append(f"15m RSI in range ({rsi:.1f})")
        dlow = row.get("dist_to_low_pct")
        if pd.notna(dlow) and dlow <= p.near_support_max_pct:
            reasons.append(f"near support ({dlow:.2f}% from low)")
        dhigh = row.get("dist_to_high_pct")
        if pd.notna(dhigh) and dhigh <= p.near_resist_max_pct:
            reasons.append(f"near resistance ({dhigh:.2f}% from high)")
        return reasons

if scan_trending is None:
    st.error("utiles.trending is missing: scan_trending")
    st.stop()

# --- Snapshot module (v4.1) ---
try:
    snap_mod = importlib.import_module("utiles.snapshot")
except Exception as e:
    st.error(f"Could not import snapshot module: {e}")
    snap_mod = None

# --- Favorites / watchlist ---
ALWAYS_INCLUDE = {"GALA/USDT", "XLM/USDT"}
DEFAULT_FAVORITES = {
    "BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","XRP/USDT",
    "LINK/USDT","AVAX/USDT","TRX/USDT","DOT/USDT","LTC/USDT",
    "INJ/USDT","GRT/USDT","CRO/USDT","XTZ/USDT","FET/USDT",
}

def _init_watchlist(exchange):
    available = {s for s in exchange.symbols if s.endswith("/USDT")}
    base = (DEFAULT_FAVORITES | ALWAYS_INCLUDE) & available
    if "working_symbols" not in st.session_state or not st.session_state["working_symbols"]:
        st.session_state["working_symbols"] = sorted(base)

# --- Page ---
st.set_page_config(page_title="Crypto Parser — LLM Snapshot", layout="wide")
st.title("Crypto Parser — Watchlist, Trending & LLM Snapshot (v4.1)")

ex = make_exchange("bitget")
_ = ex.symbols
_init_watchlist(ex)

colL, colR = st.columns([2, 1])
with colL:
    st.subheader("📌 Current list")
    st.write(", ".join(st.session_state["working_symbols"]))

with colR:
    st.subheader("➕ Add manually")
    universe = sorted([s for s in ex.symbols if s.endswith("/USDT")])
    add_symbol = st.selectbox("Add symbol", [""] + universe, index=0)
    if add_symbol and add_symbol not in st.session_state["working_symbols"]:
        st.session_state["working_symbols"].append(add_symbol)
        st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
        st.success(f"Added {add_symbol}")

st.markdown("---")
st.subheader("🔥 Find trending & explain why (Top-N)")

# New controls: how many to add & ranking metric
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    top_k = st.number_input("Top N to add", min_value=1, max_value=50, value=10, step=1)
with colB:
    rank_by = st.selectbox(
        "Rank by",
        ["Score (momentum+volume+sentiment)", "Volume z-score (24h)", "4h momentum %"],
        index=0
    )
with colC:
    st.caption("Only the top N candidates (by the chosen ranking) will be added to your list.")

params = TrendScanParams()

def _rank_df(df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    if df.empty:
        return df
    if rank_by.startswith("Score"):
        key = "score"
    elif rank_by.startswith("Volume"):
        key = "vol_z24h"
    else:
        key = "pct4h"
    # sort descending, NaNs last
    return df.sort_values(key, ascending=False, na_position="last")

if st.button("Scan market and add Top-N with reasons"):
    try:
        df = scan_trending(ex, params=params)
        if df.empty:
            st.info("No trending candidates met the thresholds.")
        else:
            ranked = _rank_df(df, rank_by)
            top = ranked.head(int(top_k)).copy()

            st.dataframe(top[[
                "symbol","score","pct4h","vol_z24h","sentiment_score",
                "rsi14_15m","dist_to_low_pct","dist_to_high_pct"
            ]])

            added = []
            for _, row in top.iterrows():
                sym = row["symbol"]
                if sym not in st.session_state["working_symbols"]:
                    st.session_state["working_symbols"].append(sym)
                    reasons = explain_trending_row(row, params)
                    added.append((sym, reasons))

            st.session_state["working_symbols"] = sorted(set(st.session_state["working_symbols"]))
            if added:
                st.success(f"Added Top-{len(added)} symbols:")
                for sym, reasons in added:
                    st.write(f"• **{sym}** — " + "; ".join(reasons))
            else:
                st.info("Top-N symbols were already in your list.")
    except Exception as e:
        st.error(f"Scan failed: {e}")

st.markdown("---")
st.subheader("📷 Build snapshot (v4.1)")

# Snapshot controls
snap_tf = st.selectbox("Timeframe", ["15m","5m","30m","1h"], index=0)
snap_limit = st.number_input("Candles limit", min_value=120, max_value=1000, value=240, step=10)

if snap_mod and st.button("Build snapshot now"):
    try:
        snap = snap_mod.build_snapshot_v41(
            ex, st.session_state["working_symbols"],
            timeframe=snap_tf, limit=int(snap_limit), quote_asset="USDT", exchange_name="bitget"
        )

        # Download FIRST (no scrolling)
        fname = f"snapshot_{snap.get('schema_version','4.1')}_{snap.get('generated_at_utc','').replace(':','').replace('.','')}.json"
        st.download_button("⬇️ Download snapshot JSON", data=json.dumps(snap, indent=2), file_name=fname, mime="application/json")

        # Show concise SUMMARY table for quick read
        if isinstance(snap, dict) and "items" in snap:
            rows: List[Dict[str, Any]] = []
            for it in snap["items"]:
                sym = it.get("symbol")
                feat = it.get("features", {})
                pb  = it.get("playbook_eval", {})
                bk  = pb.get("breakout", {}) if isinstance(pb, dict) else {}
                mr  = pb.get("mean_reversion", {}) if isinstance(pb, dict) else {}
                rows.append({
                    "symbol": sym,
                    "final": pb.get("final_recommendation"),
                    "bk_status": bk.get("status"),
                    "bk_score": bk.get("score"),
                    "mr_status": mr.get("status"),
                    "mr_score": mr.get("score"),
                    "vol_z24h": feat.get("vol_z_24h"),
                    "rsi_15m": feat.get("rsi_15m"),
                    "dist_to_high%": feat.get("dist_to_high_pct"),
                    "dist_to_low%": feat.get("dist_to_low_pct"),
                    "spread%": feat.get("spread_pct"),
                })
            if rows:
                df_sum = pd.DataFrame(rows).sort_values(
                    ["final","bk_status","mr_status","bk_score","mr_score"],
                    ascending=[True, True, True, False, False],
                    na_position="last"
                )
                st.success(f"Snapshot built: {len(rows)} symbols")
                st.dataframe(df_sum, use_container_width=True)

        # Raw JSON inside an expander (no more long scroll)
        with st.expander("🔎 Raw snapshot JSON (expand to view)"):
            st.json(snap)

    except Exception as e:
        st.error(f"Snapshot build failed: {e}")
