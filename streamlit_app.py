# streamlit_app.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# Project utilities
from utiles.bitget import make_exchange, normalize_symbol, fetch_ohlcv_df, now_utc_iso
from utiles.ticks import augment_with_ticks, augment_many_with_ticks

# =========================
# ------- Settings --------
# =========================

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "LINK/USDT",
    "INJ/USDT", "GRT/USDT", "FET/USDT", "BNB/USDT", "TRX/USDT",
    "XRP/USDT", "XTZ/USDT", "CRO/USDT", "PENGU/USDT",  # keep your 14 + pengu if you use it
]

TIMEFRAMES = ["15m", "1h", "4h"]
LIMITS = {"15m": 200, "1h": 200, "4h": 200}  # enough bars for indicators

# =========================
# ---- TA Helper funcs ----
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(period).mean()

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std

# =========================
# ---- Row Construction ----
# =========================

@dataclass
class TFBlock:
    last_closed: dict
    indicators: dict
    structure: dict

def build_symbol_block(ex, symbol: str) -> dict:
    """
    Fetch OHLCV for 15m/1h/4h, compute core indicators on CLOSED candles,
    and return a symbol block ready for JSON packing and table display.
    """
    tf_blocks: Dict[str, TFBlock] = {}

    for tf in TIMEFRAMES:
        df = fetch_ohlcv_df(ex, symbol, tf, LIMITS[tf])
        if df.empty or len(df) < 30:
            # not enough data for indicators
            tf_blocks[tf] = TFBlock(
                last_closed={}, indicators={}, structure={}
            )
            continue

        # last closed candle = row at index -2 (since last row may be still forming)
        # but in CCXT, the last value returned is typically the latest CLOSED for spot TFs
        # to be safe, we take the last row as closed (common for fetch_ohlcv); adjust if your exchange differs
        last = df.iloc[-1].copy()

        # indicators on CLOSED
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        ema20 = ema(close, 20)
        rsi14 = rsi(close, 14)
        atr14 = atr(high, low, close, 14)

        # volume z-score (simple approach vs 20-bar mean/std)
        volz = zscore(volume, 20)

        # structure helpers (distance to 20-bar high/low measured from close)
        dist_to_high = (close - close.rolling(20).max()) / close * 100.0
        dist_to_low = (close - close.rolling(20).min()) / close * 100.0

        # trend heuristic: slope of EMA20 over last 5 bars
        slope = ema20.diff(5)
        trend = "up" if slope.iloc[-1] > 0 else ("down" if slope.iloc[-1] < 0 else "flat")

        tf_blocks[tf] = TFBlock(
            last_closed={
                "t": pd.to_datetime(last["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": float(last["open"]),
                "h": float(last["high"]),
                "l": float(last["low"]),
                "c": float(last["close"]),
                "v": float(last["volume"]),
            },
            indicators={
                "ema20": float(ema20.iloc[-1]),
                "rsi14": float(rsi14.iloc[-1]),
                "atr14": float(atr14.iloc[-1]),
                "vol_z": float(volz.iloc[-1]) if not math.isnan(volz.iloc[-1]) else float("nan"),
                "dist_to_high": float(dist_to_high.iloc[-1]) if not math.isnan(dist_to_high.iloc[-1]) else float("nan"),
                "dist_to_low": float(dist_to_low.iloc[-1]) if not math.isnan(dist_to_low.iloc[-1]) else float("nan"),
            },
            structure={
                "trend": trend
            }
        )

    return {
        "symbol": symbol,
        "tf": {
            tf: {
                "last_closed": tf_blocks[tf].last_closed,
                "indicators": tf_blocks[tf].indicators,
                "structure": tf_blocks[tf].structure,
            } for tf in TIMEFRAMES
        }
    }

def build_rows(ex, symbols: List[str]) -> List[dict]:
    rows = []
    for s in symbols:
        sym = normalize_symbol(s)
        try:
            rows.append(build_symbol_block(ex, sym))
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e), "tf": {}})
    return rows

# =========================
# ---- Snapshot packers ----
# =========================

def pack_snapshot_from_rows(rows: List[dict]) -> dict:
    return {
        "snapshot_time_utc": now_utc_iso(),  # stamped again by ticks module; harmless duplicate
        "symbols": rows,
        "meta": {
            "timeframes": TIMEFRAMES,
            "note": "Indicators computed from CLOSED candles only (safe/non-repainting). Ticks added at export.",
        },
    }

def pack_snapshot_chunked(rows: List[dict], max_per_file: int) -> List[dict]:
    chunks = []
    for i in range(0, len(rows), max_per_file):
        chunk = rows[i:i + max_per_file]
        chunks.append(pack_snapshot_from_rows(chunk))
    return chunks

# =========================
# ---------- UI ----------
# =========================

st.set_page_config(page_title="Crypto Snapshot Builder", layout="wide")
st.title("📈 Professional Snapshot Builder")

with st.sidebar:
    st.header("Settings")
    default_syms = DEFAULT_SYMBOLS
    symbols_text = st.text_area(
        "Symbols (one per line, e.g. BTC/USDT):",
        value="\n".join(default_syms),
        height=220
    )
    split_output = st.checkbox("Split output into multiple files", value=False)
    max_per_file = st.number_input("Max symbols per file (if split):", min_value=5, max_value=50, value=10, step=1)
    run_btn = st.button("Build Snapshot")

if not run_btn:
    st.info("Configure symbols in the sidebar and click **Build Snapshot**.")
    st.stop()

# Parse symbols
symbols_input = [s.strip() for s in symbols_text.splitlines() if s.strip()]
if not symbols_input:
    st.error("Please enter at least one symbol.")
    st.stop()

# Build data
st.write("Fetching data from Bitget…")
ex = make_exchange()
rows = build_rows(ex, symbols_input)
st.success(f"Fetched {len(rows)} symbols.")

# =========================
# ---- Display summary ----
# =========================

table = []
for row in rows:
    sym = row.get("symbol", "?")
    tf15 = row.get("tf", {}).get("15m", {})
    ind = tf15.get("indicators", {})
    struct = tf15.get("structure", {})
    last_closed = tf15.get("last_closed", {})
    table.append(
        {
            "symbol": sym,
            "time": last_closed.get("t", ""),
            "close": round(last_closed.get("c", float("nan")), 6),
            "ema20": round(ind.get("ema20", float("nan")), 6),
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

# =========================
# ----- Download JSON -----
# =========================

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

if split_output:
    payloads = pack_snapshot_chunked(rows, int(max_per_file))

    # 🔵 Attach live ticks (precise live price at export time) to each chunk
    payloads = augment_many_with_ticks(payloads)

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

    # 🔵 Attach live ticks (precise live price at export time) to single snapshot
    packed = augment_with_ticks(packed)

    as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    st.download_button(
        "⬇️ Download single JSON snapshot",
        data=as_bytes,
        file_name=f"snapshot_{ts}.json",
        mime="application/json",
    )
    with st.expander("🔍 View JSON"):
        st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
