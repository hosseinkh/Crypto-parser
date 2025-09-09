# streamlit_app.py
# Professional Snapshot Builder — type-ahead symbols, stable list, trending scan (tech + sentiment),
# safe multiselect defaults, snapshot-time last price capture, and SENTIMENT in snapshot.

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

# ========= Optional project sentiment =========
# Supported return shapes:
# - float -> 0.35
# - dict  -> {"score": 0.35, "label":"bullish"}
# - tuple -> (0.35, "bullish")
try:
    from utils.sentiment import get_symbol_sentiment as _get_symbol_sentiment
except Exception:
    _get_symbol_sentiment = None

def _normalize_sentiment(raw: Any) -> Tuple[float, str]:
    try:
        if isinstance(raw, (int, float)):
            s = float(raw)
            return s, ("bullish" if s > 0.2 else "bearish" if s < -0.2 else "neutral")
        if isinstance(raw, dict):
            s = float(raw.get("score", 0.0))
            lbl = str(raw.get("label", "neutral"))
            return s, lbl
        if isinstance(raw, (tuple, list)) and raw:
            s = float(raw[0])
            lbl = str(raw[1]) if len(raw) > 1 else ("bullish" if s > 0.2 else "bearish" if s < -0.2 else "neutral")
            return s, lbl
    except Exception:
        pass
    return 0.0, "neutral"

def get_sentiment_for_symbol(symbol: str) -> Tuple[float, str]:
    """Safe wrapper to call user's sentiment function if present."""
    if _get_symbol_sentiment is None:
        return 0.0, "neutral"
    try:
        base = symbol.split("/")[0]  # e.g., "BTC" from "BTC/USDT"
        return _normalize_sentiment(_get_symbol_sentiment(base))
    except Exception:
        return 0.0, "neutral"


# ========= Exchange via CCXT (self-contained) =======
try:
    import ccxt  # ensure in requirements.txt
except Exception as e:
    st.error(f"Missing dependency: ccxt ({e}). Add `ccxt` to requirements.txt.")
    st.stop()

EXCHANGES = {
    "Bitget": ccxt.bitget,
    "Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "OKX": ccxt.okx,
}

SAFE_STABLE = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "ADA/USDT","DOGE/USDT","DOT/USDT","AVAX/USDT","LINK/USDT",
    "MATIC/USDT","LTC/USDT","TRX/USDT","BCH/USDT","TON/USDT",
]

TIMEFRAMES = ["1m","5m","15m","1h","4h","1d"]


def make_exchange(name: str):
    klass = EXCHANGES.get(name)
    if not klass:
        raise ValueError(f"Unsupported exchange {name}")
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

@st.cache_data(show_spinner=False, ttl=300)
def load_usdt_symbols(ex_name: str) -> List[str]:
    ex = make_exchange(ex_name)
    syms = []
    for m in ex.markets.values():
        if not m.get("active", True):
            continue
        if (m.get("type") == "spot" or m.get("spot") is True) and (str(m.get("quote")).upper() == "USDT"):
            syms.append(m["symbol"])
    return sorted(list(dict.fromkeys(syms)))


def safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
    last = None
    for _ in range(2):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            last = e
    raise last

def safe_fetch_ticker(ex, symbol: str) -> Dict[str, Any]:
    last = None
    for _ in range(2):
        try:
            return ex.fetch_ticker(symbol)
        except Exception as e:
            last = e
    raise last


# ========= Indicators (compact) =========
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)

def pct_change(new: float, old: float) -> float:
    if old in (0, None) or new in (None,):
        return np.nan
    return (new - old) / old * 100.0


# ========= Scan (tech + sentiment) =========
def compute_scan_metrics(df: pd.DataFrame, tf: str) -> Dict[str, float]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    vol = df["volume"]

    bars_per_4h = 16 if tf.endswith("m") and tf != "60m" else (4 if tf.endswith("h") else 16)
    if len(close) > bars_per_4h:
        m4 = pct_change(float(close.iloc[-1]), float(close.iloc[-bars_per_4h-1]))
    else:
        m4 = np.nan

    bars_per_24h = 96 if tf.endswith("m") and tf != "60m" else (24 if tf.endswith("h") else 30)
    vol_window = vol.tail(min(len(vol), bars_per_24h))
    vz = float(zscore(vol_window, max(5, min(50, len(vol_window)))).iloc[-1]) if len(vol_window) >= 5 else 0.0

    r = float(rsi(close, 14).iloc[-1])
    return {"pct4h": m4, "vol_z24h": vz, "rsi14": r}

def scan_trending(ex, universe: List[str], timeframe: str,
                  min_pct_4h: float, min_vol_z: float,
                  rsi_bounds: Tuple[int,int],
                  min_sentiment: float) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    passing: List[str] = []
    low, high = rsi_bounds

    for sym in universe:
        try:
            ohlcv = safe_fetch_ohlcv(ex, sym, timeframe=timeframe, limit=400)
            if not ohlcv or len(ohlcv) < 60:
                continue
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            met = compute_scan_metrics(df, timeframe)
            last = float(df["close"].iloc[-1])

            s_score, s_label = get_sentiment_for_symbol(sym)

            row = {
                "symbol": sym,
                "last": last,
                "pct4h": round(met["pct4h"], 2) if met["pct4h"] == met["pct4h"] else np.nan,
                "vol_z24h": round(met["vol_z24h"], 2),
                "rsi14": round(met["rsi14"], 1),
                "sentiment_score": round(s_score, 2),
                "sentiment_label": s_label,
            }
            rows.append(row)

            tech_ok = (
                (met["pct4h"] == met["pct4h"] and met["pct4h"] >= min_pct_4h) and
                (met["vol_z24h"] >= min_vol_z) and
                (low <= met["rsi14"] <= high)
            )
            snt_ok = s_score >= min_sentiment

            if tech_ok and snt_ok:
                passing.append(sym)

        except Exception:
            continue

    df_out = pd.DataFrame(rows).sort_values(["pct4h","vol_z24h","sentiment_score"], ascending=False, na_position="last")
    return df_out, passing


# ========= Snapshot packing (with SENTIMENT) =========
def last_price_snapshot(ex, symbol: str, fallback_close: float) -> float:
    try:
        t = safe_fetch_ticker(ex, symbol)
        val = t.get("last") or t.get("close")
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    except Exception:
        pass
    return float(fallback_close)

def build_snapshot(ex_name: str, tf: str, symbols: List[str], limit: int) -> Dict[str, Any]:
    ex = make_exchange(ex_name)
    rows, errors = [], []
    s_scores: List[float] = []

    for sym in symbols:
        try:
            ohlcv = safe_fetch_ohlcv(ex, sym, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            lp = last_price_snapshot(ex, sym, float(df["close"].iloc[-1]))

            # indicators useful for framework
            rsi14 = float(np.round(rsi(df["close"], 14).iloc[-1], 2))
            atr14 = float(np.round((df["high"] - df["low"]).rolling(14).mean().iloc[-1], 8))
            look = min(100, df.shape[0])
            hh = float(df["high"].tail(look).max())
            ll = float(df["low"].tail(look).min())
            dist_to_high = float(np.round((hh - lp) / hh * 100, 4)) if hh else None
            dist_to_low = float(np.round((lp - ll) / ll * 100, 4)) if ll else None

            # per-symbol sentiment
            s_score, s_label = get_sentiment_for_symbol(sym)
            s_scores.append(s_score)

            rows.append({
                "symbol": sym,
                "timeframe": tf,
                "last": lp,  # true last at snapshot time
                "last_candle": {
                    "ts": int(df["timestamp"].iloc[-1]),
                    "open": float(df["open"].iloc[-1]),
                    "high": float(df["high"].iloc[-1]),
                    "low": float(df["low"].iloc[-1]),
                    "close": float(df["close"].iloc[-1]),
                    "volume": float(df["volume"].iloc[-1]),
                },
                "indicators": {
                    "rsi14": rsi14,
                    "atr14": atr14,
                    "dist_to_high": dist_to_high,
                    "dist_to_low": dist_to_low,
                },
                "sentiment": {
                    "score": float(np.round(s_score, 3)),
                    "label": s_label,
                },
                "asof": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        except Exception as e:
            errors.append((sym, str(e)))

    # meta-level sentiment summary (average)
    avg_sent = float(np.round(np.nanmean(s_scores), 3)) if s_scores else 0.0
    avg_label = "bullish" if avg_sent > 0.2 else "bearish" if avg_sent < -0.2 else "neutral"

    return {
        "meta": {
            "exchange": ex_name,
            "timeframe": tf,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "notes": "last = snapshot-time price (ticker.last if available, else last close)",
            "count": len(rows),
            "sentiment_summary": {"avg_score": avg_sent, "avg_label": avg_label},
        },
        "rows": rows,
        "errors": errors,
    }


# ========= UI =========
st.set_page_config(page_title="Professional Snapshot Builder", page_icon="📈", layout="wide")
st.title("📈 Professional Snapshot Builder")

with st.sidebar:
    st.header("Settings")
    ex_name = st.selectbox("Exchange", list(EXCHANGES.keys()), index=0)
    tf = st.selectbox("Timeframe", TIMEFRAMES, index=2)  # default 15m
    limit = st.slider("Candles per symbol", 100, 2000, 1200, 50)

    st.markdown("---")
    st.subheader("Symbols")

    # Load exchange symbols (may be empty on API hiccups)
    try:
        all_usdt = load_usdt_symbols(ex_name)
    except Exception as e:
        all_usdt = []
        st.warning(f"Could not load markets: {e}")

    # Initialize working list once
    if "working_symbols" not in st.session_state:
        st.session_state.working_symbols = SAFE_STABLE.copy()

    # ---- SAFE MULTISELECT (fixes the default-in-options error) ----
    raw_options = all_usdt if all_usdt else []
    options_union = list(dict.fromkeys(raw_options + st.session_state.working_symbols))
    if not options_union:
        options_union = SAFE_STABLE.copy()  # final fallback

    st.session_state.working_symbols = st.multiselect(
        "Edit list (type to search)",
        options=options_union,
        default=st.session_state.working_symbols,  # guaranteed to be in options
        placeholder="Start typing e.g. BTC/USDT…",
    )

    st.markdown("---")
    st.subheader("Scan Trending (Tech + Sentiment)")
    min_pct_4h = st.number_input("Min 4h % change", value=2.0, step=0.5, format="%.1f")
    min_vol_z = st.number_input("Min 24h volume z-score", value=1.0, step=0.1, format="%.1f")
    rsi_low, rsi_high = st.slider("RSI14 bounds", 0, 100, (40, 70))
    min_snt = st.slider("Min sentiment score", -1.0, 1.0, 0.2, 0.05)

    colX, colY = st.columns(2)
    do_scan = colX.button("🔎 Scan & Show")
    add_pass = colY.button("➕ Add Passing to List")

# Scan results
scan_df: pd.DataFrame | None = None
scan_pass: List[str] = []

if do_scan:
    ex = make_exchange(ex_name)
    # scan the larger universe if available; otherwise scan current list
    source = all_usdt if all_usdt else st.session_state.working_symbols
    with st.spinner("Scanning universe…"):
        scan_df, scan_pass = scan_trending(
            ex,
            universe=source,
            timeframe=(tf if tf in ("15m","1h","4h") else "15m"),
            min_pct_4h=float(min_pct_4h),
            min_vol_z=float(min_vol_z),
            rsi_bounds=(int(rsi_low), int(rsi_high)),
            min_sentiment=float(min_snt),
        )

    st.subheader("🔍 Verification — why these were picked")
    if scan_df is None or scan_df.empty:
        st.warning("No matches. Try relaxing thresholds.")
    else:
        st.dataframe(
            scan_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        st.write("**Pass (tech+sentiment):**", scan_pass if scan_pass else "None")

if add_pass and scan_pass:
    merged = st.session_state.working_symbols + [s for s in scan_pass if s not in st.session_state.working_symbols]
    # always include SAFE_STABLE
    merged = list(dict.fromkeys(SAFE_STABLE + merged))
    st.session_state.working_symbols = merged
    st.success(f"Added {len(scan_pass)} symbols. Total now: {len(st.session_state.working_symbols)}")

st.markdown("---")
st.subheader("📦 Build Snapshot")

split = st.toggle("Split into multiple files", value=False)
chunk_n = st.number_input("Max symbols per file", 5, 200, 40, 5, disabled=not split)

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _download_payload(payload: Dict[str, Any], name: str):
    as_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    st.download_button(name, data=as_bytes, file_name=name.replace("⬇️ ", ""), mime="application/json")

build = st.button("🚀 Build Snapshot", type="primary", use_container_width=True)
if build:
    syms = st.session_state.working_symbols
    if not syms:
        st.warning("No symbols selected.")
        st.stop()

    if split:
        groups = _chunk(syms, int(chunk_n))
        st.info(f"Creating {len(groups)} files…")
        first_payload = None
        for i, grp in enumerate(groups, start=1):
            payload = build_snapshot(ex_name, tf, grp, limit)
            if first_payload is None:
                first_payload = payload
            _download_payload(payload, f"⬇️ snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{i:02d}.json")
        if first_payload:
            with st.expander("🔍 Preview first JSON"):
                st.code(json.dumps(first_payload, indent=2, ensure_ascii=False), language="json")
    else:
        payload = build_snapshot(ex_name, tf, syms, limit)
        _download_payload(payload, f"⬇️ snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
        with st.expander("🔍 View JSON"):
            st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
