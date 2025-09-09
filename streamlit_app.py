# streamlit_app.py
# -----------------
# Self-contained Streamlit app for building “professional” crypto snapshots.
# - No imports from local utils/ to avoid ModuleNotFoundError on Streamlit Cloud
# - Uses CCXT for market data (spot USDT pairs by default)
# - Captures the true last price at snapshot time
# - Adds a "Scan Trending" helper to auto-pick symbols by volume + momentum

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# CCXT is our data layer
try:
    import ccxt
except Exception as e:
    st.error(
        "CCXT is required. Add `ccxt` to requirements.txt and reboot the app.\n\n"
        f"Import error: {e}"
    )
    st.stop()


# -----------------------------
# Helpers: exchange + data I/O
# -----------------------------
EXCHANGE_NAMES = ["Bitget", "Binance", "Bybit", "OKX"]
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def make_exchange(name: str):
    name = (name or "").lower()
    if name == "bitget":
        return ccxt.bitget({"enableRateLimit": True})
    if name == "binance":
        return ccxt.binance({"enableRateLimit": True})
    if name == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    if name == "okx":
        return ccxt.okx({"enableRateLimit": True})
    raise ValueError(f"Unsupported exchange: {name}")


@st.cache_data(show_spinner=False, ttl=300)
def load_usdt_markets(exchange_name: str) -> List[str]:
    """Return all tradable *spot* USDT pairs supported by the exchange."""
    ex = make_exchange(exchange_name)
    ex.load_markets()
    symbols = []
    for m in ex.markets.values():
        # CCXT fields are not perfectly uniform across exchanges; be defensive
        quote = m.get("quote") or ""
        market_type = m.get("type") or m.get("spot") and "spot" or None
        active = m.get("active", True)
        symbol = m.get("symbol")
        if not symbol or not active:
            continue
        if quote.upper() == "USDT" and (market_type == "spot" or m.get("spot") is True):
            symbols.append(symbol)
    # De-dup and sort for nice UX
    return sorted(list(dict.fromkeys(symbols)))


def _safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
    """Fetch OHLCV with basic retries/backoff (kept small to stay under Streamlit timeouts)."""
    tries = 2
    last_err = None
    for _ in range(tries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            last_err = e
    raise last_err


def _safe_fetch_ticker(ex, symbol: str) -> Dict:
    tries = 2
    last_err = None
    for _ in range(tries):
        try:
            return ex.fetch_ticker(symbol)
        except Exception as e:
            last_err = e
    raise last_err


# -----------------------------
# Indicators (simple + robust)
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # True range
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / (std.replace(0, np.nan))


# -----------------------------
# Snapshot packing
# -----------------------------
def pack_row(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    last_price_snapshot: float,
) -> Dict:
    # basic structure with last candle info
    last = ohlcv_df.iloc[-1]
    ind = {
        "ma5": float(ohlcv_df["close"].rolling(5).mean().iloc[-1]),
        "ma10": float(ohlcv_df["close"].rolling(10).mean().iloc[-1]),
        "rsi14": float(rsi(ohlcv_df["close"], 14).iloc[-1]),
        "atr14": float(atr(ohlcv_df, 14).iloc[-1]),
        "vol_z": float(zscore(ohlcv_df["volume"], 50).iloc[-1]),
    }

    # simple structure label (up / down / range)
    up = ind["ma5"] > ind["ma10"]
    volatility_ok = not math.isnan(ind["atr14"]) and ind["atr14"] > 0
    trend = "up" if up and volatility_ok else ("down" if not up and volatility_ok else "range")

    # distances to intra-window high/low (last N=100 by default if available)
    window = min(100, len(ohlcv_df))
    wdf = ohlcv_df.iloc[-window:]
    w_high, w_low = float(wdf["high"].max()), float(wdf["low"].min())
    dist_high = (w_high - last_price_snapshot) / last_price_snapshot if last_price_snapshot else np.nan
    dist_low = (last_price_snapshot - w_low) / last_price_snapshot if last_price_snapshot else np.nan

    row = {
        "symbol": symbol,
        "exchange": exchange_name,
        "timeframe": timeframe,
        "snapshot_ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # exact last price captured at snapshot time
        "last_price_snapshot": float(last_price_snapshot) if last_price_snapshot is not None else None,
        # last candle (closing candle on chosen timeframe)
        "last_candle": {
            "ts": int(last["timestamp"]),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "volume": float(last["volume"]),
        },
        "indicators": ind,
        "structure": {
            "trend": trend,
            "dist_to_high": round(dist_high, 6) if dist_high == dist_high else None,  # NaN safe
            "dist_to_low": round(dist_low, 6) if dist_low == dist_low else None,
        },
    }
    return row


# -----------------------------
# Trending scan (volume + momo)
# -----------------------------
def scan_trending(exchange_name: str, limit: int = 15) -> List[str]:
    """
    Heuristic "trending" scan:
      - Spot USDT pairs only
      - Rank by: 24h quote volume (descending) + positive 24h % change
    Returns top `limit` symbols.
    """
    ex = make_exchange(exchange_name)
    ex.load_markets()
    usdt_symbols = [s for s in load_usdt_markets(exchange_name)]
    try:
        tickers = ex.fetch_tickers(usdt_symbols)
    except Exception:
        # Some exchanges throttle .fetch_tickers; fall back to per-symbol last
        tickers = {}
        for s in usdt_symbols[:200]:  # keep it bounded
            try:
                tickers[s] = ex.fetch_ticker(s)
            except Exception:
                continue

    ranked = []
    for sym, t in tickers.items():
        # CCXT fields differ; be defensive
        pct = t.get("percentage")
        vol_quote = t.get("quoteVolume") or t.get("baseVolume")
        last = t.get("last")
        if last is None or vol_quote is None or pct is None:
            continue
        try:
            score = (float(vol_quote)) * max(0.0, float(pct))
        except Exception:
            continue
        ranked.append((score, sym))

    ranked.sort(reverse=True)
    return [sym for _, sym in ranked[:limit]]


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Professional Snapshot Builder", page_icon="📈", layout="wide")

st.title("📈 Professional Snapshot Builder")
st.caption("Configure symbols in the sidebar and click **Build Snapshot**.")

with st.sidebar:
    st.header("Settings")

    ex_name = st.selectbox("Exchange", EXCHANGE_NAMES, index=0)
    tf = st.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=2)  # default 15m
    candles = st.slider("Candles per symbol", 100, 2000, 1300, step=50, help="Number of OHLCV rows to fetch.")

    st.markdown("---")
    st.subheader("Symbols")
    st.caption("Type to search USDT spot pairs. You can also add custom symbols below.")

    with st.spinner("Loading exchange markets..."):
        all_usdt = load_usdt_markets(ex_name)

    # Multiselect gives type-to-filter behavior (acts like suggestions)
    selected_symbols = st.multiselect(
        "Pick symbols",
        options=all_usdt,
        default=[s for s in all_usdt if s.split("/")[0] in {"BTC", "ETH", "SOL", "XRP"}][:6],
        help="Start typing to search (e.g., 'BTC/USDT').",
    )

    extra_text = st.text_area(
        "Extra symbols (optional, one per line)",
        value="",
        height=90,
        help="Paste any additional symbols supported by the exchange.",
    )
    extras = [s.strip() for s in extra_text.splitlines() if s.strip()]
    symbols_input = sorted(list(dict.fromkeys(selected_symbols + extras)))

    st.markdown("---")
    st.subheader("🔎 Scan Trending")
    st.caption("Auto-pick symbols by **volume + momentum** (USDT spot).")
    trending_n = st.slider("How many to add", 5, 30, 12, step=1)
    do_scan = st.button("Scan Trending", use_container_width=True)

    if do_scan:
        with st.spinner("Scanning trending symbols..."):
            trendy = scan_trending(ex_name, limit=trending_n)
        # Merge into current list
        merged = list(dict.fromkeys(symbols_input + trendy))
        st.success(f"Added {len(merged) - len(symbols_input)} trending symbols.")
        symbols_input = merged

    st.markdown("---")
    build = st.button("🚀 Build Snapshot", type="primary", use_container_width=True)

# -----------------------------
# Build snapshot on click
# -----------------------------
if build:
    if not symbols_input:
        st.warning("Please select at least one symbol.")
        st.stop()

    ex = make_exchange(ex_name)

    rows: List[Dict] = []
    errors: List[Tuple[str, str]] = []

    progress = st.progress(0.0, text="Fetching data…")
    for i, sym in enumerate(symbols_input, start=1):
        try:
            ohlcv = _safe_fetch_ohlcv(ex, sym, tf, limit=candles)
            if not ohlcv or len(ohlcv) < 50:
                raise RuntimeError("Not enough OHLCV data returned.")

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # IMPORTANT: capture last price *now*, at snapshot time
            ticker = _safe_fetch_ticker(ex, sym)
            last_price_snapshot = float(ticker.get("last")) if ticker and ticker.get("last") is not None else float(df["close"].iloc[-1])

            row = pack_row(ex_name, sym, tf, df, last_price_snapshot)
            rows.append(row)

        except Exception as e:
            errors.append((sym, str(e)))

        progress.progress(i / max(1, len(symbols_input)), text=f"Fetching {i}/{len(symbols_input)}…")

    progress.empty()

    if errors:
        with st.expander("⚠️ Errors (click to expand)"):
            for sym, msg in errors:
                st.write(f"- **{sym}**: {msg}")

    if not rows:
        st.error("No rows produced. Check symbols/timeframe/connection and try again.")
        st.stop()

    # ---------- Table preview ----------
    st.subheader("Snapshot Preview")
    table = []
    for r in rows:
        ind = r["indicators"]
        struct = r["structure"]
        lc = r["last_candle"]
        table.append(
            {
                "symbol": r["symbol"],
                "last_price_snapshot": r["last_price_snapshot"],
                "last_close": lc["close"],
                "trend": struct.get("trend"),
                "dist_to_high": struct.get("dist_to_high"),
                "dist_to_low": struct.get("dist_to_low"),
                "rsi14": round(ind.get("rsi14", float("nan")), 2),
                "atr14": round(ind.get("atr14", float("nan")), 6),
                "ma5": round(ind.get("ma5", float("nan")), 6),
                "ma10": round(ind.get("ma10", float("nan")), 6),
                "vol_z": round(ind.get("vol_z", float("nan")), 2),
            }
        )
    df = pd.DataFrame(table)
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.table(df)

    # ---------- Download JSON ----------
    st.subheader("Download")
    packed = {
        "meta": {
            "exchange": ex_name,
            "timeframe": tf,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "notes": "Each row contains last_candle (close of timeframe) *and* last_price_snapshot (true snapshot-time last).",
        },
        "rows": rows,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    st.download_button(
        "⬇️ Download snapshot JSON",
        data=as_bytes,
        file_name=f"snapshot_{ts}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("🔍 View JSON"):
        st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
else:
    st.info("Configure settings in the sidebar and press **Build Snapshot**.")
