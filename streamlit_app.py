# streamlit_app.py
# ----------------
# Professional Snapshot Builder (JSON only)
#
# What it does
#  - lets you pick symbols/timeframe/exchange
#  - fetches recent OHLCV + a tick price captured at snapshot time
#  - computes a compact indicator set (RSI14, ATR14, SMA5/10, Volume Z)
#  - packages everything into a single JSON file with a UTC timestamp
#
# No scanning / framework logic is included by design.

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# ========= Helpers =========

def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x))
    except Exception:
        return float("nan")


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Classic Wilder RSI (vectorized)."""
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50.0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def volume_zscore(vol: pd.Series, window: int = 50) -> pd.Series:
    ma = vol.rolling(window).mean()
    sd = vol.rolling(window).std(ddof=0)
    z = (vol - ma) / (sd.replace(0, np.nan))
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def infer_trend(df: pd.DataFrame) -> str:
    """Very simple structure cue: SMA5 vs SMA10 and slope."""
    sma5 = df["close"].rolling(5).mean()
    sma10 = df["close"].rolling(10).mean()
    last = len(df) - 1
    if last < 10:
        return "?"
    if sma5.iloc[last] > sma10.iloc[last] and sma5.iloc[last] > sma5.iloc[last-1]:
        return "up"
    if sma5.iloc[last] < sma10.iloc[last] and sma5.iloc[last] < sma5.iloc[last-1]:
        return "down"
    return "side"


def ccxt_client(name: str):
    name = name.lower()
    if name == "binance":
        return ccxt.binance({"enableRateLimit": True})
    # Default to Bitget (USDT-margined spot)
    return ccxt.bitget({"enableRateLimit": True})


def fetch_ohlcv_and_tick(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int
) -> Tuple[pd.DataFrame, float]:
    """
    Returns:
      df: columns = [timestamp, open, high, low, close, volume] indexed by datetime (UTC)
      tick: last price *captured now* (independent of last closed candle)
    """
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        raise RuntimeError(f"No OHLCV for {symbol} {timeframe}")
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["ts"]).astype(float)

    # Tick-at-snapshot: try ticker first, fallback to last close
    tick = float("nan")
    try:
        tkr = ex.fetch_ticker(symbol)
        tick = _safe_float(tkr.get("last"))
        if math.isnan(tick):
            tick = _safe_float(tkr.get("close"))
    except Exception:
        pass
    if math.isnan(tick):
        tick = float(df["close"].iloc[-1])
    return df, tick


def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["sma5"] = float(df["close"].rolling(5).mean().iloc[-1])
    out["sma10"] = float(df["close"].rolling(10).mean().iloc[-1])
    out["rsi14"] = float(rsi(df["close"], 14).iloc[-1])
    out["atr14"] = float(atr(df, 14).iloc[-1])

    # distance to rolling 50-bar extremes (as %)
    roll_high = df["high"].rolling(50).max().iloc[-1]
    roll_low = df["low"].rolling(50).min().iloc[-1]
    last_close = float(df["close"].iloc[-1])
    if roll_high and not math.isnan(roll_high):
        out["dist_to_high"] = float((roll_high - last_close) / roll_high * 100.0)
    else:
        out["dist_to_high"] = float("nan")
    if roll_low and not math.isnan(roll_low):
        out["dist_to_low"] = float((last_close - roll_low) / last_close * 100.0)
    else:
        out["dist_to_low"] = float("nan")

    out["vol_z"] = float(volume_zscore(df["volume"], 50).iloc[-1])

    return out


def package_symbol(
    symbol: str,
    df: pd.DataFrame,
    tick_now: float,
    exchange: str,
    timeframe: str
) -> Dict:
    last_row = df.iloc[-1]
    trend = infer_trend(df)

    indicators = compute_indicators(df)

    packaged = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "snapshot_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prices": {
            "tick_at_snapshot": round(_safe_float(tick_now), 8),
            "last_candle": {
                "open": round(_safe_float(last_row["open"]), 8),
                "high": round(_safe_float(last_row["high"]), 8),
                "low": round(_safe_float(last_row["low"]), 8),
                "close": round(_safe_float(last_row["close"]), 8),
                "volume": round(_safe_float(last_row["volume"]), 8),
            },
        },
        "indicators": {
            "sma5": round(indicators.get("sma5", float("nan")), 8),
            "sma10": round(indicators.get("sma10", float("nan")), 8),
            "rsi14": round(indicators.get("rsi14", float("nan")), 4),
            "atr14": round(indicators.get("atr14", float("nan")), 8),
            "dist_to_high": round(indicators.get("dist_to_high", float("nan")), 4),
            "dist_to_low": round(indicators.get("dist_to_low", float("nan")), 4),
            "vol_z": round(indicators.get("vol_z", float("nan")), 2),
        },
        "structure": {
            "trend": trend,  # "up" | "down" | "side" | "?"
        },
    }
    return packaged


def build_snapshot(
    exchange_name: str,
    symbols: List[str],
    timeframe: str,
    limit: int
) -> Tuple[List[Dict], List[Dict]]:
    ex = ccxt_client(exchange_name)
    ex.load_markets()
    rows: List[Dict] = []
    issues: List[Dict] = []

    for sym in symbols:
        with st.status(f"Fetching {sym} …", expanded=False) as s:
            try:
                # Normalize symbol to the exchange’s style if possible
                sym_norm = sym
                # ccxt standard spot pairs usually "BTC/USDT"
                if "/" not in sym:
                    base = sym.replace("USDT", "").replace("USD", "")
                    if f"{base}/USDT" in ex.markets:
                        sym_norm = f"{base}/USDT"
                    elif f"{base}/USD" in ex.markets:
                        sym_norm = f"{base}/USD"

                df, tick = fetch_ohlcv_and_tick(ex, sym_norm, timeframe, limit)
                packaged = package_symbol(sym_norm, df, tick, exchange_name, timeframe)
                rows.append(packaged)
                s.update(label=f"Fetched {sym_norm}", state="complete")
            except Exception as e:
                issues.append({"symbol": sym, "error": str(e)})
                s.update(label=f"Failed {sym}: {e}", state="error")
    return rows, issues


def pack_snapshot_from_rows(
    rows: List[Dict],
    meta: Dict
) -> Dict:
    return {
        "app": "professional-snapshot-builder",
        "version": "1.0.0",
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exchange": meta.get("exchange"),
        "timeframe": meta.get("timeframe"),
        "limit": meta.get("limit"),
        "symbols_count": len(rows),
        "rows": rows,
    }


# ========= UI =========

st.set_page_config(
    page_title="Professional Snapshot Builder",
    page_icon="📉",
    layout="centered",
)

st.title("📈 Professional Snapshot Builder")
st.info("Configure symbols in the sidebar and click **Build snapshot**.")

with st.sidebar:
    st.header("Settings")

    exchange = st.selectbox(
        "Exchange",
        ["Bitget", "Binance"],
        index=0,
        help="Public REST via CCXT (no key needed)."
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=2
    )

    limit = st.slider(
        "Candles per symbol",
        min_value=100, max_value=1500, value=500, step=50,
        help="More candles = more stable indicators, slower fetch."
    )

    st.markdown("---")

    default_symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "TON/USDT",
        "TRX/USDT", "LINK/USDT", "MATIC/USDT", "DOT/USDT",
        "AVAX/USDT", "LTC/USDT",
    ]
    symbols_text = st.text_area(
        "Symbols (one per line)",
        value="\n".join(default_symbols),
        height=220,
        help="Use CCXT spot notation like BTC/USDT. You can paste your own list."
    )

    build_btn = st.button("🧱 Build snapshot", type="primary", use_container_width=True)

if build_btn:
    syms = [s.strip() for s in symbols_text.splitlines() if s.strip()]
    if not syms:
        st.error("Please provide at least one symbol.")
        st.stop()

    st.subheader("Progress")
    rows, issues = build_snapshot(exchange, syms, timeframe, limit)

    if issues:
        with st.expander("Warnings / Failures"):
            for it in issues:
                st.warning(f"{it['symbol']}: {it['error']}")

    if not rows:
        st.error("No data fetched.")
        st.stop()

    # ---- Table preview (summary) ----
    st.subheader("Snapshot preview")
    table = []
    for r in rows:
        ind = r["indicators"]
        last = r["prices"]["last_candle"]
        table.append({
            "symbol": r["symbol"],
            "tick": r["prices"]["tick_at_snapshot"],
            "close": last["close"],
            "rsi14": ind["rsi14"],
            "atr14": ind["atr14"],
            "sma5": ind["sma5"],
            "sma10": ind["sma10"],
            "dist_to_high_%": ind["dist_to_high"],
            "dist_to_low_%": ind["dist_to_low"],
            "vol_z": ind["vol_z"],
            "trend": r["structure"]["trend"],
        })
    df = pd.DataFrame(table)
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)

    # ---- Download JSON ----
    meta = {"exchange": exchange, "timeframe": timeframe, "limit": limit}
    packed = pack_snapshot_from_rows(rows, meta)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    st.download_button(
        "⬇️ Download snapshot JSON",
        data=as_bytes,
        file_name=f"snapshot_{ts}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("View JSON"):
        st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")

else:
    st.caption("← Configure settings in the sidebar and click **Build snapshot**.")
