# streamlit_app.py
# ------------------------------------------------------------
# Professional Snapshot Builder — with "Scan for trends"
# and snapshot-time last price capture.
# ------------------------------------------------------------
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

import pandas as pd
import streamlit as st

# ---------- Optional imports (defensive) ----------
# All of these are *optional*. If your repo provides them, we use them;
# otherwise we fall back to simple local implementations.
try:
    from utils.exchanges import get_client as _get_client  # preferred
except Exception:
    _get_client = None

try:
    from utils.discovery import scan_trending as _scan_trending  # preferred
except Exception:
    _scan_trending = None

try:
    from utils.sentiment import get_symbol_sentiment as _sentiment  # preferred
except Exception:
    _sentiment = None

try:
    from utils.technical import compute_indicators as _compute_indis  # preferred
    from utils.technical import classify_structure as _classify_struct  # preferred
except Exception:
    _compute_indis = None
    _classify_struct = None


# ---------- Fallback helpers (used only if modules missing) ----------
def _fallback_get_client(exchange: str):
    """Very small ccxt-like client using pandas-friendly outputs.
    Replace with your real utils.exchanges.get_client if available.
    """
    import ccxt  # Streamlit Cloud supports ccxt via requirements.txt

    ex_map = {
        "Bitget": "bitget",
        "Binance": "binance",
        "Bybit": "bybit",
        "OKX": "okx",
    }
    name = ex_map.get(exchange, "bitget")
    client = getattr(ccxt, name)()
    return client  # public endpoints: OHLCV/ticker don't need keys


def _fallback_fetch_ohlcv(client, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Return OHLCV dataframe with utc datetime index."""
    ohlcv = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def _fallback_last_price(client, symbol: str) -> float:
    """Try ticker last/close; if fails, use last candle close."""
    try:
        t = client.fetch_ticker(symbol)
        for k in ("last", "close", "info.price"):
            v = t
            for p in k.split("."):
                if isinstance(v, dict) and p in v:
                    v = v[p]
                else:
                    v = None
                    break
            if isinstance(v, (int, float)):
                return float(v)
    except Exception:
        pass
    # final fallback
    df = _fallback_fetch_ohlcv(client, symbol, "1m", 2)
    return float(df["close"].iloc[-1])


def _fallback_compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Very small technical pack (RSI14, ATR14, distance to 20-period high/low)."""
    import numpy as np

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # RSI14 (Wilder-style with EMA)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, 1e-9))
    rsi14 = 100 - (100 / (1 + rs))
    rsi_val = float(rsi14.iloc[-1])

    # ATR14
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    # Distance to 20-period high/low (in %)
    win = 20
    max20 = high.rolling(win).max().iloc[-1]
    min20 = low.rolling(win).min().iloc[-1]
    last = float(close.iloc[-1])
    dist_to_high = float((max20 - last) / last * 100) if last else float("nan")
    dist_to_low = float((last - min20) / last * 100) if last else float("nan")

    return {
        "rsi14": round(rsi_val, 2),
        "atr14": round(atr14, 8),
        "dist_to_high": round(dist_to_high, 3),
        "dist_to_low": round(dist_to_low, 3),
    }


def _fallback_classify_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Simple trend read: up if last close > 50-EMA, down if < 50-EMA."""
    ema = df["close"].ewm(span=50, adjust=False).mean()
    last = float(df["close"].iloc[-1])
    last_ema = float(ema.iloc[-1])
    trend = "UP" if last > last_ema else "DOWN" if last < last_ema else "FLAT"
    return {"trend": trend}


def _fallback_sentiment(symbol: str) -> Dict[str, Any]:
    """Dummy neutral sentiment when utils.sentiment not available."""
    return {"score": 0.0, "label": "neutral", "source": "fallback"}


def _fallback_scan_trending(exchange: str, lookback_h: int = 24, top_n: int = 12) -> List[str]:
    """Quick momentum+volume proxy on USDT pairs."""
    client = _fallback_get_client(exchange)
    markets = client.load_markets()
    syms = [s for s, m in markets.items() if s.endswith("/USDT") and not m.get("future") and not m.get("swap")]
    rows = []
    for s in syms:
        try:
            t = client.fetch_ticker(s)
            chg = t.get("percentage")
            qv = t.get("quoteVolume") or t.get("baseVolume")
            if chg is None:
                last = t.get("last") or t.get("close") or 0
                open_ = t.get("open") or last
                chg = ((last - open_) / open_ * 100) if open_ else 0
            rows.append((s, float(chg), float(qv or 0)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["symbol", "pct", "qvol"])
    if df.empty:
        return []
    df["rank"] = df["pct"].rank(pct=True) + df["qvol"].rank(pct=True)
    out = list(df.sort_values("rank", ascending=False)["symbol"].head(top_n))
    return out


# ---------- Small adapters picking real utils.* when present ----------
def get_client(exchange: str):
    if _get_client is not None:
        return _get_client(exchange)
    return _fallback_get_client(exchange)


def fetch_ohlcv(client, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    try:
        return client.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=limit)  # type: ignore[attr-defined]
    except Exception:
        return _fallback_fetch_ohlcv(client, symbol, timeframe, limit)


def fetch_last_price(client, symbol: str) -> float:
    try:
        return float(client.fetch_last_price(symbol))  # type: ignore[attr-defined]
    except Exception:
        return _fallback_last_price(client, symbol)


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    if _compute_indis is not None:
        return _compute_indis(df)
    return _fallback_compute_indicators(df)


def classify_structure(df: pd.DataFrame) -> Dict[str, Any]:
    if _classify_struct is not None:
        return _classify_struct(df)
    return _fallback_classify_structure(df)


def get_symbol_sentiment(symbol: str) -> Dict[str, Any]:
    if _sentiment is not None:
        return _sentiment(symbol)
    return _fallback_sentiment(symbol)


def scan_trending(exchange: str, lookback_h: int, top_n: int) -> List[str]:
    if _scan_trending is not None:
        try:
            return list(_scan_trending(exchange, lookback_h=lookback_h, top_n=top_n))
        except TypeError:
            return list(_scan_trending(exchange))
    return _fallback_scan_trending(exchange, lookback_h, top_n)


# ---------- Packing helpers ----------
def pack_snapshot(symbols: List[str], meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "meta": meta,
        "symbols": {
            r["symbol"]: {
                "tf": r["timeframe"],
                "last": r["last_price"],           # <-- snapshot-time price
                "indicators": r["indicators"],
                "structure": r["structure"],
                "sentiment": r["sentiment"],
                "asof": r["asof_iso"],
            }
            for r in rows
        },
        "universe": symbols,
    }


# =====================  UI  =====================
st.set_page_config(page_title="Professional Snapshot Builder", page_icon="📈", layout="wide")

st.title("📈 Professional Snapshot Builder")

with st.sidebar:
    st.header("Settings")

    exchange = st.selectbox("Exchange", ["Bitget", "Binance", "Bybit", "OKX"], index=0)
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
    limit = st.slider("Candles per symbol", min_value=100, max_value=2000, value=800, step=50)

    st.markdown("---")
    st.subheader("Symbols (one per line)")
    default_syms = "BTC/USDT\nETH/USDT\nSOL/USDT\nDOT/USDT\nXRP/USDT"
    if "symbols_text" not in st.session_state:
        st.session_state.symbols_text = default_syms
    symbols_text = st.text_area(" ", value=st.session_state.symbols_text, height=170)

    st.markdown("**Scan for trends**")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        lookback_h = st.number_input("Lookback (h)", min_value=6, max_value=72, value=24, step=6)
    with col_b:
        top_n = st.number_input("Top N", min_value=3, max_value=30, value=12, step=1)
    with col_c:
        only_usdt = st.checkbox("Only */USDT", value=True)

    do_scan = st.button("🔎 Scan & Append Trending", use_container_width=True)
    st.markdown("---")
    build_now = st.button("🧰 Analyze & Build Snapshot", type="primary", use_container_width=True)

# clean symbols
def parse_symbols(text: str) -> List[str]:
    syms = [s.strip().upper() for s in text.splitlines() if s.strip()]
    # normalize common variants like BTCUSDT -> BTC/USDT
    fixed = []
    for s in syms:
        if "/" not in s and s.endswith("USDT"):
            fixed.append(s[:-4] + "/USDT")
        else:
            fixed.append(s)
    # keep order while dedup
    seen = set()
    out = []
    for s in fixed:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


symbols = parse_symbols(symbols_text)

# ---------- Scan button ----------
if do_scan:
    with st.spinner("Scanning for trending markets…"):
        try:
            found = scan_trending(exchange, lookback_h=int(lookback_h), top_n=int(top_n))
        except Exception as e:
            st.error(f"Scan failed: {e}")
            found = []

    if only_usdt:
        found = [s for s in found if s.endswith("/USDT")]

    if not found:
        st.warning("No trending symbols found.")
    else:
        # Merge with current list
        merged = symbols + [s for s in found if s not in symbols]
        st.session_state.symbols_text = "\n".join(merged)
        st.success(f"Appended {len(found)} symbols.")
        st.dataframe(pd.DataFrame({"Trending": found}), use_container_width=True)

# Show the live symbols box (always synced)
st.session_state.symbols_text = st.session_state.get("symbols_text", default_syms)
st.text_area("Active symbol list", value=st.session_state.symbols_text, height=170, key="symbols_text_display", disabled=True)

# ---------- Analyze & Build ----------
if build_now:
    if not symbols:
        st.error("Please enter at least one symbol.")
        st.stop()

    client = get_client(exchange)

    results: List[Dict[str, Any]] = []
    problems: List[Tuple[str, str]] = []
    with st.spinner("Fetching data & computing indicators…"):
        for sym in symbols:
            try:
                df = fetch_ohlcv(client, sym, timeframe, limit)
                if df.empty or len(df) < 50:
                    raise ValueError("Not enough candles")
                indis = compute_indicators(df)
                struct = classify_structure(df)
                senti = get_symbol_sentiment(sym)
                snap_price = fetch_last_price(client, sym)  # <-- snapshot-time price

                results.append(
                    {
                        "symbol": sym,
                        "timeframe": timeframe,
                        "last_price": float(snap_price),
                        "indicators": indis,
                        "structure": struct,
                        "sentiment": senti,
                        "asof_iso": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                problems.append((sym, str(e)))

    if problems:
        st.warning("Some symbols failed:")
        st.table(pd.DataFrame(problems, columns=["symbol", "error"]))

    if results:
        # Table preview
        table = []
        for r in results:
            ind = r["indicators"]
            struct = r["structure"]
            senti = r["sentiment"]
            table.append(
                {
                    "symbol": r["symbol"],
                    "tf": r["timeframe"],
                    "last": round(r["last_price"], 8),
                    "trend": struct.get("trend", "?"),
                    "rsi14": ind.get("rsi14"),
                    "atr14": ind.get("atr14"),
                    "dist_to_high%": ind.get("dist_to_high"),
                    "dist_to_low%": ind.get("dist_to_low"),
                    "sentiment": f'{senti.get("label","?")} ({senti.get("score",0):+.2f})',
                }
            )
        df_show = pd.DataFrame(table)
        st.subheader("Analysis preview")
        st.dataframe(df_show, use_container_width=True)

        # Build JSON
        meta = {
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exchange": exchange,
            "timeframe": timeframe,
            "candles_per_symbol": int(limit),
            "note": "last = snapshot-time price",
        }
        packed = pack_snapshot(symbols, meta, results)

        # Download buttons
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

# ---------- Footer ----------
st.caption("Tip: Edit the *Symbols* box, hit **Scan & Append Trending**, then **Analyze & Build Snapshot**.")
