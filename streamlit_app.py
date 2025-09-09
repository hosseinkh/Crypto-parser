# streamlit_app.py
# --- Professional Snapshot Builder with "Discover Trendy" & Autocomplete ---
from __future__ import annotations

import os
import math
import json
import time
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import requests

# ==== Your local project imports (unchanged names assumed) ====
# Each exchange module must expose:
#   - name: str
#   - list_spot_symbols() -> list[str]
#   - fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame with columns:
#         ["timestamp","open","high","low","close","volume"]
#   - fetch_last_price(symbol: str) -> float
#
# You said Bitget is already updated; keep the others similar.
from utils.exchanges import bitget, binance, bybit  # adapt to your repo layout

# ----------------------------- Config -----------------------------
EXCHANGES = {
    "Bitget": bitget,
    "Binance": binance,
    "Bybit": bybit,
}
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
DEFAULT_TF = "15m"

NEWS_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "").strip()  # optional
NEWS_WEIGHT = 0.15  # small boost if there is fresh news

# ----------------------------- Helpers -----------------------------
@dataclass
class SymbolSnapshot:
    exchange: str
    symbol: str
    timeframe: str
    candles: int
    last_price: float
    indicators: dict
    structure: dict
    meta: dict


def _ta(df: pd.DataFrame) -> dict:
    """Compute light indicators used by the framework."""
    out: dict[str, t.Any] = {}
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["volume"].astype(float).values

    def sma(x, n):
        if len(x) < n: 
            return np.full_like(x, np.nan, dtype=float)
        s = pd.Series(x, dtype="float64").rolling(n).mean().to_numpy()
        return s

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    out["ma5"] = float(ma5[-1]) if not math.isnan(ma5[-1]) else np.nan
    out["ma10"] = float(ma10[-1]) if not math.isnan(ma10[-1]) else np.nan

    # RSI(14)
    def rsi(x, n=14):
        if len(x) < n + 1:
            return np.full_like(x, np.nan, dtype=float)
        delta = np.diff(x)
        up = np.clip(delta, 0, None)
        down = -np.clip(delta, None, 0)
        roll_up = pd.Series(up).rolling(n).mean()
        roll_down = pd.Series(down).rolling(n).mean()
        rs = roll_up / (roll_down + 1e-12)
        r = 100 - (100 / (1 + rs))
        r = np.concatenate([np.full(1, np.nan), r.to_numpy()])
        return r

    rsi14 = rsi(close, 14)
    out["rsi14"] = float(rsi14[-1]) if not math.isnan(rsi14[-1]) else np.nan

    # 20-period high/low and distances (breakout context)
    look = 20 if len(high) >= 20 else len(high)
    hh = np.max(high[-look:]) if look else np.nan
    ll = np.min(low[-look:]) if look else np.nan
    last = close[-1]
    out["hh20"] = float(hh)
    out["ll20"] = float(ll)
    out["dist_to_high"] = float((hh - last) / hh * 100) if hh and hh > 0 else np.nan
    out["dist_to_low"] = float((last - ll) / ll * 100) if ll and ll > 0 else np.nan

    # Volume z-score (last vs 20)
    v_look = 20 if len(vol) >= 20 else len(vol)
    if v_look >= 3:
        v_mean = float(np.mean(vol[-v_look:]))
        v_std = float(np.std(vol[-v_look:], ddof=1))
        out["vol_z"] = float((vol[-1] - v_mean) / (v_std + 1e-12))
    else:
        out["vol_z"] = np.nan

    # 24h momentum proxy (close vs close 96 bars back on 15m ⇒ 24h)
    back = 96 if len(close) >= 97 and out.get("hh20") else min(len(close) - 1, 96)
    if back > 0:
        out["mom_24h"] = float((last / close[-1 - back] - 1) * 100)
    else:
        out["mom_24h"] = np.nan

    return out


def _structure_from_ta(ind: dict) -> dict:
    """Light structure signals used both for discovery & snapshot table."""
    ma_ok = ind.get("ma5", np.nan) > ind.get("ma10", np.nan)
    rsi_ok = 30 < ind.get("rsi14", 50) < 75
    near_break = ind.get("dist_to_high", np.inf) <= 0.5  # within 0.5% of 20-bar high
    trend = "UP" if ma_ok else "DOWN"
    bias = (
        "breakout"
        if near_break and ma_ok
        else "pullback-long" if ma_ok and not near_break and rsi_ok
        else "avoid"
    )
    return {"trend": trend, "bias": bias}


def _news_boost(symbol: str) -> float:
    """Optional: tiny score boost if CryptoPanic has fresh headlines for this ticker."""
    if not NEWS_API_KEY:
        return 0.0
    try:
        # naive: search by symbol (many exchanges share tickers; this is intentionally light)
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": NEWS_API_KEY, "currencies": symbol.split("/")[0], "public": "true"}
        r = requests.get(url, params=params, timeout=6)
        if r.ok and r.json().get("results"):
            return NEWS_WEIGHT
    except Exception:
        pass
    return 0.0


def score_trendy(df: pd.DataFrame, ind: dict, symbol: str) -> float:
    """Composite score for 'Discover Trendy'."""
    score = 0.0
    # MA alignment
    if ind.get("ma5", np.nan) > ind.get("ma10", np.nan):
        score += 0.35
    # Near breakout
    dth = ind.get("dist_to_high", np.inf)
    if not math.isnan(dth):
        score += max(0.0, 0.35 * max(0.0, (0.5 - dth) / 0.5))  # linear if within 0.5%
    # Volume surge
    vz = ind.get("vol_z", 0.0)
    if not math.isnan(vz):
        score += 0.2 * max(0.0, min(vz / 2.5, 1.0))  # cap at z=2.5
    # RSI sweet spot
    rsi = ind.get("rsi14", 50)
    if 40 <= rsi <= 65:
        score += 0.1
    # Optional tiny news boost
    score += _news_boost(symbol)
    return float(score)


def build_snapshot_for_symbols(ex_mod, symbols: list[str], timeframe: str, limit: int) -> list[SymbolSnapshot]:
    rows: list[SymbolSnapshot] = []
    for sym in symbols:
        try:
            df = ex_mod.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if df is None or len(df) == 0:
                continue
            # ensure correct dtypes
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            ind = _ta(df)
            struct = _structure_from_ta(ind)
            last_price = float(ex_mod.fetch_last_price(sym))  # capture at snapshot time
            snap = SymbolSnapshot(
                exchange=ex_mod.name,
                symbol=sym,
                timeframe=timeframe,
                candles=limit,
                last_price=last_price,
                indicators=ind,
                structure=struct,
                meta={"rows": int(len(df)), "generated_at": datetime.now(timezone.utc).isoformat()},
            )
            rows.append(snap)
        except Exception as e:
            st.warning(f"⚠️ {sym}: {e}")
    return rows


def pack_snapshot_from_rows(rows: list[SymbolSnapshot]) -> dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "items": [],
    }
    for r in rows:
        payload["items"].append(
            {
                "exchange": r.exchange,
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "candles": r.candles,
                "last_price": r.last_price,
                "ind": r.indicators,
                "struct": r.structure,
                "meta": r.meta,
            }
        )
    return payload


def pack_snapshot_chunked(rows: list[SymbolSnapshot], max_per_file: int) -> list[dict]:
    out = []
    for i in range(0, len(rows), max_per_file):
        out.append(pack_snapshot_from_rows(rows[i : i + max_per_file]))
    return out


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Professional Snapshot Builder", layout="wide")
st.title("📈 Professional Snapshot Builder")

with st.sidebar:
    st.subheader("Settings")

    ex_name = st.selectbox("Exchange", list(EXCHANGES.keys()), index=0)
    ex_mod = EXCHANGES[ex_name]

    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(DEFAULT_TF))
    limit = st.slider("Candles per symbol", min_value=200, max_value=2000, value=1300, step=50)

    # --- Symbol picker with auto-suggest ---
    with st.expander("Symbols (picker)", expanded=True):
        try:
            all_syms = ex_mod.list_spot_symbols()
        except Exception:
            all_syms = []
            st.warning("Could not fetch symbols from the exchange module.")

        default_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        universe = st.multiselect(
            "Choose symbols (search or paste):",
            options=all_syms,
            default=[s for s in default_list if s in all_syms] or default_list,
            help="Type to search. You can paste more symbols and press Enter.",
        )

        # Free-text additions (one per line)
        extra = st.text_area("Or add more (one per line, e.g., DOT/USDT)", height=90, placeholder="DOT/USDT\nLINK/USDT")
        if extra.strip():
            universe += [s.strip().upper() for s in extra.splitlines() if s.strip()]

    st.markdown("---")
    st.caption("🔎 Discover coins that are currently trending (framework + scalp filter).")
    colA, colB = st.columns([1, 1])
    with colA:
        top_n = st.number_input("How many to add?", min_value=5, max_value=40, value=12, step=1)
    with colB:
        do_discover = st.button("✨ Discover Trendy", use_container_width=True)

# --- Discover Trendy ---
if do_discover:
    with st.spinner("Scanning universe for trending candidates..."):
        try:
            scan_list = all_syms if all_syms else universe
            scored: list[tuple[str, float]] = []
            # light/thrifty pass: fewer candles just for scoring speed
            scan_limit = max(120, min(400, limit // 6))
            for sym in scan_list:
                try:
                    ddf = ex_mod.fetch_ohlcv(sym, timeframe=timeframe, limit=scan_limit)
                    if ddf is None or len(ddf) < 60:
                        continue
                    for c in ("open", "high", "low", "close", "volume"):
                        ddf[c] = pd.to_numeric(ddf[c], errors="coerce")
                    ind = _ta(ddf)
                    # scalp filtering: MA alignment, RSI in tradable band, vol z positive
                    scalp_ok = (
                        ind.get("ma5", np.nan) > ind.get("ma10", np.nan)
                        and 35 < ind.get("rsi14", 50) < 72
                        and ind.get("vol_z", 0) > -0.3
                    )
                    if not scalp_ok:
                        continue
                    s = score_trendy(ddf, ind, sym)
                    scored.append((sym, s))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[1], reverse=True)
            picked = [s for s, _ in scored[: int(top_n)]]
            st.success(f"Found {len(picked)} trendy candidates.")
            if picked:
                st.write(", ".join(picked))
                # merge into universe (unique, keep order)
                seen = set(universe)
                universe.extend([p for p in picked if p not in seen])
        except Exception as e:
            st.error(f"Discovery error: {e}")

# --- Build snapshot ---
st.markdown("### Build")
build = st.button("🧱 Build Snapshot", type="primary")

if build:
    with st.spinner("Building snapshot..."):
        t0 = time.time()
        rows = build_snapshot_for_symbols(ex_mod, universe, timeframe, limit)
        took = time.time() - t0
        st.success(f"Built {len(rows)} items in {took:.1f}s.")

        # Table preview
        table = []
        for r in rows:
            ind, struct = r.indicators, r.structure
            table.append(
                {
                    "symbol": r.symbol,
                    "last": r.last_price,
                    "trend": struct.get("trend", "?"),
                    "bias": struct.get("bias", "?"),
                    "ma5": round(ind.get("ma5", float("nan")), 6),
                    "ma10": round(ind.get("ma10", float("nan")), 6),
                    "rsi14": round(ind.get("rsi14", float("nan")), 2),
                    "dist_to_high%": round(ind.get("dist_to_high", float("nan")), 3),
                    "vol_z": round(ind.get("vol_z", float("nan")), 2),
                    "mom_24h%": round(ind.get("mom_24h", float("nan")), 2),
                }
            )
        df = pd.DataFrame(table)

        try:
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.table(df)

        # === Downloads ===
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        split_output = st.toggle("Split into multiple files", value=False)
        max_per_file = st.number_input("Max symbols per file", min_value=10, max_value=100, value=40, step=5)

        if split_output:
            payloads = pack_snapshot_chunked(rows, int(max_per_file))
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
            as_bytes = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            st.download_button(
                "⬇️ Download single JSON snapshot",
                data=as_bytes,
                file_name=f"snapshot_{ts}.json",
                mime="application/json",
            )
            with st.expander("🔍 View JSON"):
                st.code(json.dumps(packed, indent=2, ensure_ascii=False), language="json")
