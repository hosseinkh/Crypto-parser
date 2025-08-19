from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import timezone
from .indicators import rsi, ema, atr, volume_zscore, ema_trend, recent_range, swing_pattern, nearest_sr_from_pivots

def df_to_last_candles(df: pd.DataFrame, max_rows: int) -> list[dict]:
    rows = df.tail(max_rows)
    out = []
    for _, r in rows.iterrows():
        out.append({
            "t": r["ts"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": float(r["open"]), "h": float(r["high"]),
            "l": float(r["low"]), "c": float(r["close"]),
            "v": float(r["volume"])
        })
    return out

def build_tf_block(df: pd.DataFrame, price: float, tf: str, lc_count: int) -> dict:
    df = df.copy()
    df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
    ind = {}
    ind["ema20"] = float(ema(df["close"], 20).iloc[-1])
    ind["ema50"] = float(ema(df["close"], 50).iloc[-1])
    ind["ema200"] = float(ema(df["close"], 200).iloc[-1]) if len(df) >= 200 else float(ema(df["close"], 200).iloc[-1])
    ind["rsi"] = float(rsi(df["close"], 14).iloc[-1])
    _atr = atr(df[["high","low","close"]], 14)
    ind["atr"] = float(_atr.iloc[-1])
    ind["atr_pct"] = float(ind["atr"] / max(price, 1e-9))

    trend = ema_trend(df["close"], span=50, eps=0.0)
    rng_low, rng_high = recent_range(df["close"], window=min(40, len(df)))
    sp = swing_pattern(df["close"], _atr, mult=1.5)

    # nearest S/R
    sup, res = nearest_sr_from_pivots(df["close"], window=min(40, len(df)))
    dist_sup = (price - sup) / max(price, 1e-9)
    dist_res = (res - price) / max(price, 1e-9)

    vol_z = volume_zscore(df["volume"], 20).iloc[-1]
    vol_block = {
        "current": float(df["volume"].iloc[-1]),
        "mean20": float(df["volume"].rolling(20).mean().iloc[-1]),
        "vol_z": None if np.isnan(vol_z) else float(vol_z)
    }

    # breakout event
    breakout = bool(price > rng_high) or bool(price < rng_low)
    breakout_dir = "up" if price > rng_high else ("down" if price < rng_low else None)

    tf_block = {
        "asof": df["ts"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_candles": df_to_last_candles(df, lc_count),
        "indicators": ind,
        "structure": {
            "trend": trend,
            "swing_pattern": sp,
            "in_range": bool(not breakout),
            "range_window": min(40, len(df)),
            "range_low": float(rng_low),
            "range_high": float(rng_high)
        },
        "levels": {
            "nearest_support": float(sup),
            "nearest_resistance": float(res),
            "dist_to_sup_pct": float(dist_sup),
            "dist_to_res_pct": float(dist_res)
        },
        "volume": vol_block,
        "events": {
            "breakout": bool(breakout),
            "breakout_dir": breakout_dir,
            "divergence": { "rsi_bull": False, "rsi_bear": False }  # placeholder
        },
        # risk_box/confidence filled later by screeners if desired
    }
    return tf_block

def attach_screen_flags(coin_obj: dict, filters_mod) -> dict:
    # Evaluate filters on fastest TF present (prefer 15m, then 1h, else 4h)
    tf_choice = None
    for tf in ("15m","1h","4h"):
        if tf in coin_obj["timeframes"]:
            tf_choice = tf; break
    flags = {
        "pullback_in_uptrend": False,
        "oversold_near_support": False,
        "breakout_with_volume": False
    }
    if tf_choice:
        block = coin_obj["timeframes"][tf_choice]
        try: flags["pullback_in_uptrend"] = filters_mod.pullback_in_uptrend(block)
        except: pass
        try: flags["oversold_near_support"] = filters_mod.oversold_near_support(block)
        except: pass
        try: flags["breakout_with_volume"] = filters_mod.breakout_with_volume(block)
        except: pass
    coin_obj["screen_flags"] = flags
    return coin_obj
