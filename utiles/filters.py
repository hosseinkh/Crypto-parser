from __future__ import annotations
import pandas as pd

def pullback_in_uptrend(tf_block: dict) -> bool:
    ind = tf_block["indicators"]
    trend = tf_block["structure"]["trend"]
    price = tf_block["last_candles"][-1]["c"]
    ema20 = ind["ema20"]; rsi = ind["rsi"]
    return (trend == "up") and (price <= ema20) and (35 <= rsi <= 50)

def oversold_near_support(tf_block: dict) -> bool:
    ind = tf_block["indicators"]
    levels = tf_block["levels"]
    rsi = ind["rsi"]; price = tf_block["last_candles"][-1]["c"]
    sup = levels["nearest_support"]
    dist = abs(price - sup) / max(price, 1e-9)
    return (rsi < 35) and (dist <= 0.01)

def breakout_with_volume(tf_block: dict) -> bool:
    events = tf_block.get("events", {})
    if "breakout" in events:
        return bool(events["breakout"])
    # fallback from range + vol_z
    struct = tf_block["structure"]
    volz = tf_block["volume"]["vol_z"]
    price = tf_block["last_candles"][-1]["c"]
    rh = struct["range_high"]
    return (price > rh) and (volz is not None and volz >= 1.0)
