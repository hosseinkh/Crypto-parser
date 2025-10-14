from __future__ import annotations
import pandas as pd

def _get_last_close(tf_block: dict) -> float:
    return tf_block["last_candles"][-1]["c"]

def _get_vol_z(tf_block: dict):
    """
    Compatibility: some modules emit 'vol_z', others 'volume_z'.
    Returns a float or None.
    """
    vol = tf_block.get("volume", {}) or {}
    if "vol_z" in vol:
        return vol.get("vol_z")
    if "volume_z" in vol:
        return vol.get("volume_z")
    return None

# ---------------------------------------------------------------------------
# Mean-reversion style helpers (v4.1 aligned)
# ---------------------------------------------------------------------------

def pullback_in_uptrend(tf_block: dict) -> bool:
    """
    v4.1 alignment:
      - RSI window widened to 35..58 (was 35..50)
      - Keep 'price <= EMA20' as your original 'controlled pullback' proxy
    """
    ind = tf_block["indicators"]
    trend = tf_block["structure"]["trend"]
    price = _get_last_close(tf_block)
    ema20 = ind["ema20"]
    rsi = ind["rsi"]
    return (trend == "up") and (price <= ema20) and (35 <= rsi <= 58)

def oversold_near_support(tf_block: dict) -> bool:
    """
    v4.1 alignment:
      - Distance to nearest support widened to ≤ 3% (was 1%)
      - Kept RSI < 35 signal as your 'oversold' variant
    """
    ind = tf_block["indicators"]
    levels = tf_block["levels"]
    rsi = ind["rsi"]
    price = _get_last_close(tf_block)
    sup = levels["nearest_support"]
    dist = abs(price - sup) / max(price, 1e-9)
    return (rsi < 35) and (dist <= 0.03)

# ---------------------------------------------------------------------------
# Breakout with volume (v4.1 aligned)
# ---------------------------------------------------------------------------

def breakout_with_volume(tf_block: dict) -> bool:
    """
    v4.1 alignment:
      - Location filter: allow break readiness when price is within 2.0% of range high
      - Volume tiering: primary >= 0.8, fallback >= 0.4
      - Keep existing 'events.breakout' short-circuit if your upstream detector set it

    Logic:
      If an upstream detector already flags a breakout, return it.
      Else:
        near_high = (price >= range_high) OR ( (range_high - price)/price <= 0.02 )
        vol_ok = (vol_z >= 0.8) OR (vol_z >= 0.4 and near_high)
        return near_high and vol_ok
    """
    events = tf_block.get("events", {}) or {}
    if "breakout" in events:
        return bool(events["breakout"])

    struct = tf_block["structure"]
    price = _get_last_close(tf_block)
    rh = struct["range_high"]

    # distance to range high (≤ 2%)
    # consider both strict break and "pressing" the level
    within_2pct = abs(rh - price) / max(price, 1e-9) <= 0.02
    near_or_above_high = (price >= rh) or within_2pct

    volz = _get_vol_z(tf_block)

    # volume tiers (primary/fallback)
    vol_primary_ok = (volz is not None) and (volz >= 0.8)
    vol_fallback_ok = (volz is not None) and (volz >= 0.4) and near_or_above_high

    return bool(near_or_above_high and (vol_primary_ok or vol_fallback_ok))
