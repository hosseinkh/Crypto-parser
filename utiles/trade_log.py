from __future__ import annotations
import io
import pandas as pd
from datetime import datetime, timezone

COLS = [
    "trade_id","snapshot_id","time_open","symbol","side",
    "entry_plan_entry","entry_plan_sl","entry_plan_tp","size",
    "time_fill","price_fill","time_exit","price_exit","status",
    "pnl_usd","hold_minutes","setup_tag","notes"
]

def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)

def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def new_trade(trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
    row = {c: kwargs.get(c, "") for c in COLS}
    return pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
