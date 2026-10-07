from __future__ import annotations

import pandas as pd

from . import data, panels
from .momentum import rolling_compound


def pead() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = data.load_base("pead_events")
    events["available_date"] = pd.to_datetime(events["available_date"])
    months = panels.crsp_monthly()[["date", "asset_id"]]
    out = pd.merge_asof(
        months.sort_values("date"),
        events.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        by="asset_id",
        direction="backward",
    )
    fresh = out["date"].between(out["available_date"], out["available_date"] + pd.DateOffset(months=6))
    out = out[fresh & out["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)].copy()
    return out[["date", "asset_id", "pead_car"]].rename(columns={"pead_car": "exposure"}), out


def long_term_reversal() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = -rolling_compound(m["ret"], m["asset_id"], skip=13, window=48, min_periods=36)
    return m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date", "asset_id", "exposure"]]


def intermediate_momentum() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = rolling_compound(m["ret"], m["asset_id"], skip=7, window=6, min_periods=5)
    return m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date", "asset_id", "exposure"]]


def daily_characteristic(column: str, direction: float = 1.0) -> pd.DataFrame:
    d = data.load_base("extended_daily_characteristics", columns=["date", "asset_id", column])
    d["date"] = pd.to_datetime(d["date"])
    d["exposure"] = direction * pd.to_numeric(d[column], errors="coerce")
    return d[d["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date", "asset_id", "exposure"]]
