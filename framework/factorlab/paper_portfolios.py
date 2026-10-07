from __future__ import annotations

import numpy as np
import pandas as pd

from . import panels
from .extended_factors import pead


def _buckets(frame: pd.DataFrame, characteristic: str, lower: float, upper: float) -> pd.DataFrame:
    """Independent NYSE size/characteristic breakpoints at each formation date."""
    pieces = []
    for date, g in frame.groupby("rebalance_date", sort=True):
        nyse = g[g["exchange_code"].eq(1)]
        if len(nyse) < 3:
            continue
        size_cut = nyse["market_equity"].median()
        lo, hi = nyse[characteristic].quantile([lower, upper])
        x = g.copy()
        x["size_bucket"] = np.where(x["market_equity"].le(size_cut), "S", "B")
        x["characteristic_bucket"] = np.select(
            [x[characteristic].le(lo), x[characteristic].gt(hi)], ["L", "H"], default="M"
        )
        x["group"] = x["size_bucket"] + x["characteristic_bucket"]
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _factor_from_groups(group_returns: pd.DataFrame, name: str) -> pd.DataFrame:
    wide = group_returns.pivot(index="date", columns="group", values="ret").reindex(
        columns=["SL", "SM", "SH", "BL", "BM", "BH"]
    )
    out = ((wide["SH"] + wide["BH"]) / 2 - (wide["SL"] + wide["BL"]) / 2).rename(name)
    return out.dropna().reset_index()


def _value_weighted_groups(joined: pd.DataFrame) -> pd.DataFrame:
    ok = joined["ret"].notna() & joined["weight"].gt(0)
    x = joined.loc[ok, ["date", "group", "ret", "weight"]].copy()
    x["weighted"] = x["ret"] * x["weight"]
    out = x.groupby(["date", "group"], as_index=False).agg(
        numerator=("weighted", "sum"), denominator=("weight", "sum"), n_assets=("ret", "count")
    )
    out["ret"] = out["numerator"] / out["denominator"]
    return out[["date", "group", "ret", "n_assets"]]


def annual_characteristic_factor(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exact annual-June 2x3 implementation for RMW, CMA, or NOA.

    The stored exposure is the signed characteristic (larger means the factor's long
    side); group and factor returns use NYSE breakpoints and monthly lagged-ME weights.
    """
    lag_cols = ["atq"] if kind in {"cma", "noa"} else []
    a = panels.annual_accounting_panel(lag_cols=lag_cols).copy()
    june = panels.crsp_monthly()[["date", "asset_id", "exchange_code", "siccd"]].rename(
        columns={"date": "rebalance_date"}
    )
    a = a.merge(june, on=["rebalance_date", "asset_id"], how="left")
    a = a[~panels.is_financial(a["siccd"])]
    if kind == "rmw":
        a["characteristic"] = a["operating_profit_q_ttm"] / a["book_equity"].where(a["book_equity"].gt(0))
        name = "RMW"
    elif kind == "cma":
        investment = a["atq"] / a["lag_atq"].where(a["lag_atq"].gt(0)) - 1
        a["characteristic"] = -investment
        name = "CMA"
    elif kind == "noa":
        operating_assets = a["atq"] - a["cheq"].fillna(0)
        operating_liabilities = (
            a["atq"] - a["dlcq"].fillna(0) - a["dlttq"].fillna(0) - a["mibq"].fillna(0)
            - a["pstkq"].fillna(0) - a["ceqq"].fillna(0)
        )
        noa = (operating_assets - operating_liabilities) / a["lag_atq"].where(a["lag_atq"].gt(0))
        a["characteristic"] = -noa
        name = "LOW_NOA"
    else:
        raise ValueError(f"Unknown annual characteristic: {kind}")
    a = a.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["characteristic", "market_equity", "exchange_code"]
    )
    a = _buckets(a, "characteristic", 0.3, 0.7)
    a["formation_year"] = a["rebalance_date"].dt.year

    m = panels.crsp_monthly().copy()
    m["formation_year"] = np.where(m["date"].dt.month.ge(7), m["date"].dt.year, m["date"].dt.year - 1)
    m["weight"] = m.groupby("asset_id")["market_equity"].shift(1)
    joined = m.merge(a[["formation_year", "asset_id", "group"]], on=["formation_year", "asset_id"], how="inner")
    groups = _value_weighted_groups(joined)
    factors = _factor_from_groups(groups, name)
    exposure = a[["rebalance_date", "asset_id", "characteristic"]].rename(
        columns={"rebalance_date": "date", "characteristic": "exposure"}
    )
    return exposure, a, groups.merge(factors, on="date", how="left")


def pead_factor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DHS monthly PEAD 2x3 factor using 20/80 NYSE CAR breakpoints."""
    exposure, events = pead()
    m = panels.crsp_monthly()
    formed = exposure.rename(columns={"date": "rebalance_date", "exposure": "characteristic"}).merge(
        m[["date", "asset_id", "market_equity", "exchange_code", "siccd"]].rename(columns={"date": "rebalance_date"}),
        on=["rebalance_date", "asset_id"], how="inner"
    )
    formed = formed[~panels.is_financial(formed["siccd"])]
    formed = _buckets(formed.dropna(subset=["characteristic", "market_equity", "exchange_code"]), "characteristic", 0.2, 0.8)
    formed["date"] = formed["rebalance_date"] + pd.offsets.MonthEnd(1)
    formed["weight"] = formed["market_equity"]
    next_ret = m[["date", "asset_id", "ret"]]
    joined = formed.merge(next_ret, on=["date", "asset_id"], how="left")
    groups = _value_weighted_groups(joined)
    factor = _factor_from_groups(groups, "PEAD")
    details = formed.merge(events, left_on=["rebalance_date", "asset_id"], right_on=["date", "asset_id"], how="left")
    return exposure, details, groups.merge(factor, on="date", how="left")


def bab_factor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frazzini-Pedersen rank-weighted, beta-scaled monthly BAB factor."""
    from . import data

    beta = data.load_base("extended_daily_characteristics", columns=["date", "asset_id", "beta_fp"])
    beta["date"] = pd.to_datetime(beta["date"])
    beta["shrunk_beta"] = 0.6 * beta["beta_fp"] + 0.4
    m = panels.crsp_monthly()
    formed = beta.merge(m[["date", "asset_id", "market_equity"]], on=["date", "asset_id"], how="inner")
    formed = formed.dropna(subset=["shrunk_beta"])
    formed["rank"] = formed.groupby("date")["shrunk_beta"].rank(method="average", pct=True)
    formed["centered"] = formed["rank"] - formed.groupby("date")["rank"].transform("mean")
    formed["low_raw"] = (-formed["centered"]).clip(lower=0)
    formed["high_raw"] = formed["centered"].clip(lower=0)
    for leg in ("low", "high"):
        formed[f"w_{leg}"] = formed[f"{leg}_raw"] / formed.groupby("date")[f"{leg}_raw"].transform("sum")
    formed["beta_low_part"] = formed["w_low"] * formed["shrunk_beta"]
    formed["beta_high_part"] = formed["w_high"] * formed["shrunk_beta"]
    formed["beta_low"] = formed.groupby("date")["beta_low_part"].transform("sum")
    formed["beta_high"] = formed.groupby("date")["beta_high_part"].transform("sum")
    formed["return_date"] = formed["date"] + pd.offsets.MonthEnd(1)
    joined = formed.merge(m[["date", "asset_id", "ret"]], left_on=["return_date", "asset_id"], right_on=["date", "asset_id"], how="left", suffixes=("_formation", ""))
    rf = panels.risk_free().rename(columns={"RF": "rf"})
    joined = joined.merge(rf, on="date", how="left")
    rows = []
    for date, g in joined.groupby("date"):
        valid = g["ret"].notna()
        if valid.sum() < 10:
            continue
        # Preserve the paper's formation weights; missing returns receive no ex-post redistribution.
        low = (g.loc[valid, "w_low"] * g.loc[valid, "ret"]).sum()
        high = (g.loc[valid, "w_high"] * g.loc[valid, "ret"]).sum()
        b_low, b_high = g["beta_low"].iloc[0], g["beta_high"].iloc[0]
        risk_free = g["rf"].iloc[0]
        bab = (low - risk_free) / b_low - (high - risk_free) / b_high
        rows.append({"date": date, "low_ret": low, "high_ret": high, "beta_low": b_low, "beta_high": b_high, "BAB": bab})
    factor = pd.DataFrame(rows)
    exposure = formed[["date", "asset_id", "shrunk_beta"]].assign(exposure=lambda x: -x["shrunk_beta"])[["date", "asset_id", "exposure"]]
    return exposure, formed, factor
