from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, panels
from .fama_french import link_to_permno
from .paper_portfolios import _value_weighted_groups


def sue() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = panels.compustat_quarterly().sort_values(["gvkey", "datadate"]).copy()
    q = q[q["rdq"].notna() & ~q["rdq_imputed"]].copy()
    q["adjusted_eps"] = q["epspxq"] / q["ajexq"].where(q["ajexq"].gt(0))
    q["unexpected_eps"] = q["adjusted_eps"] - q.groupby("gvkey")["adjusted_eps"].shift(4)
    q["ue_scale"] = q.groupby("gvkey")["unexpected_eps"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=4).std()
    )
    q["sue"] = q["unexpected_eps"] / q["ue_scale"].replace(0, np.nan)
    q = link_to_permno(q, data.load_base("ccm_link"), key="datadate")
    months = panels.crsp_monthly()[["date", "asset_id"]]
    cols = ["asset_id", "available_date", "datadate", "adjusted_eps", "unexpected_eps", "ue_scale", "sue"]
    out = pd.merge_asof(months.sort_values("date"), q[cols].sort_values("available_date"),
                        left_on="date", right_on="available_date", by="asset_id", direction="backward")
    fresh = out["date"].between(out["available_date"], out["available_date"] + pd.DateOffset(months=6))
    out = out[fresh & out["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)].copy()
    return out[["date", "asset_id", "sue"]].rename(columns={"sue": "exposure"}), out


def piotroski_fscore() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lag = ["atq", "net_income_q_ttm", "dlttq", "actq", "lctq", "gross_profit_q_ttm",
           "saleq_ttm", "cshoq", "ajexq"]
    a = panels.annual_accounting_panel(lag_cols=lag).copy()
    lag_at = a["lag_atq"].where(a["lag_atq"].gt(0))
    roa = a["net_income_q_ttm"] / lag_at
    lag_roa = a["lag_net_income_q_ttm"] / lag_at
    cfo = a["oancfy"] / lag_at
    leverage = a["dlttq"] / a["atq"].where(a["atq"].gt(0))
    lag_leverage = a["lag_dlttq"] / lag_at
    current = a["actq"] / a["lctq"].where(a["lctq"].gt(0))
    lag_current = a["lag_actq"] / a["lag_lctq"].where(a["lag_lctq"].gt(0))
    shares = a["cshoq"] * a["ajexq"]
    lag_shares = a["lag_cshoq"] * a["lag_ajexq"]
    margin = a["gross_profit_q_ttm"] / a["saleq_ttm"].where(a["saleq_ttm"].ne(0))
    lag_margin = a["lag_gross_profit_q_ttm"] / a["lag_saleq_ttm"].where(a["lag_saleq_ttm"].ne(0))
    turnover = a["saleq_ttm"] / a["atq"].where(a["atq"].gt(0))
    lag_turnover = a["lag_saleq_ttm"] / lag_at
    signals = pd.DataFrame({
        "f_roa": roa.gt(0), "f_cfo": cfo.gt(0), "f_droa": roa.gt(lag_roa), "f_accrual": cfo.gt(roa),
        "f_dlever": leverage.lt(lag_leverage), "f_dliquid": current.gt(lag_current),
        "f_no_issue": shares.le(lag_shares), "f_dmargin": margin.gt(lag_margin),
        "f_dturn": turnover.gt(lag_turnover),
    }, index=a.index)
    complete = pd.concat([roa, cfo, lag_roa, leverage, lag_leverage, current, lag_current,
                          shares, lag_shares, margin, lag_margin, turnover, lag_turnover], axis=1).notna().all(axis=1)
    a["f_score"] = signals.sum(axis=1).where(complete)
    # Original strategy operates inside the highest book-to-market quintile.
    a["value_cut"] = a[a["exchange_code"].eq(1)].groupby("rebalance_date")["bm"].transform("quantile", q=.8)
    cuts = a[a["exchange_code"].eq(1)].groupby("rebalance_date")["bm"].quantile(.8)
    a["value_cut"] = a["rebalance_date"].map(cuts)
    a = a[a["bm"].ge(a["value_cut"]) & a["f_score"].notna()].copy()
    a["formation_year"] = a["rebalance_date"].dt.year
    a["group"] = np.select([a["f_score"].le(1), a["f_score"].ge(8)], ["LOW", "HIGH"], default="MID")
    m = panels.crsp_monthly().copy()
    m["formation_year"] = np.where(m["date"].dt.month.ge(7), m["date"].dt.year, m["date"].dt.year - 1)
    joined = m.merge(a[["formation_year", "asset_id", "group"]], on=["formation_year", "asset_id"], how="inner")
    joined["weight"] = 1.0
    groups = _value_weighted_groups(joined)
    wide = groups.pivot(index="date", columns="group", values="ret")
    factor = (wide.get("HIGH") - wide.get("LOW")).rename("F_SCORE").dropna().reset_index()
    exposure = a[["rebalance_date", "asset_id", "f_score"]].rename(columns={"rebalance_date":"date","f_score":"exposure"})
    return exposure, a, factor


def net_share_issuance() -> pd.DataFrame:
    m = panels.crsp_monthly()
    if "retx" not in m.columns:
        raise ValueError("crsp_monthly requires retx; rebuild with pipeline/build_crsp_monthly.py")
    me = m["market_equity_security"].where(m["market_equity_security"].gt(0))
    log_me = np.log(me)
    log_price_return = np.log1p(m["retx"].where(m["retx"].gt(-1)))
    # Pontiff-Woodgate: real share change over t-18 to t-6.
    issuance = log_me.groupby(m["asset_id"]).shift(6) - log_me.groupby(m["asset_id"]).shift(18)
    price_growth = log_price_return.groupby(m["asset_id"]).shift(6).groupby(m["asset_id"]).rolling(12, min_periods=12).sum().reset_index(level=0, drop=True)
    m["exposure"] = -(issuance - price_growth)
    return m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date","asset_id","exposure"]]


def weekly_realized_moment(column: str, direction: float) -> pd.DataFrame:
    files = sorted((data.BASE / "rsj_daily").glob("????Q?.parquet"))
    d = pd.concat([pd.read_parquet(p, columns=["date","asset_id",column]) for p in files], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["asset_id","date"])
    d["signal"] = d.groupby("asset_id")[column].transform(lambda x: x.rolling(5, min_periods=5).mean())
    dates = pd.Index(d["date"].drop_duplicates().sort_values())
    d["trade_index"] = dates.get_indexer(d["date"])
    first = d.groupby("asset_id")["trade_index"].shift(4)
    d = d[d["date"].dt.weekday.eq(1) & d["trade_index"].sub(first).eq(4)].copy()
    d["exposure"] = direction * d["signal"]
    return d[["date","asset_id","exposure"]]


def q_factor_model() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """HXZ 2x3x3 size, investment-to-assets and monthly ROE portfolios."""
    annual = panels.annual_accounting_panel(lag_cols=["atq"]).copy()
    annual["investment"] = annual["atq"] / annual["lag_atq"].where(annual["lag_atq"].gt(0)) - 1
    june_sic = panels.crsp_monthly()[["date","asset_id","siccd"]].rename(columns={"date":"rebalance_date"})
    annual = annual.merge(june_sic,on=["rebalance_date","asset_id"],how="left")
    annual = annual[~panels.is_financial(annual["siccd"])].dropna(subset=["investment","market_equity","exchange_code"])
    parts=[]
    for date,g in annual.groupby("rebalance_date"):
        nyse=g[g["exchange_code"].eq(1)]; size=nyse["market_equity"].median(); lo,hi=nyse["investment"].quantile([.3,.7])
        x=g.copy(); x["size_bucket"]=np.where(x["market_equity"].le(size),"S","B")
        x["inv_bucket"]=np.select([x["investment"].le(lo),x["investment"].gt(hi)],["L","H"],default="M"); parts.append(x)
    annual=pd.concat(parts,ignore_index=True); annual["formation_year"]=annual["rebalance_date"].dt.year

    q=panels.compustat_quarterly().sort_values(["gvkey","datadate"]).copy()
    gap=(q["datadate"]-q.groupby("gvkey")["datadate"].shift(1)).dt.days.between(60,130)
    q["roe"]=q["ibq"] / q.groupby("gvkey")["book_equity"].shift(1).where(gap & q.groupby("gvkey")["book_equity"].shift(1).gt(0))
    q=q[q["rdq"].notna() & ~q["rdq_imputed"]]
    q=link_to_permno(q,data.load_base("ccm_link"),key="datadate")
    m=panels.crsp_monthly(); sample=m[m["date"].between(panels.SAMPLE_START,panels.SAMPLE_END)].copy()
    formed=pd.merge_asof(sample.sort_values("date"),q[["asset_id","available_date","datadate","roe"]].sort_values("available_date"),left_on="date",right_on="available_date",by="asset_id",direction="backward")
    formed=formed[(formed["date"]-formed["datadate"]).dt.days.le(180)]
    formed["formation_year"]=np.where(formed["date"].dt.month.ge(7),formed["date"].dt.year,formed["date"].dt.year-1)
    formed=formed.merge(annual[["formation_year","asset_id","size_bucket","inv_bucket"]],on=["formation_year","asset_id"],how="inner").dropna(subset=["roe"])
    out=[]
    for date,g in formed.groupby("date"):
        nyse=g[g["exchange_code"].eq(1)]; lo,hi=nyse["roe"].quantile([.3,.7]); x=g.copy()
        x["roe_bucket"]=np.select([x["roe"].le(lo),x["roe"].gt(hi)],["L","H"],default="M")
        x["group"]=x["size_bucket"]+x["inv_bucket"]+x["roe_bucket"]; out.append(x)
    formed=pd.concat(out,ignore_index=True); formed["return_date"]=formed["date"]+pd.offsets.MonthEnd(1); formed["weight"]=formed["market_equity"]
    joined=formed.merge(m[["date","asset_id","ret"]],left_on=["return_date","asset_id"],right_on=["date","asset_id"],suffixes=("_formation",""))
    groups=_value_weighted_groups(joined)
    wide=groups.pivot(index="date",columns="group",values="ret")
    small=[c for c in wide if c.startswith("S")]; big=[c for c in wide if c.startswith("B")]
    low_i=[c for c in wide if c[1]=="L"]; high_i=[c for c in wide if c[1]=="H"]
    low_r=[c for c in wide if c[2]=="L"]; high_r=[c for c in wide if c[2]=="H"]
    factors=pd.DataFrame({"r_ME":wide[small].mean(axis=1)-wide[big].mean(axis=1),"r_IA":wide[low_i].mean(axis=1)-wide[high_i].mean(axis=1),"r_ROE":wide[high_r].mean(axis=1)-wide[low_r].mean(axis=1)}).reset_index()
    ff=data.load_base("french_factors_monthly",columns=["date","MKT_RF"]); ff["date"]=pd.to_datetime(ff["date"])
    factors=factors.merge(ff,on="date",how="left")
    exposure=formed[["date","asset_id","roe"]].rename(columns={"roe":"exposure"})
    return exposure,formed,factors
