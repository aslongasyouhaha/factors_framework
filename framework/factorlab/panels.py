from __future__ import annotations

from typing import Sequence

import pandas as pd

from . import data
from .fama_french import annual_characteristics as _annual_characteristics
from .fama_french import link_to_permno
from .panel import to_wide

# Standard research panels assembled from data/base. Projects build their factors from these so
# that every project sees the same universe, timing and accounting conventions.

# Default research sample shared by the projects: monthly rebalancing dates and June formations.
SAMPLE_START = "2008-01-31"
SAMPLE_END = "2025-11-30"  # last rebalance with a following month of returns (CRSP ends 2025-12)
SAMPLE_YEARS = (2008, 2025)

FINANCIAL_SIC = (6000, 6999)  # banks, insurers, brokers, real estate and other financials
YEAR_AGO_DAYS = (330, 400)  # a "same quarter last year" row must be this far back
MAX_ACCOUNTING_AGE_DAYS = 366  # point-in-time data older than this (by fiscal period end) is stale


def crsp_monthly(primary_only: bool = True) -> pd.DataFrame:
    """CRSP monthly rows; by default one row per company (its largest share class)."""
    m = data.load_base("crsp_monthly")
    if primary_only:
        m = m[m["ff_primary_share"]].copy()
    m["date"] = pd.to_datetime(m["date"])
    for col in ["ret", "market_equity", "dollar_volume"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    return m.sort_values(["asset_id", "date"])


def is_financial(siccd: pd.Series) -> pd.Series:
    """True for financial firms by historical CRSP SIC code (6000-6999); unknown codes are False."""
    return siccd.between(*FINANCIAL_SIC)


def monthly_matrix(col: str, primary_only: bool = True) -> pd.DataFrame:
    """date x asset matrix of a CRSP monthly column, e.g. "ret" or "market_equity"."""
    return to_wide(crsp_monthly(primary_only), col)


def risk_free() -> pd.DataFrame:
    """Monthly risk-free rate (date, RF) in decimal per month (Kenneth French's T-bill return)."""
    rf = data.load_base("rf_monthly")
    rf["date"] = pd.to_datetime(rf["date"])
    return rf[["date", "RF"]]


def compustat_quarterly() -> pd.DataFrame:
    return data.load_base("compustat_quarterly").sort_values(["gvkey", "datadate"]).reset_index(drop=True)


def compustat_annual(lag_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Fiscal-year (fiscal Q4) accounting rows linked to permno (asset_id).

    lag_cols adds lag_<col>: the value on the company's previous fiscal-Q4 row, taken before
    duplicate fiscal years are dropped (the latest datadate of a fiscal year is kept).
    """
    q4 = compustat_quarterly()
    q4 = q4[q4["fqtr"].eq(4)].copy()
    for col in lag_cols:
        q4[f"lag_{col}"] = q4.groupby("gvkey")[col].shift(1)
    q4 = q4.sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    return link_to_permno(q4, data.load_base("ccm_link"))


def book_equity() -> pd.DataFrame:
    """Positive fiscal-year book equity linked to permno: (asset_id, fyear, datadate, available_date, book_equity)."""
    q4 = data.load_base("compustat_quarterly", columns=["gvkey", "datadate", "fyear", "fqtr", "available_date", "book_equity"])
    be = q4[q4["fqtr"].eq(4) & q4["book_equity"].gt(0)]
    be = be.sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    return link_to_permno(be, data.load_base("ccm_link"))[["asset_id", "fyear", "datadate", "available_date", "book_equity"]]


def annual_characteristics(start_year: int = SAMPLE_YEARS[0], end_year: int = SAMPLE_YEARS[1]) -> pd.DataFrame:
    """Fama-French June formation universe (see fama_french.annual_characteristics); book
    equity must have been announced by the June formation date."""
    return _annual_characteristics(crsp_monthly(primary_only=False), book_equity(), start_year, end_year)


def annual_accounting_panel(start_year: int = SAMPLE_YEARS[0], end_year: int = SAMPLE_YEARS[1], lag_cols: Sequence[str] = ()) -> pd.DataFrame:
    """June formation universe joined with the fiscal year ending in the previous calendar year,
    keeping only fiscal years announced by the June formation date."""
    chars = annual_characteristics(start_year, end_year)[["rebalance_date", "asset_id", "market_equity", "bm", "dec_market_equity"]]
    chars["be_year"] = chars["rebalance_date"].dt.year - 1
    acc = compustat_annual(lag_cols).rename(columns={"fyear": "be_year"}).drop(columns=["book_equity"])
    out = chars.merge(acc, on=["asset_id", "be_year"], how="inner")
    return out[out["available_date"].le(out["rebalance_date"])]


def point_in_time_quarters(year_ago_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Quarterly accounting rows linked to permno, in the order they became public.

    year_ago_cols adds yoy_<col>: the value of the same fiscal quarter one year earlier (the row
    four quarters back, if its fiscal period ended 330-400 days before). A row is dropped when a
    later fiscal quarter of the same stock was already public, so the latest public quarter never
    moves backwards in time.
    """
    q = compustat_quarterly()
    g = q.groupby("gvkey")
    gap = (q["datadate"] - g["datadate"].shift(4)).dt.days
    year_ago = gap.between(*YEAR_AGO_DAYS)
    for col in year_ago_cols:
        q[f"yoy_{col}"] = g[col].shift(4).where(year_ago)
    q = link_to_permno(q, data.load_base("ccm_link"), key="datadate")
    q = q.sort_values(["asset_id", "available_date", "datadate"])
    newest_known = q.groupby("asset_id")["datadate"].cummax()
    previous_known = newest_known.groupby(q["asset_id"]).shift(1)
    q = q[previous_known.isna() | q["datadate"].gt(previous_known)]
    return q.reset_index(drop=True)


def monthly_accounting_panel(
    start: str = SAMPLE_START,
    end: str = SAMPLE_END,
    year_ago_cols: Sequence[str] = (),
    max_age_days: int = MAX_ACCOUNTING_AGE_DAYS,
    exclude_financials: bool = False,
) -> pd.DataFrame:
    """Point-in-time accounting data at each month end.

    For every CRSP stock-month in [start, end] (one share class per company), the latest fiscal
    quarter whose available_date (announcement date, or fiscal period end + 90 days when unknown)
    is on or before the month end. Quarters whose fiscal period ended more than max_age_days
    earlier are treated as stale and dropped. exclude_financials drops stock-months whose CRSP SIC
    code at that month end is 6000-6999. Columns: date, asset_id, market_equity, siccd and every
    quarterly field (flows as *_ttm, balance-sheet items as of the quarter end, yoy_<col>).
    """
    crsp = crsp_monthly()
    crsp = crsp[crsp["date"].between(pd.Timestamp(start), pd.Timestamp(end))][["date", "asset_id", "market_equity", "siccd"]]
    if exclude_financials:
        crsp = crsp[~is_financial(crsp["siccd"])]
    acc = point_in_time_quarters(year_ago_cols)
    out = pd.merge_asof(
        crsp.sort_values("date"),
        acc.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        by="asset_id",
        direction="backward",
    )
    fresh = (out["date"] - out["datadate"]).dt.days.le(max_age_days)
    return out[fresh].sort_values(["date", "asset_id"]).reset_index(drop=True)
