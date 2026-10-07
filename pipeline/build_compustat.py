"""Compustat quarterly + CCM link -> data/base/compustat_quarterly.parquet and data/base/ccm_link.parquet.

compustat_quarterly keeps every quarter (industrial, consolidated, standard format) with:
- raw balance-sheet items and book equity (Fama-French definition from quarterly items);
- trailing-four-quarter flows (*_ttm), valid only when the four quarters span about a year;
- rdq (earnings announcement date) and available_date, the first date the quarter can be used:
  rdq when it is on or after the fiscal quarter end, otherwise datadate + 90 days.
Values are in USD millions. Ratios and growth rates are factor definitions and belong to projects.

    python pipeline/build_compustat.py --start-year 2004 --end-year 2026
"""
from __future__ import annotations

import argparse
import zipfile

import pandas as pd

from common import CCM_LINK_ZIP, COMPUSTAT_QUARTERLY_ZIP, save_base
from factorlab import FactorBuildTools

ITEMS = ["atq", "ltq", "seqq", "ceqq", "txditcq", "pstkrq", "pstkq", "saleq", "cogsq", "xsgaq",
         "xintq", "mibq", "niq", "ibq", "oiadpq", "cheq", "dlttq", "dlcq", "actq", "lctq", "dpq", "capxy",
         "oancfy", "cshoq", "ajexq", "epspxq", "rectq", "invtq", "ppentq", "apq", "txpq", "xrdq", "revtq"]
FLOWS = ["saleq", "gross_profit_q", "operating_profit_q", "net_income_q", "oiadpq", "dpq", "capx_q"]
FALLBACK_DAYS = 90       # availability lag when rdq is missing or earlier than the quarter end
MAX_TTM_SPAN_DAYS = 300  # datadate[t] - datadate[t-3]: about 9 months for four consecutive quarters


def build_quarterly(start_year: int, end_year: int, chunksize: int) -> pd.DataFrame:
    cols = ["gvkey", "datadate", "rdq", "fyearq", "fqtr", "indfmt", "consol", "datafmt", *ITEMS]
    parts = []
    with zipfile.ZipFile(COMPUSTAT_QUARTERLY_ZIP) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")][0]
        with z.open(csv_name) as f:
            for chunk in pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False):
                chunk = chunk[
                    chunk["fyearq"].between(start_year, end_year)
                    & chunk["indfmt"].eq("INDL")
                    & chunk["consol"].eq("C")
                    & chunk["datafmt"].eq("STD")
                ]
                if not chunk.empty:
                    parts.append(chunk.copy())
    if not parts:
        raise RuntimeError("No Compustat quarterly rows for the requested years.")

    return add_derived_fields(pd.concat(parts, ignore_index=True))


def add_derived_fields(q: pd.DataFrame) -> pd.DataFrame:
    """Types, single-quarter capex, trailing-four-quarter flows, book equity and availability dates."""
    q = q.copy()
    q["gvkey"] = q["gvkey"].astype("string").str.zfill(6)
    q["datadate"] = pd.to_datetime(q["datadate"])
    q["rdq"] = pd.to_datetime(q["rdq"], errors="coerce")
    for col in ["fyearq", "fqtr", *ITEMS]:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    g = q.groupby("gvkey")

    q["gross_profit_q"] = q["saleq"] - q["cogsq"]
    # French's OP numerator: revenue minus COGS, SG&A and interest.  Revenue is
    # required and at least one expense must be reported; missing remaining
    # expenses are treated as zero, matching the data-library convention.
    expenses = q[["cogsq", "xsgaq", "xintq"]]
    q["operating_profit_q"] = (
        q["saleq"] - expenses.fillna(0).sum(axis=1)
    ).where(q["saleq"].notna() & expenses.notna().any(axis=1))
    q["net_income_q"] = q["niq"].combine_first(q["ibq"])
    # capxy is year-to-date within the fiscal year: difference consecutive quarters of a year.
    prev_capx, prev_fqtr, prev_fyear = g["capxy"].shift(1), g["fqtr"].shift(1), g["fyearq"].shift(1)
    same_year = prev_fyear.eq(q["fyearq"]) & prev_fqtr.eq(q["fqtr"] - 1)
    q["capx_q"] = q["capxy"].where(q["fqtr"].eq(1), (q["capxy"] - prev_capx).where(same_year))

    span_ok = (q["datadate"] - g["datadate"].shift(3)).dt.days.le(MAX_TTM_SPAN_DAYS)
    for col in FLOWS:
        ttm = q.groupby("gvkey")[col].transform(lambda x: x.rolling(4, min_periods=4).sum())
        q[f"{col}_ttm"] = ttm.where(span_ok)

    q["book_equity"] = FactorBuildTools.book_equity(
        seq=q["seqq"],
        ceq=q["ceqq"],
        txditc=q["txditcq"],
        pstkrv=q["pstkrq"],
        pstkl=None,
        pstk=q["pstkq"],
        at=q["atq"],
        lt=q["ltq"],
    )
    valid_rdq = q["rdq"].ge(q["datadate"])
    q["available_date"] = q["rdq"].where(valid_rdq, q["datadate"] + pd.Timedelta(days=FALLBACK_DAYS))
    q["rdq_imputed"] = ~valid_rdq
    q = q[q["fyearq"].notna() & q["fqtr"].notna()].copy()
    q["fyear"] = q["fyearq"].astype("int64")
    q["fqtr"] = q["fqtr"].astype("int64")
    keep = ["gvkey", "datadate", "fyear", "fqtr", "rdq", "available_date", "rdq_imputed", "book_equity",
            *ITEMS, "capx_q", *[f"{c}_ttm" for c in FLOWS]]
    return q[keep].reset_index(drop=True)


def build_link() -> pd.DataFrame:
    usecols = ["gvkey", "LINKPRIM", "LINKTYPE", "LPERMNO", "LINKDT", "LINKENDDT"]
    with zipfile.ZipFile(CCM_LINK_ZIP) as z:
        with z.open(z.namelist()[0]) as f:
            link = pd.read_csv(f, usecols=usecols, dtype={"gvkey": "string"})
    link = link[link["LPERMNO"].notna() & link["LINKTYPE"].isin(["LU", "LC"]) & link["LINKPRIM"].isin(["P", "C"])].copy()
    link["gvkey"] = link["gvkey"].astype("string").str.zfill(6)
    link["asset_id"] = link["LPERMNO"].astype("int64")
    link["linkdt"] = pd.to_datetime(link["LINKDT"])
    link["linkenddt"] = pd.to_datetime(link["LINKENDDT"].replace("E", "2262-04-11"))
    return link[["gvkey", "asset_id", "linkdt", "linkenddt", "LINKTYPE", "LINKPRIM"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    q = build_quarterly(args.start_year, args.end_year, args.chunksize)
    imputed = q["rdq_imputed"].mean()
    print(f"Wrote {len(q):,} quarterly rows ({imputed:.1%} without a usable rdq) to {save_base(q, 'compustat_quarterly')}")
    link = build_link()
    print(f"Wrote {len(link):,} CCM link rows to {save_base(link, 'ccm_link')}")


if __name__ == "__main__":
    main()
