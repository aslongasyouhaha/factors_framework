"""CRSP daily (CIZ v2) -> data/base/crsp_monthly.parquet.

Monthly returns compounded from daily returns for US common stocks on NYSE / AMEX / NASDAQ,
with company (PERMCO) level market equity assigned to each company's largest share class
(ff_primary_share), as in Fama-French. Market equity and dollar volume are in USD thousands.
siccd is the historical CRSP SIC code at the month's last trading day (NaN when unknown).

    python pipeline/build_crsp_monthly.py --start-year 2016 --end-year 2025
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from common import CRSP_DAILY_ZIP, NestedZipFirstCsvReader, save_base

USECOLS = [
    "PERMNO", "PERMCO", "DlyCalDt", "DlyRet", "DlyRetx", "DlyPrc", "DlyCap", "DlyVol", "PrimaryExch",
    "SecurityType", "SecuritySubType", "ShareType", "USIncFlg", "TradingStatusFlg", "ConditionalType", "SecurityBegDt",
    "SICCD",
]


def exchange_code(primary_exchange: pd.Series) -> pd.Series:
    return primary_exchange.map({"N": 1, "A": 2, "Q": 3}).astype("float")


def monthly_from_chunk(chunk: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame | None:
    chunk = chunk.rename(
        columns={
            "PERMNO": "asset_id",
            "PERMCO": "company_id",
            "DlyCalDt": "date",
            "DlyRet": "ret",
            "DlyRetx": "retx",
            "DlyPrc": "price",
            "DlyCap": "market_equity",
            "DlyVol": "volume",
            "PrimaryExch": "primary_exchange",
            "SecurityBegDt": "first_trade_date",
            "SICCD": "siccd",
        }
    )
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk = chunk[chunk["date"].dt.year.between(start_year, end_year)]
    chunk = chunk[
        chunk["SecurityType"].eq("EQTY")
        & chunk["SecuritySubType"].eq("COM")
        & chunk["USIncFlg"].eq("Y")
        & chunk["TradingStatusFlg"].eq("A")
        & chunk["ConditionalType"].eq("RW")
        & chunk["primary_exchange"].isin(["N", "A", "Q"])
    ].copy()
    if chunk.empty:
        return None
    chunk["ret"] = pd.to_numeric(chunk["ret"], errors="coerce")
    chunk["retx"] = pd.to_numeric(chunk["retx"], errors="coerce")
    chunk["company_id"] = pd.to_numeric(chunk["company_id"], errors="coerce")
    chunk["market_equity"] = pd.to_numeric(chunk["market_equity"], errors="coerce")
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").abs()
    chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce")
    # Historical SIC code on each date; 0 means unknown.
    chunk["siccd"] = pd.to_numeric(chunk["siccd"], errors="coerce").where(lambda x: x.gt(0))
    chunk = chunk.dropna(subset=["company_id"])
    chunk["company_id"] = chunk["company_id"].astype("int64")
    chunk["exchange_code"] = exchange_code(chunk["primary_exchange"])
    chunk["month"] = chunk["date"] + pd.offsets.MonthEnd(0)
    chunk["ret_gross"] = 1.0 + chunk["ret"]
    chunk["retx_gross"] = 1.0 + chunk["retx"]
    chunk["dollar_volume"] = chunk["price"] * chunk["volume"]
    monthly = (
        chunk.sort_values(["asset_id", "date"])
        .groupby(["asset_id", "month"], as_index=False)
        .agg(
            company_id=("company_id", "last"),
            ret_gross=("ret_gross", "prod"),
            retx_gross=("retx_gross", "prod"),
            n_ret=("ret", "count"),
            last_trade_date=("date", "last"),
            market_equity=("market_equity", "last"),
            exchange_code=("exchange_code", "last"),
            dollar_volume=("dollar_volume", "mean"),
            first_trade_date=("first_trade_date", "first"),
            siccd=("siccd", "last"),
        )
    )
    monthly["ret"] = monthly["ret_gross"] - 1.0
    monthly["retx"] = monthly["retx_gross"] - 1.0
    return monthly.drop(columns=["ret_gross", "retx_gross"])


def assign_permco_market_equity(monthly: pd.DataFrame) -> pd.DataFrame:
    """Sum market equity across a company's share classes and flag its largest security."""
    out = monthly.copy()
    out["company_market_equity"] = out.groupby(["date", "company_id"])["market_equity"].transform("sum")
    rank = out.groupby(["date", "company_id"])["market_equity"].rank(method="first", ascending=False)
    out["ff_primary_share"] = rank.eq(1)
    out["market_equity_security"] = out["market_equity"]
    out["market_equity"] = out["company_market_equity"]
    return out


def build(start_year: int, end_year: int, chunksize: int) -> pd.DataFrame:
    parts = []
    text = io.TextIOWrapper(NestedZipFirstCsvReader(CRSP_DAILY_ZIP), encoding="utf-8", newline="")
    for i, chunk in enumerate(pd.read_csv(text, usecols=USECOLS, chunksize=chunksize, low_memory=False)):
        part = monthly_from_chunk(chunk, start_year, end_year)
        if part is not None:
            parts.append(part)
        if i % 25 == 0:
            print(f"CRSP chunks read: {i:,}; monthly parts: {len(parts):,}", flush=True)
    if not parts:
        raise RuntimeError("No CRSP rows found for the requested years.")
    # A security-month can straddle two chunks: combine the partial months.
    monthly = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["asset_id", "month", "last_trade_date"])
        .groupby(["asset_id", "month"], as_index=False)
        .agg(
            company_id=("company_id", "last"),
            ret=("ret", lambda x: (1.0 + x).prod() - 1.0),
            retx=("retx", lambda x: (1.0 + x).prod() - 1.0),
            n_ret=("n_ret", "sum"),
            last_trade_date=("last_trade_date", "last"),
            market_equity=("market_equity", "last"),
            exchange_code=("exchange_code", "last"),
            dollar_volume=("dollar_volume", "mean"),
            first_trade_date=("first_trade_date", "first"),
            siccd=("siccd", "last"),
        )
        .rename(columns={"month": "date"})
    )
    monthly["first_trade_date"] = pd.to_datetime(monthly["first_trade_date"], errors="coerce")
    return assign_permco_market_equity(monthly)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    monthly = build(args.start_year, args.end_year, args.chunksize)
    path = save_base(monthly, "crsp_monthly")
    print(f"Wrote {len(monthly):,} rows ({monthly['date'].min():%Y-%m} to {monthly['date'].max():%Y-%m}) to {path}")


if __name__ == "__main__":
    main()
