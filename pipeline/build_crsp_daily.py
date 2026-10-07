"""CRSP daily CIZ v2 -> year-partitioned data/base/crsp_daily/*.parquet.

The table is the point-in-time bridge needed by weekly and intraday factors:
PERMNO, historical trading symbol, CRSP total return, price and market equity.
Only active US common equities on NYSE / AMEX / NASDAQ are retained.

    python pipeline/build_crsp_daily.py --start-year 2008 --end-year 2025
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import BASE, CRSP_DAILY_ZIP, NestedZipFirstCsvReader


USECOLS = [
    "PERMNO",
    "PERMCO",
    "DlyCalDt",
    "DlyRet",
    "DlyPrc",
    "DlyCap",
    "DlyVol",
    "PrimaryExch",
    "SecurityType",
    "SecuritySubType",
    "USIncFlg",
    "TradingStatusFlg",
    "ConditionalType",
    "SecurityBegDt",
    "Ticker",
    "TradingSymbol",
]

SCHEMA = pa.schema(
    [
        ("asset_id", pa.int64()),
        ("company_id", pa.int64()),
        ("date", pa.timestamp("ns")),
        ("symbol", pa.string()),
        ("ret", pa.float64()),
        ("price", pa.float64()),
        ("market_equity", pa.float64()),
        ("volume", pa.float64()),
        ("exchange_code", pa.int8()),
        ("first_trade_date", pa.timestamp("ns")),
    ]
)


def clean_chunk(chunk: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter one raw CRSP chunk and standardize its point-in-time fields."""
    x = chunk.rename(
        columns={
            "PERMNO": "asset_id",
            "PERMCO": "company_id",
            "DlyCalDt": "date",
            "DlyRet": "ret",
            "DlyPrc": "price",
            "DlyCap": "market_equity",
            "DlyVol": "volume",
            "PrimaryExch": "primary_exchange",
            "SecurityBegDt": "first_trade_date",
        }
    ).copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x[
        x["date"].dt.year.between(start_year, end_year)
        & x["SecurityType"].eq("EQTY")
        & x["SecuritySubType"].eq("COM")
        & x["USIncFlg"].eq("Y")
        & x["TradingStatusFlg"].eq("A")
        & x["ConditionalType"].eq("RW")
        & x["primary_exchange"].isin(["N", "A", "Q"])
    ].copy()
    if x.empty:
        return pd.DataFrame(columns=SCHEMA.names)

    trading = x["TradingSymbol"].astype("string").str.strip().str.upper()
    ticker = x["Ticker"].astype("string").str.strip().str.upper()
    x["symbol"] = trading.where(trading.notna() & trading.ne(""), ticker)
    x["asset_id"] = pd.to_numeric(x["asset_id"], errors="coerce")
    x["company_id"] = pd.to_numeric(x["company_id"], errors="coerce")
    for col in ("ret", "price", "market_equity", "volume"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["price"] = x["price"].abs()
    x["first_trade_date"] = pd.to_datetime(x["first_trade_date"], errors="coerce")
    x["exchange_code"] = x["primary_exchange"].map({"N": 1, "A": 2, "Q": 3})
    x = x.dropna(subset=["asset_id", "company_id", "date", "symbol", "exchange_code"])
    x["asset_id"] = x["asset_id"].astype("int64")
    x["company_id"] = x["company_id"].astype("int64")
    x["exchange_code"] = x["exchange_code"].astype("int8")
    return x[SCHEMA.names]


def build(start_year: int, end_year: int, chunksize: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / f"{year}.parquet" for year in range(start_year, end_year + 1) if (output / f"{year}.parquet").exists()]
    if existing:
        raise FileExistsError(f"CRSP daily output already exists (first: {existing[0]}). Remove it explicitly before rebuilding.")

    writers: dict[int, pq.ParquetWriter] = {}
    counts: dict[int, int] = {}
    text = io.TextIOWrapper(NestedZipFirstCsvReader(CRSP_DAILY_ZIP), encoding="utf-8", newline="")
    try:
        for i, chunk in enumerate(pd.read_csv(text, usecols=USECOLS, chunksize=chunksize, low_memory=False)):
            clean = clean_chunk(chunk, start_year, end_year)
            if not clean.empty:
                for year, part in clean.groupby(clean["date"].dt.year, sort=False):
                    year = int(year)
                    table = pa.Table.from_pandas(part, schema=SCHEMA, preserve_index=False)
                    if year not in writers:
                        writers[year] = pq.ParquetWriter(output / f"{year}.parquet", SCHEMA, compression="zstd")
                        counts[year] = 0
                    writers[year].write_table(table)
                    counts[year] += len(part)
            if i % 25 == 0:
                kept = sum(counts.values())
                print(f"CRSP chunks read: {i:,}; daily rows kept: {kept:,}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        text.close()
    missing = sorted(set(range(start_year, end_year + 1)).difference(counts))
    if missing:
        raise RuntimeError(f"No CRSP daily rows written for years: {missing}")
    # The CIZ source occasionally repeats an identical security-day around a
    # chunk boundary/distribution record. Collapse these before downstream
    # return compounding or ticker matching.
    final_rows = 0
    for year in range(start_year, end_year + 1):
        path = output / f"{year}.parquet"
        frame = pd.read_parquet(path)
        frame = frame.sort_values(["asset_id", "date"]).drop_duplicates(["asset_id", "date"], keep="last")
        temp = output / f".{year}.deduplicated.parquet"
        frame.to_parquet(temp, index=False, compression="zstd")
        temp.replace(path)
        final_rows += len(frame)
    print(f"Wrote {final_rows:,} deduplicated rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=BASE / "crsp_daily")
    args = parser.parse_args()
    build(args.start_year, args.end_year, args.chunksize, args.output)


if __name__ == "__main__":
    main()
