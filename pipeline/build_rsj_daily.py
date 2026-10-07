"""Vendor one-minute OHLCV + point-in-time CRSP -> data/base/rsj_daily/*.parquet.

The vendor files are quarterly, symbol-sorted Parquet files. Processing is done
in symbol chunks so a 25+ GB source never needs to fit in memory.

    python pipeline/build_rsj_daily.py --source F:\\temp --start-year 2008 --end-year 2025
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import BASE


PROJECT = Path(__file__).resolve().parents[1] / "factors" / "bollerslev_li_zhao_2020_good_bad_volatility"
sys.path.insert(0, str(PROJECT))

from rsj import realized_measures_from_minute_ohlcv  # noqa: E402


FILE_RE = re.compile(r"^(\d{4})Q([1-4])\.parquet$", re.IGNORECASE)
MINUTE_COLUMNS = ["timestamp", "date", "minute_of_day", "symbol", "open", "close", "volume"]


def quarter_bounds(year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)
    return start, start + pd.offsets.QuarterEnd(0)


def session_open_utc_minute(dates: pd.Series) -> pd.Series:
    """UTC minute-of-day of 09:30 America/New_York, DST aware."""
    local = pd.to_datetime(dates).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    utc = local.dt.tz_convert("UTC")
    return (utc.dt.hour * 60 + utc.dt.minute).astype("int16")


def filter_bad_price_bars(
    bars: pd.DataFrame,
    *,
    min_reference_ratio: float,
    max_reference_ratio: float,
    max_local_ratio: float,
) -> tuple[pd.DataFrame, int, int]:
    """Remove vendor bad ticks using CRSP and a centered local price anchor.

    The CRSP screen catches decimal/place-scale errors. The local median screen
    catches isolated spikes that are still within the deliberately wide CRSP
    band, while retaining price moves that persist for subsequent observations.
    """
    floor = min_reference_ratio * bars["daily_price"]
    ceiling = max_reference_ratio * bars["daily_price"]
    reference_ok = bars["open"].between(floor, ceiling) & bars["close"].between(floor, ceiling)
    reference_dropped = int((~reference_ok).sum())
    out = bars[reference_ok].copy()
    out = out.sort_values(["asset_id", "date_key", "session_minute"], kind="stable")
    local_median = out.groupby(["asset_id", "date_key"], observed=True)["close"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).median()
    )
    local_ok = local_median.isna() | (
        out["open"].between(local_median / max_local_ratio, local_median * max_local_ratio)
        & out["close"].between(local_median / max_local_ratio, local_median * max_local_ratio)
    )
    local_dropped = int((~local_ok).sum())
    return out[local_ok].copy(), reference_dropped, local_dropped


def load_crsp_map(crsp_year: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    cols = ["asset_id", "date", "symbol", "price", "market_equity"]
    x = pd.read_parquet(crsp_year, columns=cols)
    x["date"] = pd.to_datetime(x["date"])
    x = x[x["date"].between(start, end) & x["price"].between(5.0, 1000.0)].copy()
    x["symbol"] = x["symbol"].astype("string").str.strip().str.upper()
    x["date_key"] = x["date"].dt.strftime("%Y%m%d").astype("int32")
    x = x.sort_values(["asset_id", "date"]).drop_duplicates(["asset_id", "date"], keep="last")

    key_counts = x.groupby(["symbol", "date_key"], observed=True)["asset_id"].nunique()
    ambiguous = key_counts[key_counts.gt(1)].index
    n_ambiguous = len(ambiguous)
    if n_ambiguous:
        bad = pd.MultiIndex.from_frame(x[["symbol", "date_key"]]).isin(ambiguous)
        x = x[~bad].copy()
    x["session_open_minute"] = session_open_utc_minute(x["date"])
    x = x.rename(columns={"price": "daily_price"})
    return x[
        ["asset_id", "date_key", "symbol", "daily_price", "market_equity", "session_open_minute"]
    ], n_ambiguous


def process_quarter(
    minute_file: Path,
    crsp_file: Path,
    output_file: Path,
    *,
    symbol_chunk_size: int,
    min_intraday_returns: int,
    min_reference_price_ratio: float,
    max_reference_price_ratio: float,
    max_local_price_ratio: float,
) -> pd.DataFrame:
    match = FILE_RE.match(minute_file.name)
    if not match:
        raise ValueError(f"Unexpected quarterly file name: {minute_file.name}")
    year, quarter = int(match.group(1)), int(match.group(2))
    start, end = quarter_bounds(year, quarter)
    crsp, n_ambiguous = load_crsp_map(crsp_file, start, end)
    symbols = sorted(crsp["symbol"].dropna().unique())
    pieces: list[pd.DataFrame] = []
    source_rows = matched_rows = reference_bad_price_rows = local_bad_price_rows = 0

    for offset in range(0, len(symbols), symbol_chunk_size):
        symbol_slice = symbols[offset : offset + symbol_chunk_size]
        table = pq.read_table(minute_file, columns=MINUTE_COLUMNS, filters=[("symbol", "in", symbol_slice)])
        if table.num_rows == 0:
            continue
        bars = table.to_pandas()
        source_rows += len(bars)
        bars["symbol"] = bars["symbol"].astype("string").str.strip().str.upper()
        bars = bars.rename(columns={"date": "date_key"})
        bars = bars.merge(crsp, on=["symbol", "date_key"], how="inner", validate="many_to_one")
        if bars.empty:
            continue
        minute_offset = bars["minute_of_day"].astype("int32") - bars["session_open_minute"].astype("int32")
        bars = bars[minute_offset.between(0, 390)].copy()
        bars["session_minute"] = minute_offset[minute_offset.between(0, 390)].to_numpy()
        bars, n_reference_bad, n_local_bad = filter_bad_price_bars(
            bars,
            min_reference_ratio=min_reference_price_ratio,
            max_reference_ratio=max_reference_price_ratio,
            max_local_ratio=max_local_price_ratio,
        )
        reference_bad_price_rows += n_reference_bad
        local_bad_price_rows += n_local_bad
        matched_rows += len(bars)
        daily = realized_measures_from_minute_ohlcv(
            bars,
            date_col="date_key",
            min_intraday_returns=min_intraday_returns,
        )
        if not daily.empty:
            pieces.append(daily)
        print(
            f"{minute_file.stem}: symbols {min(offset + symbol_chunk_size, len(symbols)):,}/{len(symbols):,}; "
            f"daily rows {sum(len(x) for x in pieces):,}",
            flush=True,
        )

    if not pieces:
        raise RuntimeError(f"No RSJ observations produced for {minute_file}")
    daily = pd.concat(pieces, ignore_index=True).sort_values(["date", "asset_id"])
    if daily.duplicated(["date", "asset_id"]).any():
        raise RuntimeError(f"Duplicate daily RSJ keys produced for {minute_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_file, index=False, compression="zstd")
    return pd.DataFrame(
        [
            {
                "quarter": minute_file.stem,
                "crsp_symbols": len(symbols),
                "ambiguous_symbol_dates_dropped": n_ambiguous,
                "source_rows_for_crsp_symbols": source_rows,
                "regular_session_rows_matched": matched_rows,
                "reference_bad_price_rows_dropped": reference_bad_price_rows,
                "local_bad_price_rows_dropped": local_bad_price_rows,
                "daily_rsj_rows": len(daily),
                "assets": daily["asset_id"].nunique(),
                "first_date": daily["date"].min(),
                "last_date": daily["date"].max(),
            }
        ]
    )


def inventory_outputs(output: Path) -> pd.DataFrame:
    """Validate completed quarter files and return a compact audit table."""
    records = []
    for file in sorted(output.glob("????Q?.parquet")):
        x = pd.read_parquet(
            file,
            columns=["date", "asset_id", "rv", "rsj", "n_intraday_returns", "n_observed_minutes"],
        )
        duplicate_keys = int(x.duplicated(["date", "asset_id"]).sum())
        if duplicate_keys:
            raise RuntimeError(f"{file} contains {duplicate_keys:,} duplicate date/asset keys")
        if not x["rsj"].between(-1.0, 1.0).all():
            raise RuntimeError(f"{file} contains RSJ outside [-1, 1]")
        records.append(
            {
                "quarter": file.stem,
                "rows": len(x),
                "assets": x["asset_id"].nunique(),
                "first_date": pd.to_datetime(x["date"]).min(),
                "last_date": pd.to_datetime(x["date"]).max(),
                "duplicate_keys": duplicate_keys,
                "rv_p99": x["rv"].quantile(0.99),
                "rv_max": x["rv"].max(),
                "mean_observed_minutes": x["n_observed_minutes"].mean(),
                "file_bytes": file.stat().st_size,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    default_source = Path(os.environ.get("RSJ_INTRADAY_SOURCE", r"F:\temp"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--crsp-daily", type=Path, default=BASE / "crsp_daily")
    parser.add_argument("--output", type=Path, default=BASE / "rsj_daily")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--symbol-chunk-size", type=int, default=250)
    parser.add_argument("--min-intraday-returns", type=int, default=60)
    parser.add_argument("--min-reference-price-ratio", type=float, default=0.5)
    parser.add_argument("--max-reference-price-ratio", type=float, default=2.0)
    parser.add_argument("--max-local-price-ratio", type=float, default=1.5)
    args = parser.parse_args()

    if not 0 < args.min_reference_price_ratio < 1:
        parser.error("--min-reference-price-ratio must be between 0 and 1")
    if args.max_reference_price_ratio <= 1:
        parser.error("--max-reference-price-ratio must exceed 1")
    if args.max_local_price_ratio <= 1:
        parser.error("--max-local-price-ratio must exceed 1")

    files = []
    for path in sorted(args.source.glob("*.parquet")):
        match = FILE_RE.match(path.name)
        if match and args.start_year <= int(match.group(1)) <= args.end_year:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No YYYYQn.parquet files found in {args.source}")

    diagnostics = []
    for minute_file in files:
        year = int(FILE_RE.match(minute_file.name).group(1))
        crsp_file = args.crsp_daily / f"{year}.parquet"
        if not crsp_file.exists():
            raise FileNotFoundError(f"{crsp_file} not found; run pipeline/build_crsp_daily.py first")
        output_file = args.output / minute_file.name
        if output_file.exists():
            print(f"Skipping existing {output_file}", flush=True)
            continue
        diagnostics.append(
            process_quarter(
                minute_file,
                crsp_file,
                output_file,
                symbol_chunk_size=args.symbol_chunk_size,
                min_intraday_returns=args.min_intraday_returns,
                min_reference_price_ratio=args.min_reference_price_ratio,
                max_reference_price_ratio=args.max_reference_price_ratio,
                max_local_price_ratio=args.max_local_price_ratio,
            )
        )
        pd.concat(diagnostics, ignore_index=True).to_csv(args.output / "diagnostics.csv", index=False)
    inventory = inventory_outputs(args.output)
    inventory.to_csv(args.output / "inventory.csv", index=False)
    print(
        f"RSJ daily build complete: {len(inventory):,} quarters, "
        f"{inventory['rows'].sum():,} rows in {args.output}"
    )


if __name__ == "__main__":
    main()
