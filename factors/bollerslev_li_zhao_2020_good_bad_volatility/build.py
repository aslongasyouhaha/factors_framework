"""Build the weekly low-RSJ signal into data/factors/<project>/.

Prerequisites:
    python pipeline/build_crsp_daily.py --start-year 2008 --end-year 2025
    python pipeline/build_rsj_daily.py --source F:\\temp --start-year 2008 --end-year 2025
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))
sys.path.insert(0, str(HERE))

from factorlab import data  # noqa: E402
from rsj import make_weekly_rsj_panel  # noqa: E402


META = {
    "name": "RSJ",
    "title": "Relative signed jump variation",
    "paper": "Bollerslev, Li & Zhao (2020), Good volatility, bad volatility, and the cross section of stock returns, JFQA",
    "description": "Negative five-day mean RSJ; larger exposure means more downside than upside jump variation (long low RSJ, short high RSJ).",
    "exposure": "-mean_5d((RV_plus - RV_minus) / (RV_plus + RV_minus))",
    "frequency": "weekly (Tuesday close)",
    "sample": "2008-2025; point-in-time CRSP US common stocks on NYSE/AMEX/NASDAQ with price $5-$1,000",
    "report": {
        "n_groups": 5,
        "weighting": "value",
        "long_groups": ["D05"],
        "short_groups": ["D01"],
        "winsorize": None,
        "periods_per_year": 52,
        "rebalances_per_year": 52,
    },
}


def load_daily(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    files = []
    for file in sorted(path.glob("????Q?.parquet")):
        year = int(file.stem[:4])
        if start_year <= year <= end_year:
            files.append(file)
    if not files:
        raise FileNotFoundError(f"No quarterly RSJ files found in {path}; run pipeline/build_rsj_daily.py first")
    cols = ["date", "asset_id", "rv", "rsj", "rsk", "rkt"]
    return pd.concat([pd.read_parquet(file, columns=cols) for file in files], ignore_index=True)


def build_exposure(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = make_weekly_rsj_panel(daily)
    # Repository convention: a larger exposure is the long direction. The
    # paper predicts lower returns for high raw RSJ, hence exposure = -RSJ.
    exposure = weekly.rename(columns={"rebalance_date": "date"})[["date", "asset_id"]].copy()
    exposure["exposure"] = -weekly["signal_value"].to_numpy()
    raw = weekly.rename(columns={"rebalance_date": "date", "signal_value": "rsj_week"})
    return exposure, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--daily-rsj", type=Path, default=data.BASE / "rsj_daily")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    daily = load_daily(args.daily_rsj, args.start_year, args.end_year)
    exposure, weekly = build_exposure(daily)
    path = data.save_factor(HERE.name, exposure, META, tables={"weekly_realized_measures": weekly})
    print(
        f"Wrote {len(exposure):,} weekly RSJ exposures for {exposure['asset_id'].nunique():,} assets "
        f"({exposure['date'].min():%Y-%m-%d} to {exposure['date'].max():%Y-%m-%d}) to {path}"
    )


if __name__ == "__main__":
    main()

