from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))
sys.path.insert(0, str(HERE))

from factorlab import FactorBacktester, FactorConfig, QuantileSignalFactorBuilder, data, forward_returns, to_wide  # noqa: E402
from rsj import make_weekly_rsj_panel, realized_measures_from_bars  # noqa: E402

META = {
    "name": "RSJ",
    "title": "Relative signed jump variation",
    "paper": "Bollerslev, Li & Zhao (2020), Good volatility, bad volatility, and the cross section of stock returns, JFQA",
    "description": "Weekly mean of daily RSJ = (RV+ - RV-) / (RV+ + RV-) from intraday returns; the traded factor is low minus high",
    "frequency": "weekly (Tuesday close)",
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=True)
    else:
        frame.to_parquet(path, index=False, compression="zstd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and backtest the JFQA 2020 RSJ factor.")
    parser.add_argument("--bars", type=Path, required=True, help="5-minute intraday bars (CSV or Parquet).")
    parser.add_argument("--returns", type=Path, help="Optional daily return panel for backtesting.")
    parser.add_argument("--market-equity", type=Path, help="Required for value weighting.")
    parser.add_argument("--output", type=Path, default=HERE / "result")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--asset-col", default="asset_id")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--return-col", help="Intraday log-return column; bypasses price differencing.")
    parser.add_argument("--min-intraday-returns", type=int, default=60)
    parser.add_argument("--weighting", choices=("equal", "value"), default="equal")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    bars = read_table(args.bars)
    daily = realized_measures_from_bars(
        bars,
        timestamp_col=args.timestamp_col,
        asset_col=args.asset_col,
        price_col=args.price_col,
        return_col=args.return_col,
        min_intraday_returns=args.min_intraday_returns,
    )
    weekly = make_weekly_rsj_panel(daily)

    if args.weighting == "value":
        if args.market_equity is None:
            raise ValueError("--market-equity is required when --weighting value")
        me = read_table(args.market_equity).copy()
        me["date"] = pd.to_datetime(me["date"], errors="coerce").dt.normalize()
        me = me.rename(columns={args.asset_col: "asset_id", "date": "rebalance_date"})
        me = me[["rebalance_date", "asset_id", "market_equity"]]
        if me.duplicated(["rebalance_date", "asset_id"]).any():
            raise ValueError("Market-equity input contains duplicate (date, asset_id) rows")
        weekly = weekly.merge(me, on=["rebalance_date", "asset_id"], how="left")

    low, high = "D01", f"D{args.groups:02d}"
    cfg = FactorConfig(
        name="RSJ_LOW_MINUS_HIGH",
        frequency="weekly",
        n_groups=args.groups,
        weighting=args.weighting,
        long_groups=(low,),
        short_groups=(high,),
    )
    builder = QuantileSignalFactorBuilder(
        signal_source_col="signal_value",
        config=cfg,
        winsorize=False,
    )
    factor = builder.build({"panel": weekly})
    spread = builder.spread_holdings(factor.holdings)

    # Factor data: the weekly RSJ signal is the exposure (data/factors/<project>/).
    exposure = weekly.rename(columns={"rebalance_date": "date", "signal_value": "exposure"})[["date", "asset_id", "exposure"]]
    data.save_factor(HERE.name, exposure, META, tables={"daily_realized_measures": daily.reset_index()})
    write_frame(daily, out / "rsj_daily_realized_measures.parquet")
    write_frame(weekly, out / "rsj_weekly_signals.parquet")
    write_frame(factor.holdings, out / "rsj_quintile_holdings.parquet")
    write_frame(spread, out / "rsj_low_minus_high_holdings.parquet")
    factor.diagnostics.to_csv(out / "rsj_diagnostics.csv", index=False)

    if args.returns is None:
        print(f"Built {len(weekly):,} weekly signals and {len(spread):,} spread holdings in {out}")
        return

    returns = read_table(args.returns).rename(columns={args.asset_col: "asset_id"})
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
    returns_wide = to_wide(returns, "ret")
    signal = to_wide(weekly.rename(columns={"rebalance_date": "date"}), "signal_value")
    weight_base = None
    if args.weighting == "value":
        weight_base = to_wide(weekly.rename(columns={"rebalance_date": "date"}), "market_equity")
    wide_factor = builder.build_wide(signal, weight_base=weight_base)
    bt = FactorBacktester(periods_per_year=252)
    fwd = forward_returns(returns_wide, wide_factor.book.dates)
    result = bt.run(
        wide_factor.book,
        returns_wide,
        signal=wide_factor.signal,
        forward=fwd,
        cost_bps=args.cost_bps,
        drift=True,
        make_plots=False,
    )
    result.returns.to_csv(out / "rsj_factor_returns.csv")
    result.layer_returns.to_csv(out / "rsj_layer_returns.csv")
    result.turnover.to_csv(out / "rsj_turnover.csv")
    result.summary.to_csv(out / "rsj_summary.csv")
    result.ic.to_csv(out / "rsj_ic.csv")
    result.r2.to_csv(out / "rsj_r2.csv")
    result.by_year.to_csv(out / "rsj_performance_by_year.csv")
    print(result.summary.loc[["RSJ_LOW_MINUS_HIGH"]].to_string())


if __name__ == "__main__":
    main()

