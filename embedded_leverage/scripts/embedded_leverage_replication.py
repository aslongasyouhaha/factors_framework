from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECTS = Path("D:/Projects")
OPTIONS = PROJECTS / "data" / "options_data"
CLOSES = PROJECTS / "data" / "close_price_data"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()
DATA_OUT = BASE_DIR / "data" / "index_options"
RESULTS_OUT = BASE_DIR / "results" / "index_options"

OPTION_COLS = [
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "optionid",
    "ticker",
    "contract_size",
    "dte",
    "strike",
]

PRICE_FILES = {
    "SPX": ("SP500.csv", "SP500"),
    "XEO": ("SP100.csv", "SP100"),
    "OEX": ("SP100.csv", "SP100"),
    "NDX": ("NDX.csv", "NDX"),
    "QQQ": ("QQQ.csv", "QQQ"),
}

INDEX_FILES = {
    "SPX": "spx_index_{year}.parquet",
    "XEO": "xeo_index_{year}.parquet",
    "OEX": "oex_index_{year}.parquet",
    "NDX": "ndx_index_{year}.parquet",
    "QQQ": "qqq_etf_{year}.parquet",
}


@dataclass(frozen=True)
class Config:
    start_year: int = 2017
    end_year: int = 2025
    min_dte: int = 14
    max_dte: int = 760
    min_abs_delta: float = 0.05
    max_abs_delta: float = 0.95
    min_open_interest: int = 1
    max_rel_spread: float = 1.00
    omega_trim: float = 0.01
    min_options_per_underlying_month: int = 4


def third_friday(year: int, month: int) -> pd.Timestamp:
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [
        pd.Timestamp(day)
        for day in c.itermonthdates(year, month)
        if day.month == month and day.weekday() == 4
    ]
    return fridays[2]


def load_prices(tickers: list[str]) -> pd.DataFrame:
    parts = []
    for ticker in tickers:
        file_name, col = PRICE_FILES[ticker]
        df = pd.read_csv(CLOSES / file_name, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", col: "spot"})
        df["ticker"] = ticker
        df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
        parts.append(df[["date", "ticker", "spot"]].dropna())
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])


def build_roll_pairs(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Entry is first available trading day after the previous standard monthly expiry;
    # exit is the last available trading day up to the next standard monthly expiry.
    out = []
    for ticker, px in prices.groupby("ticker"):
        dates = pd.Series(sorted(px["date"].unique()))
        date_set = set(dates)
        for year in range(cfg.start_year, cfg.end_year + 1):
            for month in range(1, 13):
                prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
                entry_anchor = third_friday(prev_year, prev_month)
                exit_anchor = third_friday(year, month)
                entry_candidates = dates[dates > entry_anchor]
                exit_candidates = dates[dates <= exit_anchor]
                if entry_candidates.empty or exit_candidates.empty:
                    continue
                entry = entry_candidates.iloc[0]
                exit_ = exit_candidates.iloc[-1]
                if entry >= exit_ or entry not in date_set or exit_ not in date_set:
                    continue
                out.append({"ticker": ticker, "entry_date": entry, "exit_date": exit_})
    pairs = pd.DataFrame(out).drop_duplicates()
    return pairs[(pairs["entry_date"].dt.year >= cfg.start_year) & (pairs["entry_date"].dt.year <= cfg.end_year)]


def scan_option_dates(tickers: list[str], dates: set[pd.Timestamp], cfg: Config) -> pd.DataFrame:
    parts = []
    years = sorted({d.year for d in dates})
    for ticker in tickers:
        pattern = INDEX_FILES[ticker]
        for year in years:
            path = OPTIONS / pattern.format(year=year)
            if not path.exists():
                continue
            pf = pq.ParquetFile(path)
            cols = [c for c in OPTION_COLS if c in pf.schema.names]
            kept = 0
            for rg in range(pf.num_row_groups):
                df = pf.read_row_group(rg, columns=cols).to_pandas()
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"].isin(dates)]
                if df.empty:
                    continue
                df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
                df = df[df["ticker"].eq(ticker)]
                if df.empty:
                    continue
                df["exdate"] = pd.to_datetime(df["exdate"])
                if "dte" not in df.columns:
                    df["dte"] = (df["exdate"] - df["date"]).dt.days
                if "strike" not in df.columns:
                    df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0
                df["mid"] = (pd.to_numeric(df["best_bid"], errors="coerce") + pd.to_numeric(df["best_offer"], errors="coerce")) / 2.0
                df["spread"] = pd.to_numeric(df["best_offer"], errors="coerce") - pd.to_numeric(df["best_bid"], errors="coerce")
                df["rel_spread"] = df["spread"] / df["mid"].replace(0.0, np.nan)
                df["contract_size"] = pd.to_numeric(df.get("contract_size", 100.0), errors="coerce").fillna(100.0)
                df = df[
                    df["cp_flag"].isin(["C", "P"])
                    & df["dte"].between(cfg.min_dte, cfg.max_dte)
                    & df["mid"].gt(0)
                    & df["best_bid"].gt(0)
                    & df["best_offer"].ge(df["best_bid"])
                    & df["rel_spread"].le(cfg.max_rel_spread)
                    & df["open_interest"].fillna(0).ge(cfg.min_open_interest)
                    & df["delta"].abs().between(cfg.min_abs_delta, cfg.max_abs_delta)
                ]
                if not df.empty:
                    parts.append(df)
                    kept += len(df)
            print(f"{ticker} {year}: kept {kept:,} option rows")
    if not parts:
        return pd.DataFrame(columns=OPTION_COLS + ["mid", "spread", "rel_spread"])
    return pd.concat(parts, ignore_index=True)


def maturity_bucket(dte: pd.Series) -> pd.Categorical:
    return pd.cut(
        dte,
        bins=[13, 45, 75, 105, 210, 390, np.inf],
        labels=["1m", "2m", "3m", "6m", "12m", ">12m"],
    )


def delta_bucket(abs_delta: pd.Series) -> pd.Categorical:
    return pd.cut(
        abs_delta,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["DOTM", "OTM", "ATM", "ITM", "DITM"],
        include_lowest=True,
    )


def compute_period_returns(options: pd.DataFrame, prices: pd.DataFrame, rolls: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    entry = options.merge(
        rolls,
        left_on=["ticker", "date"],
        right_on=["ticker", "entry_date"],
        how="inner",
    )
    exitq = options[["ticker", "date", "optionid", "mid"]].rename(columns={"date": "exit_date", "mid": "mid_exit"})
    entry = entry.merge(exitq, on=["ticker", "exit_date", "optionid"], how="inner")
    entry = entry.merge(
        prices.rename(columns={"date": "entry_date", "spot": "spot_entry"}),
        on=["ticker", "entry_date"],
        how="left",
    )
    entry = entry.merge(
        prices.rename(columns={"date": "exit_date", "spot": "spot_exit"}),
        on=["ticker", "exit_date"],
        how="left",
    )
    entry = entry.dropna(subset=["spot_entry", "spot_exit", "delta", "mid", "mid_exit"])
    entry = entry[entry["exdate"].ge(entry["exit_date"])]
    entry["omega"] = (entry["delta"].abs() * entry["spot_entry"] / entry["mid"]).replace([np.inf, -np.inf], np.nan)
    entry = entry.dropna(subset=["omega"])
    if cfg.omega_trim > 0 and not entry.empty:
        lo = entry["omega"].quantile(cfg.omega_trim)
        hi = entry["omega"].quantile(1 - cfg.omega_trim)
        entry = entry[entry["omega"].between(lo, hi)]

    # Fast first-pass delta hedge: use entry delta for the holding-period spot move.
    # Use --daily-hedge for the closer-to-paper daily re-hedged PnL.
    entry["dh_ret"] = (entry["mid_exit"] - entry["mid"] - entry["delta"] * (entry["spot_exit"] - entry["spot_entry"])) / entry["mid"]
    entry["option_mkt_cap"] = entry["mid"] * entry["open_interest"].fillna(0) * entry["contract_size"]
    entry["abs_delta"] = entry["delta"].abs()
    entry["delta_bucket"] = delta_bucket(entry["abs_delta"])
    entry["maturity_bucket"] = maturity_bucket(entry["dte"])
    return entry.replace([np.inf, -np.inf], np.nan).dropna(subset=["dh_ret", "omega"])


def scan_daily_quotes_for_selection(tickers: list[str], selection: pd.DataFrame) -> pd.DataFrame:
    parts = []
    if selection.empty:
        return pd.DataFrame()
    selection = selection.copy()
    selection["entry_date"] = pd.to_datetime(selection["entry_date"])
    selection["exit_date"] = pd.to_datetime(selection["exit_date"])
    selection["optionid"] = pd.to_numeric(selection["optionid"], errors="coerce").astype("Int64")
    years = range(selection["entry_date"].dt.year.min(), selection["exit_date"].dt.year.max() + 1)
    for ticker in tickers:
        ticker_sel = selection[selection["ticker"].eq(ticker)]
        if ticker_sel.empty:
            continue
        optionids = set(ticker_sel["optionid"].dropna().astype("int64"))
        pattern = INDEX_FILES[ticker]
        for year in years:
            path = OPTIONS / pattern.format(year=year)
            if not path.exists():
                continue
            pf = pq.ParquetFile(path)
            cols = [c for c in ["date", "ticker", "optionid", "best_bid", "best_offer", "delta"] if c in pf.schema.names]
            kept = 0
            for rg in range(pf.num_row_groups):
                df = pf.read_row_group(rg, columns=cols).to_pandas()
                df["optionid"] = pd.to_numeric(df["optionid"], errors="coerce").astype("Int64")
                df = df[df["optionid"].isin(optionids)]
                if df.empty:
                    continue
                df["date"] = pd.to_datetime(df["date"])
                df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
                df = df[df["ticker"].eq(ticker)]
                df["mid"] = (pd.to_numeric(df["best_bid"], errors="coerce") + pd.to_numeric(df["best_offer"], errors="coerce")) / 2.0
                df = df[df["mid"].gt(0) & df["delta"].notna()]
                if not df.empty:
                    parts.append(df[["date", "ticker", "optionid", "mid", "delta"]])
                    kept += len(df)
            print(f"{ticker} {year}: daily hedge rows kept {kept:,}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["date", "ticker", "optionid"], keep="last")


def apply_daily_delta_hedge(ret: pd.DataFrame, prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    selection = ret[
        ["entry_date", "exit_date", "ticker", "optionid", "mid"]
    ].reset_index(drop=True).rename(columns={"mid": "mid_entry"})
    selection["trade_id"] = np.arange(len(selection), dtype=np.int64)
    quotes = scan_daily_quotes_for_selection(tickers, selection)
    if quotes.empty:
        print("No daily hedge quotes found; keeping fast holding-period delta hedge.")
        return ret

    daily = quotes.merge(selection, on=["ticker", "optionid"], how="inner")
    daily = daily[(daily["date"] >= daily["entry_date"]) & (daily["date"] <= daily["exit_date"])]
    daily = daily.merge(prices, on=["ticker", "date"], how="left").dropna(subset=["spot"])
    daily = daily.sort_values(["trade_id", "date"])
    daily["d_mid"] = daily.groupby("trade_id")["mid"].diff()
    daily["d_spot"] = daily.groupby("trade_id")["spot"].diff()
    daily["lag_delta"] = daily.groupby("trade_id")["delta"].shift(1)
    daily["pnl"] = daily["d_mid"] - daily["lag_delta"] * daily["d_spot"]
    daily["pnl"] = daily["pnl"].fillna(0.0)

    coverage = daily.groupby("trade_id").agg(
        first_date=("date", "first"),
        last_date=("date", "last"),
        n_daily_obs=("date", "size"),
        pnl=("pnl", "sum"),
        mid_entry=("mid_entry", "first"),
    )
    coverage = coverage[
        coverage["first_date"].eq(selection.set_index("trade_id")["entry_date"])
        & coverage["last_date"].eq(selection.set_index("trade_id")["exit_date"])
        & coverage["mid_entry"].gt(0)
    ].copy()
    coverage["dh_ret_daily"] = coverage["pnl"] / coverage["mid_entry"]

    out = ret.reset_index(drop=True).copy()
    out["trade_id"] = np.arange(len(out), dtype=np.int64)
    out = out.merge(coverage[["dh_ret_daily", "n_daily_obs"]], left_on="trade_id", right_index=True, how="inner")
    out["dh_ret_fast"] = out["dh_ret"]
    out["dh_ret"] = out["dh_ret_daily"]
    return out.drop(columns=["trade_id"])


def weighted_avg(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna() & w.gt(0)
    if not mask.any():
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def make_bucket_table(ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["delta_bucket", "maturity_bucket"]
    for key, g in ret.groupby(keys, observed=True):
        rows.append(
            {
                "delta_bucket": key[0],
                "maturity_bucket": key[1],
                "n": len(g),
                "omega_vw": weighted_avg(g["omega"], g["option_mkt_cap"]),
                "dh_ret_vw_pct": 100 * weighted_avg(g["dh_ret"], g["option_mkt_cap"]),
                "dh_ret_ew_pct": 100 * g["dh_ret"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def make_bab(ret: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_rows = []
    leg_rows = []
    for (date, ticker), g in ret.groupby(["entry_date", "ticker"]):
        if len(g) < cfg.min_options_per_underlying_month:
            continue
        med = g["omega"].median()
        low = g[g["omega"] <= med]
        high = g[g["omega"] > med]
        if low.empty or high.empty:
            continue
        ret_l = weighted_avg(low["dh_ret"], low["option_mkt_cap"])
        ret_h = weighted_avg(high["dh_ret"], high["option_mkt_cap"])
        om_l = weighted_avg(low["omega"], low["option_mkt_cap"])
        om_h = weighted_avg(high["omega"], high["option_mkt_cap"])
        mcap_total = g["option_mkt_cap"].sum()
        if any(pd.isna(v) or v <= 0 for v in [om_l, om_h, mcap_total]):
            continue
        bab = ret_l / om_l - ret_h / om_h
        period_rows.append(
            {
                "entry_date": date,
                "ticker": ticker,
                "bab_ret": bab,
                "low_ret": ret_l,
                "high_ret": ret_h,
                "low_omega": om_l,
                "high_omega": om_h,
                "n_options": len(g),
                "option_mkt_cap": mcap_total,
            }
        )
        leg_rows.append({"entry_date": date, "ticker": ticker, "side": "low", "ret": ret_l, "omega": om_l, "n": len(low)})
        leg_rows.append({"entry_date": date, "ticker": ticker, "side": "high", "ret": ret_h, "omega": om_h, "n": len(high)})
    by_underlying = pd.DataFrame(period_rows)
    if by_underlying.empty:
        return by_underlying, pd.DataFrame()
    agg_rows = []
    for date, g in by_underlying.groupby("entry_date"):
        agg_rows.append(
            {
                "entry_date": date,
                "bab_ret": weighted_avg(g["bab_ret"], g["option_mkt_cap"]),
                "n_underlyings": g["ticker"].nunique(),
                "n_options": g["n_options"].sum(),
            }
        )
    return by_underlying, pd.DataFrame(agg_rows).sort_values("entry_date")


def summarize_returns(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {}
    vol = s.std(ddof=1)
    sharpe = np.nan if vol == 0 else np.sqrt(12) * s.mean() / vol
    tstat = np.nan if vol == 0 else s.mean() / vol * np.sqrt(len(s))
    return {
        "n_months": len(s),
        "mean_monthly_pct": 100 * s.mean(),
        "vol_monthly_pct": 100 * vol,
        "ann_sharpe": sharpe,
        "tstat_mean": tstat,
        "min_monthly_pct": 100 * s.min(),
        "max_monthly_pct": 100 * s.max(),
        "positive_months_pct": 100 * (s > 0).mean(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast replication of Frazzini-Pedersen Embedded Leverage on local option data.")
    p.add_argument("--tickers", default="SPX,NDX,XEO,OEX", help="Comma-separated option roots/tickers.")
    p.add_argument("--start-year", type=int, default=2017)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--max-rel-spread", type=float, default=1.0)
    p.add_argument("--min-open-interest", type=int, default=1)
    p.add_argument("--omega-trim", type=float, default=0.01)
    p.add_argument("--min-options-per-underlying-month", type=int, default=4)
    p.add_argument("--daily-hedge", action="store_true", help="Use daily option quotes and lagged delta for hedged PnL.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        start_year=args.start_year,
        end_year=args.end_year,
        max_rel_spread=args.max_rel_spread,
        min_open_interest=args.min_open_interest,
        omega_trim=args.omega_trim,
        min_options_per_underlying_month=args.min_options_per_underlying_month,
    )
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    prices = load_prices(tickers)
    rolls = build_roll_pairs(prices, cfg)
    dates = set(rolls["entry_date"]).union(set(rolls["exit_date"]))
    print(f"Roll pairs: {len(rolls):,}; quote dates needed: {len(dates):,}")

    opts = scan_option_dates(tickers, dates, cfg)
    print(f"Loaded filtered quote rows: {len(opts):,}")
    ret = compute_period_returns(opts, prices, rolls, cfg)
    if args.daily_hedge:
        print("Applying daily delta hedge. This is slower but closer to the paper.")
        ret = apply_daily_delta_hedge(ret, prices, tickers)
    print(f"Holding-period option returns: {len(ret):,}")

    bucket = make_bucket_table(ret)
    by_underlying, bab = make_bab(ret, cfg)
    summary = (
        pd.DataFrame([summarize_returns(bab["bab_ret"])])
        if "bab_ret" in bab.columns
        else pd.DataFrame([{"n_months": 0}])
    )

    ret.to_parquet(DATA_OUT / "index_option_period_returns.parquet", index=False, compression="zstd")
    bucket.to_csv(RESULTS_OUT / "index_option_bucket_table.csv", index=False)
    by_underlying.to_csv(RESULTS_OUT / "index_option_bab_by_underlying.csv", index=False)
    bab.to_csv(RESULTS_OUT / "index_option_bab_monthly.csv", index=False)
    summary.to_csv(RESULTS_OUT / "index_option_bab_summary.csv", index=False)

    print("\nBAB summary")
    print(summary.to_string(index=False))
    print("\nBucket table preview")
    print(bucket.head(20).to_string(index=False))
    print(f"\nWrote period-return data under {DATA_OUT.resolve()}")
    print(f"Wrote results under {RESULTS_OUT.resolve()}")


if __name__ == "__main__":
    main()
