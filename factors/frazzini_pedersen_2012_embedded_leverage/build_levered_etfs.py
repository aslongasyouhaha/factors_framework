from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

from factorlab.data import FACTORS, SOURCE  # noqa: E402

DATA_OUT = SOURCE / "yahoo"
YIELD_DATA = SOURCE / "fred"
RESULTS_OUT = HERE / "result" / "yahoo_etfs"
START = "2017-01-01"
END = "2026-01-01"  # exclusive; covers trading days through 2025.

PAIRS = {
    "Dow Jones": ("DIA", "DDM"),
    "S&P MidCap 400": ("MDY", "MVV"),
    "Nasdaq 100": ("QQQ", "QLD"),
    "Russell 2000": ("IWM", "UWM"),
    "Russell 3000": ("IWV", "UWC"),
    "S&P 500": ("SPY", "SSO"),
    "S&P SmallCap 600": ("IJR", "SAA"),
}

EXPENSE_RATIOS_BPS = {
    "DIA": 17,
    "DDM": 95,
    "MDY": 25,
    "MVV": 95,
    "QQQ": 20,  # The paper lists QQQQ; QQQ is the current ticker.
    "QLD": 95,
    "IWM": 20,
    "UWM": 95,
    "IWV": 20,
    "UWC": 95,
    "SPY": 9,
    "SSO": 92,
    "IJR": 20,
    "SAA": 95,
}

TRADING_DAYS = 252


def load_daily_rf(dates: pd.Series) -> pd.DataFrame:
    path = YIELD_DATA / "DGS1MO.csv"
    if not path.exists():
        out = pd.DataFrame({"date": pd.to_datetime(sorted(pd.Series(dates).dropna().unique()))})
        out["risk_free_daily"] = 0.0
        return out
    rf = pd.read_csv(path, parse_dates=["observation_date"])
    rf = rf.rename(columns={"observation_date": "date", "DGS1MO": "risk_free_annual_pct"})
    rf["risk_free_annual_pct"] = pd.to_numeric(rf["risk_free_annual_pct"], errors="coerce")
    rf = rf.sort_values("date").ffill()
    out = pd.DataFrame({"date": pd.to_datetime(sorted(pd.Series(dates).dropna().unique()))})
    out = pd.merge_asof(out.sort_values("date"), rf[["date", "risk_free_annual_pct"]], on="date")
    out["risk_free_daily"] = out["risk_free_annual_pct"].fillna(0.0) / 100.0 / TRADING_DAYS
    return out[["date", "risk_free_daily"]]


def normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=1, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    else:
        df = raw.reset_index().rename(columns={"Date": "date"})
        df["ticker"] = next(iter(PAIRS.values()))[0]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.sort_values(["ticker", "date"])


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = DATA_OUT / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))
    tickers = sorted({t for pair in PAIRS.values() for t in pair})
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    prices = normalize_download(raw)
    prices.to_parquet(DATA_OUT / "etf_raw_daily_2017_2025.parquet", index=False, compression="zstd")

    returns = prices[["date", "ticker", "adj_close", "close"]].copy()
    returns["price"] = returns["adj_close"].where(returns["adj_close"].notna(), returns["close"])
    returns["ret_after_fee"] = returns.groupby("ticker")["price"].pct_change(fill_method=None)
    returns["expense_ratio_bps"] = returns["ticker"].map(EXPENSE_RATIOS_BPS)
    returns["ret_before_fee"] = returns["ret_after_fee"] + returns["expense_ratio_bps"] / 10000.0 / TRADING_DAYS
    returns = returns.merge(load_daily_rf(returns["date"]), on="date", how="left")
    returns["risk_free_daily"] = returns["risk_free_daily"].fillna(0.0)
    returns["ret_excess_after_fee"] = returns["ret_after_fee"] - returns["risk_free_daily"]
    returns["ret_excess_before_fee"] = returns["ret_before_fee"] - returns["risk_free_daily"]

    rows = []
    for name, (one_x, two_x) in PAIRS.items():
        wide_after = returns[returns["ticker"].isin([one_x, two_x])].pivot(
            index="date", columns="ticker", values="ret_after_fee"
        )
        wide_before = returns[returns["ticker"].isin([one_x, two_x])].pivot(
            index="date", columns="ticker", values="ret_before_fee"
        )
        wide_excess_after = returns[returns["ticker"].isin([one_x, two_x])].pivot(
            index="date", columns="ticker", values="ret_excess_after_fee"
        )
        wide_excess_before = returns[returns["ticker"].isin([one_x, two_x])].pivot(
            index="date", columns="ticker", values="ret_excess_before_fee"
        )
        wide_rf = returns[returns["ticker"].isin([one_x])].set_index("date")["risk_free_daily"]
        wide = wide_after
        if one_x not in wide or two_x not in wide:
            continue
        bab_after = wide_excess_after[one_x] - 0.5 * wide_excess_after[two_x]
        bab_before = wide_excess_before[one_x] - 0.5 * wide_excess_before[two_x]
        pair_df = pd.DataFrame(
            {
                "date": bab_after.index,
                "index_name": name,
                "one_x": one_x,
                "two_x": two_x,
                "expense_ratio_1x_bps": EXPENSE_RATIOS_BPS.get(one_x),
                "expense_ratio_2x_bps": EXPENSE_RATIOS_BPS.get(two_x),
                "ret_1x_after_fee": wide_after[one_x].values,
                "ret_2x_after_fee": wide_after[two_x].values,
                "ret_1x_before_fee": wide_before[one_x].values,
                "ret_2x_before_fee": wide_before[two_x].values,
                "risk_free_daily": wide_rf.reindex(bab_after.index).values,
                "ret_1x_excess_after_fee": wide_excess_after[one_x].values,
                "ret_2x_excess_after_fee": wide_excess_after[two_x].values,
                "ret_1x_excess_before_fee": wide_excess_before[one_x].values,
                "ret_2x_excess_before_fee": wide_excess_before[two_x].values,
                "bab_ret": bab_after.values,
                "bab_ret_after_fee": bab_after.values,
                "bab_ret_before_fee": bab_before.values,
            }
        ).dropna(subset=["bab_ret"])
        rows.append(pair_df)
    bab_pairs = pd.concat(rows, ignore_index=True)

    bab_daily = (
        bab_pairs.groupby("date", as_index=False)
        .agg(
            bab_ret=("bab_ret_after_fee", "mean"),
            bab_ret_after_fee=("bab_ret_after_fee", "mean"),
            bab_ret_before_fee=("bab_ret_before_fee", "mean"),
            risk_free_daily=("risk_free_daily", "mean"),
            n_pairs=("index_name", "nunique"),
        )
        .sort_values("date")
    )
    bab_pairs["level"] = "pair"
    bab_pairs["n_pairs"] = pd.NA
    bab_daily["level"] = "portfolio"
    bab_daily["index_name"] = "ALL"
    bab_daily["one_x"] = pd.NA
    bab_daily["two_x"] = pd.NA
    bab_daily["expense_ratio_1x_bps"] = pd.NA
    bab_daily["expense_ratio_2x_bps"] = pd.NA
    bab_daily["ret_1x_after_fee"] = pd.NA
    bab_daily["ret_2x_after_fee"] = pd.NA
    bab_daily["ret_1x_before_fee"] = pd.NA
    bab_daily["ret_2x_before_fee"] = pd.NA
    bab_daily["ret_1x_excess_after_fee"] = pd.NA
    bab_daily["ret_2x_excess_after_fee"] = pd.NA
    bab_daily["ret_1x_excess_before_fee"] = pd.NA
    bab_daily["ret_2x_excess_before_fee"] = pd.NA
    result_cols = [
        "date",
        "level",
        "index_name",
        "one_x",
        "two_x",
        "expense_ratio_1x_bps",
        "expense_ratio_2x_bps",
        "ret_1x_after_fee",
        "ret_2x_after_fee",
        "ret_1x_before_fee",
        "ret_2x_before_fee",
        "risk_free_daily",
        "ret_1x_excess_after_fee",
        "ret_2x_excess_after_fee",
        "ret_1x_excess_before_fee",
        "ret_2x_excess_before_fee",
        "bab_ret",
        "bab_ret_after_fee",
        "bab_ret_before_fee",
        "n_pairs",
    ]
    results = pd.concat(
        [
            bab_pairs[result_cols],
            bab_daily[result_cols],
        ],
        ignore_index=True,
    ).sort_values(["date", "level", "index_name"])
    results.to_csv(RESULTS_OUT / "etf_bab_results_2017_2025.csv", index=False)

    ann = TRADING_DAYS
    summary_rows = []
    for label, col in [("after_fee", "bab_ret_after_fee"), ("before_fee", "bab_ret_before_fee")]:
        s = bab_daily[col].dropna()
        summary_rows.append(
            {
                "return_type": label,
                "start": bab_daily["date"].min().date(),
                "end": bab_daily["date"].max().date(),
                "n_days": len(s),
                "mean_daily_pct": 100 * s.mean(),
                "vol_daily_pct": 100 * s.std(ddof=1),
                "ann_return_pct": 100 * ann * s.mean(),
                "ann_vol_pct": 100 * np.sqrt(ann) * s.std(ddof=1),
                "ann_sharpe": np.sqrt(ann) * s.mean() / s.std(ddof=1),
                "min_daily_pct": 100 * s.min(),
                "max_daily_pct": 100 * s.max(),
                "positive_days_pct": 100 * (s > 0).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    pair_summary = []
    for name, g in bab_pairs.groupby("index_name"):
        x = g["bab_ret_after_fee"].dropna()
        xb = g["bab_ret_before_fee"].dropna()
        pair_summary.append(
            {
                "index_name": name,
                "one_x": g["one_x"].iloc[0],
                "two_x": g["two_x"].iloc[0],
                "n_days": len(x),
                "mean_daily_pct": 100 * x.mean(),
                "ann_sharpe": np.sqrt(ann) * x.mean() / x.std(ddof=1),
                "mean_daily_before_fee_pct": 100 * xb.mean(),
                "ann_sharpe_before_fee": np.sqrt(ann) * xb.mean() / xb.std(ddof=1),
                "positive_days_pct": 100 * (x > 0).mean(),
            }
        )
    pair_summary = pd.DataFrame(pair_summary).sort_values("index_name")

    print("Downloaded tickers:", ", ".join(tickers))
    print("\nBAB summary")
    print(summary.to_string(index=False))
    print("\nPair summary")
    print(pair_summary.to_string(index=False))
    print(f"\nWrote raw data under {DATA_OUT.resolve()}")
    print(f"Wrote results under {RESULTS_OUT.resolve()}")


if __name__ == "__main__":
    main()
