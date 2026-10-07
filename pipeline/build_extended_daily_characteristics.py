"""Build reusable monthly characteristics from partitioned daily CRSP data.

Outputs data/base/extended_daily_characteristics.parquet with a simple trailing beta,
the Frazzini-Pedersen beta estimator, and monthly MAX.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))
from factorlab import data, panels  # noqa: E402


def _rolling_std_numpy(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    valid = np.isfinite(values)
    x = np.where(valid, values, 0.0)
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - window)
    def window_sum(v: np.ndarray) -> np.ndarray:
        c = np.concatenate(([0.0], np.cumsum(v)))
        return c[end] - c[start]
    n, sx, sx2 = window_sum(valid.astype(float)), window_sum(x), window_sum(x * x)
    variance = (sx2 - sx * sx / np.where(n > 0, n, 1.0)) / np.where(n > 1, n - 1, 1.0)
    return np.where(n >= minimum, np.sqrt(np.maximum(variance, 0)), np.nan)


def _corr_numpy(x: np.ndarray, y: np.ndarray, window: int, minimum: int) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - window)
    sums = []
    for v in (valid.astype(float), xx, yy, xx * yy, xx * xx, yy * yy):
        c = np.concatenate(([0.0], np.cumsum(v)))
        sums.append(c[end] - c[start])
    n, sx, sy, sxy, sx2, sy2 = sums
    safe = np.where(n > 0, n, 1.0)
    cov = sxy - sx * sy / safe
    den = np.sqrt(np.maximum(sx2 - sx * sx / safe, 0) * np.maximum(sy2 - sy * sy / safe, 0))
    return np.where((n >= minimum) & (den > 0), cov / np.where(den > 0, den, 1.0), np.nan)


def _fp_beta_at_month_ends(joined: pd.DataFrame, current_start: pd.Timestamp) -> pd.DataFrame:
    dates = joined["date"].to_numpy(dtype="datetime64[ns]")
    ret = joined["ret"].to_numpy(dtype=float)
    mkt = joined["mkt_ret"].to_numpy(dtype=float)
    rf = joined["rf_daily"].to_numpy(dtype=float)
    pieces = []
    for asset_id, positions in joined.groupby("asset_id", sort=False).indices.items():
        stock = np.log1p(ret[positions].clip(min=-0.999999)) - np.log1p(rf[positions])
        market = np.log1p(mkt[positions].clip(min=-0.999999)) - np.log1p(rf[positions])
        stock3 = np.full(len(positions), np.nan)
        market3 = np.full(len(positions), np.nan)
        if len(positions) >= 3:
            valid_stock = np.isfinite(stock[:-2]) & np.isfinite(stock[1:-1]) & np.isfinite(stock[2:])
            valid_market = np.isfinite(market[:-2]) & np.isfinite(market[1:-1]) & np.isfinite(market[2:])
            stock3[2:] = np.where(valid_stock, stock[:-2] + stock[1:-1] + stock[2:], np.nan)
            market3[2:] = np.where(valid_market, market[:-2] + market[1:-1] + market[2:], np.nan)
        beta = _corr_numpy(stock3, market3, 1250, 750) * _rolling_std_numpy(stock, 252, 120) / _rolling_std_numpy(market, 252, 120)
        asset_dates = dates[positions]
        months = asset_dates.astype("datetime64[M]")
        endpoint = np.r_[months[1:] != months[:-1], True]
        use = endpoint & (asset_dates >= np.datetime64(current_start))
        if use.any():
            month_end = months[use] + np.timedelta64(1, "M") - np.timedelta64(1, "D")
            pieces.append(pd.DataFrame({"date": month_end.astype("datetime64[ns]"), "asset_id": asset_id, "beta_fp": beta[use]}))
    return pd.concat(pieces, ignore_index=True)


def _month_features(current: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    d = current.assign(date_month=current["date"].dt.to_period("M").dt.to_timestamp("M"))
    out = d.groupby(["date_month", "asset_id"], as_index=False).agg(
        n_daily=("ret", "count"), max_return=("ret", "max")
    ).rename(columns={"date_month": "date"})
    out["max_return"] = out["max_return"].where(out["n_daily"].ge(15))
    return out.merge(beta, on=["date", "asset_id"], how="left")


def build() -> pd.DataFrame:
    files = sorted((data.BASE / "crsp_daily").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily CRSP files under {data.BASE / 'crsp_daily'}")
    previous = pd.DataFrame()
    pieces = []
    rf = panels.risk_free().copy()
    rf["month"] = rf["date"].dt.to_period("M")
    rf_by_month = rf.set_index("month")["RF"]
    for path in files:
        current = pd.read_parquet(path, columns=["date", "asset_id", "ret", "market_equity"])
        current["date"] = pd.to_datetime(current["date"])
        current = current.sort_values(["asset_id", "date"])
        current["lag_me"] = current.groupby("asset_id")["market_equity"].shift(1)
        current["lag_me"] = current["lag_me"].fillna(current["market_equity"])
        valid = current["ret"].notna() & current["lag_me"].gt(0)
        market = (current["ret"] * current["lag_me"]).where(valid).groupby(current["date"]).sum(min_count=1) / current["lag_me"].where(valid).groupby(current["date"]).sum(min_count=1)
        current["mkt_ret"] = current["date"].map(market)
        month = current["date"].dt.to_period("M")
        n_trading = current[["date"]].drop_duplicates().assign(month=lambda x: x["date"].dt.to_period("M")).groupby("month").size()
        current["rf_daily"] = month.map((1.0 + rf_by_month).pow(1.0 / n_trading) - 1.0).astype(float)
        cols = ["date", "asset_id", "ret", "mkt_ret", "rf_daily"]
        joined = pd.concat([previous, current[cols]], ignore_index=True).sort_values(["asset_id", "date"]).reset_index(drop=True)
        beta = _fp_beta_at_month_ends(joined, current["date"].min())
        pieces.append(_month_features(current, beta))
        previous = joined.groupby("asset_id", sort=False).tail(1252)[cols]
        print(f"Processed {path.name}")
    out = pd.concat(pieces, ignore_index=True).sort_values(["asset_id", "date"])

    return out


if __name__ == "__main__":
    result = build()
    path = data.save_base(result, "extended_daily_characteristics")
    print(f"Wrote {len(result):,} stock-months to {path}")
