from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab import Factor, FactorConfig, QuantileSignalFactorBuilder


RSJ_COLUMNS = (
    "rv_plus",
    "rv_minus",
    "rv",
    "sj",
    "rsj",
    "rsk",
    "rkt",
    "n_intraday_returns",
)


def realized_measures_from_minute_ohlcv(
    bars: pd.DataFrame,
    *,
    asset_col: str = "asset_id",
    date_col: str = "date",
    session_minute_col: str = "session_minute",
    open_col: str = "open",
    close_col: str = "close",
    min_intraday_returns: int = 60,
) -> pd.DataFrame:
    """Aggregate unadjusted one-minute OHLC bars to paper-style daily measures.

    Rows must already be point-in-time mapped to a permanent asset id and carry
    a session minute in [0, 390], where 0 is 09:30 and 390 is 16:00 New York.
    Minutes 0..389 form 78 five-minute intervals. Missing intervals use the
    previous observed price. When present, the *open* of the 16:00 bar replaces
    the last endpoint so the closing auction is included without using the
    post-market trades contained in that minute bar.
    """
    required = {asset_col, date_col, session_minute_col, open_col, close_col}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Minute OHLC bars missing columns: {sorted(missing)}")
    if not 1 <= min_intraday_returns <= 78:
        raise ValueError("min_intraday_returns must be between 1 and 78")

    x = bars[[asset_col, date_col, session_minute_col, open_col, close_col]].copy()
    x[session_minute_col] = pd.to_numeric(x[session_minute_col], errors="coerce")
    x[open_col] = pd.to_numeric(x[open_col], errors="coerce").where(lambda s: s.gt(0))
    x[close_col] = pd.to_numeric(x[close_col], errors="coerce").where(lambda s: s.gt(0))
    x = x.dropna(subset=[asset_col, date_col, session_minute_col])
    x = x[x[session_minute_col].between(0, 390)].copy()
    x = x.sort_values([asset_col, date_col, session_minute_col], kind="stable")
    x = x.drop_duplicates([asset_col, date_col, session_minute_col], keep="last")
    keys = [asset_col, date_col]

    core = x[x[session_minute_col].lt(390)].copy()
    if core.empty:
        return pd.DataFrame(columns=["date", "asset_id", *RSJ_COLUMNS, "n_observed_minutes"])
    core["bucket"] = (core[session_minute_col].astype(int) // 5).clip(0, 77)
    day_open = core.groupby(keys, sort=True, observed=True)[open_col].first()
    observed = core.groupby(keys, sort=True, observed=True)[close_col].count()
    endpoints = (
        core.groupby([asset_col, date_col, "bucket"], sort=True, observed=True)[close_col]
        .last()
        .unstack("bucket")
        .reindex(columns=range(78))
    )

    auction = x[x[session_minute_col].eq(390)].groupby(keys, sort=True, observed=True)[open_col].first()
    if not auction.empty:
        auction = auction.reindex(endpoints.index)
        endpoints[77] = auction.combine_first(endpoints[77])
    endpoints = endpoints.ffill(axis=1)

    price = endpoints.to_numpy(dtype=float)
    previous = np.roll(price, 1, axis=1)
    previous[:, 0] = np.nan
    finite = np.isfinite(price)
    has_price = finite.any(axis=1)
    first = np.argmax(finite, axis=1)
    open_values = day_open.reindex(endpoints.index).to_numpy(dtype=float)
    rows = np.flatnonzero(has_price & np.isfinite(open_values) & (open_values > 0))
    previous[rows, first[rows]] = open_values[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(price / previous)
    returns[~np.isfinite(returns)] = np.nan

    n = np.isfinite(returns).sum(axis=1)
    r2 = returns**2
    rv_plus = np.nansum(np.where(returns > 0, r2, 0.0), axis=1)
    rv_minus = np.nansum(np.where(returns < 0, r2, 0.0), axis=1)
    rv = rv_plus + rv_minus
    sj = rv_plus - rv_minus
    valid_rv = np.where(rv > 0, rv, np.nan)
    out = pd.DataFrame(
        {
            "rv_plus": rv_plus,
            "rv_minus": rv_minus,
            "rv": rv,
            "sj": sj,
            "rsj": np.clip(sj / valid_rv, -1.0, 1.0),
            "rsk": np.sqrt(n) * np.nansum(returns**3, axis=1) / valid_rv**1.5,
            "rkt": n * np.nansum(returns**4, axis=1) / valid_rv**2,
            "n_intraday_returns": n,
            "n_observed_minutes": observed.reindex(endpoints.index).fillna(0).to_numpy(dtype=int),
        },
        index=endpoints.index,
    ).reset_index()
    out = out[out["n_intraday_returns"].ge(min_intraday_returns) & out["rv"].gt(0)].copy()
    out = out.rename(columns={asset_col: "asset_id", date_col: "date"})
    if pd.api.types.is_integer_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out[["date", "asset_id", *RSJ_COLUMNS, "n_observed_minutes"]].sort_values(
        ["asset_id", "date"]
    ).reset_index(drop=True)


def realized_measures_from_bars(
    bars: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset_id",
    price_col: str = "price",
    return_col: str | None = None,
    min_intraday_returns: int = 60,
) -> pd.DataFrame:
    """Compute daily realized variation measures from regular intraday bars.

    `return_col`, when supplied, must contain intraday log returns. Otherwise
    log returns are computed from `price_col` within each asset-day. Overnight
    returns are deliberately excluded. Duplicate asset/timestamp observations
    keep the last row, matching a last-price bar convention.
    """
    required = {timestamp_col, asset_col, return_col or price_col}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Intraday bars missing columns: {sorted(missing)}")
    if min_intraday_returns < 1:
        raise ValueError("min_intraday_returns must be positive")

    cols = [timestamp_col, asset_col, return_col or price_col]
    x = bars[cols].copy()
    x[timestamp_col] = pd.to_datetime(x[timestamp_col], errors="coerce")
    x = x.dropna(subset=[timestamp_col, asset_col])
    x = x.sort_values([asset_col, timestamp_col], kind="stable")
    x = x.drop_duplicates([asset_col, timestamp_col], keep="last")
    x["date"] = x[timestamp_col].dt.normalize()

    keys = [x[asset_col], x["date"]]
    if return_col is None:
        price = pd.to_numeric(x[price_col], errors="coerce").where(lambda s: s.gt(0))
        x["intraday_ret"] = np.log(price).groupby(keys, sort=False).diff()
    else:
        x["intraday_ret"] = pd.to_numeric(x[return_col], errors="coerce")

    r = x["intraday_ret"].where(np.isfinite(x["intraday_ret"]))
    r2 = r.pow(2)
    x["_rv_plus"] = r2.where(r.gt(0), 0.0).where(r.notna())
    x["_rv_minus"] = r2.where(r.lt(0), 0.0).where(r.notna())
    x["_r3"] = r.pow(3)
    x["_r4"] = r.pow(4)

    daily = (
        x.groupby([asset_col, "date"], sort=True, observed=True)
        .agg(
            rv_plus=("_rv_plus", "sum"),
            rv_minus=("_rv_minus", "sum"),
            sum_r3=("_r3", "sum"),
            sum_r4=("_r4", "sum"),
            n_intraday_returns=("intraday_ret", "count"),
        )
        .reset_index()
        .rename(columns={asset_col: "asset_id"})
    )
    daily = daily[daily["n_intraday_returns"].ge(min_intraday_returns)].copy()
    daily["rv"] = daily["rv_plus"] + daily["rv_minus"]
    daily["sj"] = daily["rv_plus"] - daily["rv_minus"]
    positive_rv = daily["rv"].where(daily["rv"].gt(0))
    daily["rsj"] = (daily["sj"] / positive_rv).clip(-1.0, 1.0)
    n = daily["n_intraday_returns"].astype(float)
    daily["rsk"] = np.sqrt(n) * daily["sum_r3"] / positive_rv.pow(1.5)
    daily["rkt"] = n * daily["sum_r4"] / positive_rv.pow(2)
    return daily[["date", "asset_id", *RSJ_COLUMNS]].sort_values(["asset_id", "date"]).reset_index(drop=True)


def make_weekly_rsj_panel(
    daily: pd.DataFrame,
    *,
    lookback_days: int = 5,
    min_days: int | None = None,
    formation_weekday: int = 1,
) -> pd.DataFrame:
    """Create paper-style weekly signals (Tuesday=1, Monday=0).

    RSJ/RSK/RKT are arithmetic means of daily measures, as in equation (10) of
    the paper. Weekly realized volatility follows equation (9). Rolling windows
    use only rows dated on or before the formation date.
    """
    required = {"date", "asset_id", "rv", "rsj", "rsk", "rkt"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily realized measures missing columns: {sorted(missing)}")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    min_days = lookback_days if min_days is None else min_days
    if not 1 <= min_days <= lookback_days:
        raise ValueError("min_days must be between 1 and lookback_days")
    if formation_weekday not in range(7):
        raise ValueError("formation_weekday must be in [0, 6]")

    x = daily.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["date", "asset_id"]).sort_values(["asset_id", "date"], kind="stable")
    if x.duplicated(["date", "asset_id"]).any():
        raise ValueError("Daily realized measures contain duplicate (date, asset_id) rows")

    # Rolling over observations alone could silently bridge a missing trading
    # day. A global trading-date index lets us require five consecutive market
    # dates for the five-day paper window.
    trade_dates = pd.Index(x["date"].drop_duplicates().sort_values())
    x["_trade_index"] = trade_dates.get_indexer(x["date"])
    grouped = x.groupby("asset_id", sort=False, group_keys=False)
    for col in ("rsj", "rsk", "rkt"):
        x[f"{col}_week"] = grouped[col].transform(
            lambda s: s.rolling(lookback_days, min_periods=min_days).mean()
        )
    rv_sum = grouped["rv"].transform(lambda s: s.rolling(lookback_days, min_periods=min_days).sum())
    x["rvol_week"] = np.sqrt((252.0 / lookback_days) * rv_sum)
    x["n_days"] = grouped["rv"].transform(
        lambda s: s.rolling(lookback_days, min_periods=1).count()
    ).astype(int)
    first_index = grouped["_trade_index"].shift(lookback_days - 1)
    consecutive = x["_trade_index"].sub(first_index).eq(lookback_days - 1)

    out = x[
        x["date"].dt.weekday.eq(formation_weekday)
        & x["n_days"].ge(min_days)
        & (consecutive | (min_days < lookback_days))
    ].copy()
    out = out.rename(columns={"date": "rebalance_date", "rsj_week": "signal_value"})
    out["hold_start"] = out["rebalance_date"] + pd.Timedelta(days=1)
    out["hold_end"] = out["rebalance_date"] + pd.Timedelta(days=7)
    cols = [
        "rebalance_date",
        "hold_start",
        "hold_end",
        "asset_id",
        "signal_value",
        "rsk_week",
        "rkt_week",
        "rvol_week",
        "n_days",
    ]
    return out[cols].sort_values(["rebalance_date", "asset_id"]).reset_index(drop=True)


def build_rsj_factor(
    weekly_panel: pd.DataFrame,
    *,
    weighting: str = "equal",
    n_groups: int = 5,
    winsorize: bool = False,
) -> tuple[Factor, pd.DataFrame]:
    """Build quintile layers and low-RSJ-minus-high-RSJ spread holdings."""
    required = {"rebalance_date", "asset_id", "signal_value"}
    missing = required.difference(weekly_panel.columns)
    if missing:
        raise ValueError(f"Weekly RSJ panel missing columns: {sorted(missing)}")
    if weighting == "value" and "market_equity" not in weekly_panel.columns:
        raise ValueError("Value weighting requires market_equity in weekly_panel")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    low = "D01"
    high = f"D{n_groups:02d}"
    cfg = FactorConfig(
        name="RSJ_LOW_MINUS_HIGH",
        frequency="weekly",
        n_groups=n_groups,
        weighting=weighting,
        long_groups=(low,),
        short_groups=(high,),
    )
    builder = QuantileSignalFactorBuilder(
        signal_source_col="signal_value",
        config=cfg,
        winsorize=winsorize,
    )
    factor = builder.build({"panel": weekly_panel})
    spread = builder.spread_holdings(factor.holdings)
    factor.metadata.update(
        {
            "definition": "mean_5d((RV_plus - RV_minus) / (RV_plus + RV_minus))",
            "direction": "long low RSJ, short high RSJ",
            "formation_weekday": "Tuesday",
        }
    )
    return factor, spread

