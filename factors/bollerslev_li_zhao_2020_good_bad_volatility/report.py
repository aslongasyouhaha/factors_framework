"""Paper-style weekly quintile report for the RSJ factor."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

from factorlab import data, to_wide  # noqa: E402
from factorlab.report import quantile_report  # noqa: E402


def weekly_crsp_inputs(
    crsp_daily: Path,
    rebalance_dates: pd.DatetimeIndex,
    assets: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compound CRSP daily total returns into (Tuesday, next Tuesday] periods."""
    rebalances = pd.DatetimeIndex(sorted(pd.to_datetime(rebalance_dates).unique()))
    return_parts: list[pd.DataFrame] = []
    me_parts: list[pd.DataFrame] = []
    for file in sorted(crsp_daily.glob("????.parquet")):
        x = pd.read_parquet(file, columns=["date", "asset_id", "ret", "market_equity"])
        x["date"] = pd.to_datetime(x["date"])
        x = x[x["asset_id"].isin(assets)].copy()
        x = x.sort_values(["asset_id", "date"]).drop_duplicates(["asset_id", "date"], keep="last")
        if x.empty:
            continue

        me_parts.append(x[x["date"].isin(rebalances)][["date", "asset_id", "market_equity"]])
        positions = rebalances.searchsorted(x["date"], side="left")
        valid = positions < len(rebalances)
        x = x[valid].copy()
        x["date"] = rebalances[positions[valid]].to_numpy()
        x["gross"] = 1.0 + pd.to_numeric(x["ret"], errors="coerce").fillna(0.0)
        return_parts.append(x.groupby(["date", "asset_id"], as_index=False, observed=True)["gross"].prod())

    if not return_parts:
        raise RuntimeError(f"No CRSP daily observations matched RSJ assets in {crsp_daily}")
    weekly = (
        pd.concat(return_parts, ignore_index=True)
        .groupby(["date", "asset_id"], as_index=False, observed=True)["gross"]
        .prod()
    )
    weekly["ret"] = weekly["gross"] - 1.0
    me = (
        pd.concat(me_parts, ignore_index=True)
        .sort_values(["date", "asset_id"])
        .drop_duplicates(["date", "asset_id"], keep="last")
    )
    return to_wide(weekly, "ret"), to_wide(me, "market_equity")


if __name__ == "__main__":
    exposure, _ = data.load_factor(HERE.name)
    returns, market_equity = weekly_crsp_inputs(
        data.BASE / "crsp_daily",
        pd.DatetimeIndex(exposure.index),
        set(exposure.columns.astype(int)),
    )
    out = quantile_report(HERE.name, HERE / "result", returns=returns, market_equity=market_equity)
    print(out["summary"].to_string())
