"""Build point-in-time earnings-announcement abnormal returns from observed Compustat RDQ."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))
from factorlab import data, panels  # noqa: E402
from factorlab.fama_french import link_to_permno  # noqa: E402


def _load(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["date", "asset_id", "ret", "market_equity"])
    d["date"] = pd.to_datetime(d["date"])
    return d


def build() -> pd.DataFrame:
    q = panels.compustat_quarterly()
    q = q[q["rdq"].notna() & ~q["rdq_imputed"]].copy()
    lag = (q["rdq"] - q["datadate"]).dt.days
    q = q[lag.between(0, 180)]
    q = link_to_permno(q, data.load_base("ccm_link"), key="datadate")
    events = q[["asset_id", "gvkey", "datadate", "rdq"]].drop_duplicates(["asset_id", "datadate"])
    files = {int(p.stem): p for p in (data.BASE / "crsp_daily").glob("*.parquet")}
    pieces = []
    for year, path in sorted(files.items()):
        ev = events[events["rdq"].dt.year.eq(year)].copy()
        if ev.empty:
            continue
        paths = [files[y] for y in (year - 1, year, year + 1) if y in files]
        daily = pd.concat([_load(p) for p in paths], ignore_index=True).drop_duplicates(["date", "asset_id"])
        daily = daily.sort_values(["asset_id", "date"])
        daily["lag_me"] = daily.groupby("asset_id")["market_equity"].shift(1)
        valid = daily["ret"].notna() & daily["lag_me"].gt(0)
        market = (daily["ret"] * daily["lag_me"]).where(valid).groupby(daily["date"]).sum(min_count=1) / daily["lag_me"].where(valid).groupby(daily["date"]).sum(min_count=1)
        daily["abret"] = daily["ret"] - daily["date"].map(market)
        calendar = np.array(sorted(daily["date"].unique()), dtype="datetime64[ns]")
        pos = np.searchsorted(calendar, ev["rdq"].to_numpy(dtype="datetime64[ns]"), side="left")
        # DHS use the four trading days [-2,+1] and only admit an announcement
        # to a month-end sort when RDQ is at least two trading days old.
        in_bounds = (pos >= 2) & (pos + 2 < len(calendar))
        ev = ev.loc[in_bounds].copy()
        pos = pos[in_bounds]
        ev["event_date"] = pd.to_datetime(calendar[pos])
        ev["available_date"] = pd.to_datetime(calendar[pos + 2])
        windows = []
        for offset in (-2, -1, 0, 1):
            part = ev[["asset_id", "gvkey", "datadate", "rdq", "event_date", "available_date"]].copy()
            part["trade_date"] = pd.to_datetime(calendar[pos + offset])
            part["offset"] = offset
            windows.append(part)
        window = pd.concat(windows, ignore_index=True).merge(
            daily[["asset_id", "date", "abret"]], left_on=["asset_id", "trade_date"], right_on=["asset_id", "date"], how="left"
        )
        out = window.groupby(["asset_id", "gvkey", "datadate", "rdq", "event_date", "available_date"], as_index=False).agg(
            pead_car=("abret", "sum"), n_event_days=("abret", "count")
        )
        pieces.append(out[out["n_event_days"].ge(2)])
        print(f"Processed announcements in {year}")
    return pd.concat(pieces, ignore_index=True).sort_values(["asset_id", "available_date"])


if __name__ == "__main__":
    result = build()
    path = data.save_base(result, "pead_events")
    print(f"Wrote {len(result):,} announcement events to {path}")
