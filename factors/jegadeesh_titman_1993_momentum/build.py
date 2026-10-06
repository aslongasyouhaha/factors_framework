"""Momentum: build the factor exposure into data/factors/<project>/.

Paper: Jegadeesh & Titman (1993), Returns to buying winners and selling losers, Journal of Finance 48(1); 12-2 convention from Carhart (1997)
Exposure: Compounded return from month t-12 to t-2 (11 months, skipping the most recent month); at least 8 observed months and no missing month in the window. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorlab import data, panels  # noqa: E402

META = {
    "name": "MOM_12_2",
    "title": "Momentum",
    "paper": "Jegadeesh & Titman (1993), Returns to buying winners and selling losers, Journal of Finance 48(1); 12-2 convention from Carhart (1997)",
    "description": "Prior 12-to-2 month winner minus loser",
    "exposure": "Compounded return from month t-12 to t-2 (11 months, skipping the most recent month); at least 8 observed months and no missing month in the window.",
    "frequency": "monthly",
    "report": {}
}


def rolling_compound(ret: pd.Series, asset: pd.Series, skip: int, window: int, min_periods: int) -> pd.Series:
    """Per-asset compounded return over `window` rows ending `skip` rows back.

    Same as rolling(window, min_periods).apply(np.prod) on (1 + ret).shift(skip): NaN whenever
    the window holds a missing return, but computed as a rolling log-sum.
    """
    gross = (1.0 + ret).groupby(asset).shift(skip)
    missing = gross.isna().astype(float)
    wiped = gross.le(0).astype(float)
    log_gross = np.log(gross.where(gross.gt(0), 1.0)).where(gross.notna())

    def roll(s: pd.Series, mp: int) -> pd.Series:
        return s.groupby(asset).rolling(window, min_periods=mp).sum().reset_index(level=0, drop=True)

    out = np.exp(roll(log_gross, min_periods)) - 1.0
    out = out.where(roll(wiped, 1).eq(0), -1.0).where(out.notna())
    return out.where(roll(missing, 1).eq(0))


def build_exposure() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = rolling_compound(m["ret"], m["asset_id"], skip=2, window=11, min_periods=8)
    m = m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)]
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
