"""Asset growth: build the factor exposure into data/factors/<project>/.

Paper: Cooper, Gulen & Schill (2008), Asset growth and the cross-section of stock returns, Journal of Finance 63(4)
Exposure: Minus the growth of total assets over the same fiscal quarter a year earlier. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "ASSET_GROWTH",
    "title": "Asset growth",
    "paper": "Cooper, Gulen & Schill (2008), Asset growth and the cross-section of stock returns, Journal of Finance 63(4)",
    "description": "Conservative minus aggressive investment; signal = -asset growth",
    "exposure": "Minus the growth of total assets over the same fiscal quarter a year earlier. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(year_ago_cols=["atq"], exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = -(m["atq"] / m["yoy_atq"].where(m["yoy_atq"].gt(0)) - 1.0)
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
