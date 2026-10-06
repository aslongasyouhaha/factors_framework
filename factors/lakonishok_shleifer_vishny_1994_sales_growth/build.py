"""Sales growth: build the factor exposure into data/factors/<project>/.

Paper: Lakonishok, Shleifer & Vishny (1994), Contrarian investment, extrapolation, and risk, Journal of Finance 49(5)
Exposure: Growth of trailing-four-quarter sales over the same fiscal quarter a year earlier. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "SALES_GROWTH",
    "title": "Sales growth",
    "paper": "Lakonishok, Shleifer & Vishny (1994), Contrarian investment, extrapolation, and risk, Journal of Finance 49(5)",
    "description": "High TTM sales growth minus low sales growth",
    "exposure": "Growth of trailing-four-quarter sales over the same fiscal quarter a year earlier. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(year_ago_cols=["saleq_ttm"], exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = m["saleq_ttm"] / m["yoy_saleq_ttm"].where(m["yoy_saleq_ttm"].gt(0)) - 1.0
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
