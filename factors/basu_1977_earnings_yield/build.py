"""Earnings yield: build the factor exposure into data/factors/<project>/.

Paper: Basu (1977), Investment performance of common stocks in relation to their price-earnings ratios, Journal of Finance 32(3)
Exposure: Trailing-four-quarter net income / market equity at the month end. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "EARNINGS_YIELD",
    "title": "Earnings yield",
    "paper": "Basu (1977), Investment performance of common stocks in relation to their price-earnings ratios, Journal of Finance 32(3)",
    "description": "Earnings yield = TTM net income / December market equity",
    "exposure": "Trailing-four-quarter net income / market equity at the month end. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = m["net_income_q_ttm"] * 1000.0 / m["market_equity"].where(m["market_equity"].gt(0))
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
