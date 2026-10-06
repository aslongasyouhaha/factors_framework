"""Return on assets: build the factor exposure into data/factors/<project>/.

Paper: Haugen & Baker (1996), Commonality in the determinants of expected stock returns, Journal of Financial Economics 41(3)
Exposure: Net income (income before extraordinary items when missing) over the last four quarters / total assets. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "ROA",
    "title": "Return on assets",
    "paper": "Haugen & Baker (1996), Commonality in the determinants of expected stock returns, Journal of Financial Economics 41(3)",
    "description": "Return on assets = TTM net income / assets",
    "exposure": "Net income (income before extraordinary items when missing) over the last four quarters / total assets. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = m["net_income_q_ttm"] / assets
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
