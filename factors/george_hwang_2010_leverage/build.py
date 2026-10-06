"""Leverage: build the factor exposure into data/factors/<project>/.

Paper: George & Hwang (2010), A resolution of the distress risk and leverage puzzles, Journal of Financial Economics 96(1); Bhandari (1988), Debt/equity ratio and expected common stock returns, Journal of Finance 43(2)
Exposure: Minus (long-term debt + debt in current liabilities) / total assets. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "LOW_LEVERAGE",
    "title": "Leverage",
    "paper": "George & Hwang (2010), A resolution of the distress risk and leverage puzzles, Journal of Financial Economics 96(1); Bhandari (1988), Debt/equity ratio and expected common stock returns, Journal of Finance 43(2)",
    "description": "Low debt/assets minus high debt/assets; signal = -leverage",
    "exposure": "Minus (long-term debt + debt in current liabilities) / total assets. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = -((m["dlttq"].fillna(0.0) + m["dlcq"].fillna(0.0)) / assets)
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
