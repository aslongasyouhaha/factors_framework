"""Operating profitability: build the factor exposure into data/factors/<project>/.

Paper: Ball, Gerakos, Linnainmaa & Nikolaev (2015), Deflating profitability, Journal of Financial Economics 117(2); Fama & French (2015), A five-factor asset pricing model, Journal of Financial Economics 116(1)
Exposure: Operating income after depreciation over the last four quarters / total assets. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "OP_PROFIT",
    "title": "Operating profitability",
    "paper": "Ball, Gerakos, Linnainmaa & Nikolaev (2015), Deflating profitability, Journal of Financial Economics 117(2); Fama & French (2015), A five-factor asset pricing model, Journal of Financial Economics 116(1)",
    "description": "Operating profitability = TTM operating income / assets",
    "exposure": "Operating income after depreciation over the last four quarters / total assets. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = m["oiadpq_ttm"] / assets
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
