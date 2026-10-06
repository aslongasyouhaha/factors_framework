"""Cash holdings: build the factor exposure into data/factors/<project>/.

Paper: Palazzo (2012), Cash holdings, risk, and expected returns, Journal of Financial Economics 104(1)
Exposure: Cash and short-term investments / total assets. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "CASH",
    "title": "Cash holdings",
    "paper": "Palazzo (2012), Cash holdings, risk, and expected returns, Journal of Financial Economics 104(1)",
    "description": "High cash/assets minus low cash/assets",
    "exposure": "Cash and short-term investments / total assets. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(exclude_financials=True)
    assets = m["atq"].where(m["atq"].gt(0))
    m["exposure"] = m["cheq"] / assets
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
