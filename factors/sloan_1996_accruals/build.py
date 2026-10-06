"""Accruals: build the factor exposure into data/factors/<project>/.

Paper: Sloan (1996), Do stock prices fully reflect information in accruals and cash flows about future earnings?, The Accounting Review 71(3)
Exposure: Minus balance-sheet accruals (change over the same quarter a year earlier in non-cash current assets minus change in current liabilities excluding short-term debt, minus trailing-four-quarter depreciation) / average total assets. Latest fiscal quarter announced by the month end. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "LOW_ACCRUALS",
    "title": "Accruals",
    "paper": "Sloan (1996), Do stock prices fully reflect information in accruals and cash flows about future earnings?, The Accounting Review 71(3)",
    "description": "Low accruals minus high accruals; signal = -accruals/assets",
    "exposure": "Minus balance-sheet accruals (change over the same quarter a year earlier in non-cash current assets minus change in current liabilities excluding short-term debt, minus trailing-four-quarter depreciation) / average total assets. Latest fiscal quarter announced by the month end.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.monthly_accounting_panel(year_ago_cols=["atq", "actq", "cheq", "lctq", "dlcq"], exclude_financials=True)
    # A blank debt in current liabilities means no short-term debt.
    short_debt_change = m["dlcq"].fillna(0.0) - m["yoy_dlcq"].fillna(0.0)
    accruals = (
        (m["actq"] - m["yoy_actq"])
        - (m["cheq"] - m["yoy_cheq"])
        - ((m["lctq"] - m["yoy_lctq"]) - short_debt_change)
        - m["dpq_ttm"].fillna(0.0)
    )
    average_assets = ((m["atq"] + m["yoy_atq"]) / 2.0).where(lambda a: a.gt(0))
    m["exposure"] = -(accruals / average_assets)
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
