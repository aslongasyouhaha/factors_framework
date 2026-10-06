"""Illiquidity: build the factor exposure into data/factors/<project>/.

Paper: Amihud (2002), Illiquidity and stock returns: cross-section and time-series effects, Journal of Financial Markets 5(1)
Exposure: |monthly return| / average daily dollar volume in the month. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "ILLIQ",
    "title": "Illiquidity",
    "paper": "Amihud (2002), Illiquidity and stock returns: cross-section and time-series effects, Journal of Financial Markets 5(1)",
    "description": "High Amihud illiquidity minus low illiquidity",
    "exposure": "|monthly return| / average daily dollar volume in the month.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = m["ret"].abs() / m["dollar_volume"].where(m["dollar_volume"].gt(0))
    m = m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)]
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
