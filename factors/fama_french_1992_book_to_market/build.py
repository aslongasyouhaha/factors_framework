"""Book-to-market: build the factor exposure into data/factors/<project>/.

Paper: Fama & French (1992), The cross-section of expected stock returns, Journal of Finance 47(2); Rosenberg, Reid & Lanstein (1985)
Exposure: Book equity of the fiscal year ending in calendar year t-1 (announced by June t) over December t-1 market equity, formed each June. Larger values are bought. Rebalanced each June 2008 - 2025.
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
    "name": "BM",
    "title": "Book-to-market",
    "paper": "Fama & French (1992), The cross-section of expected stock returns, Journal of Finance 47(2); Rosenberg, Reid & Lanstein (1985)",
    "description": "High BM minus low BM",
    "exposure": "Book equity of the fiscal year ending in calendar year t-1 (announced by June t) over December t-1 market equity, formed each June.",
    "frequency": "annual",
    "report": {
        "rebalances_per_year": 1
    }
}


def build_exposure() -> pd.DataFrame:
    m = panels.annual_characteristics()
    m["exposure"] = m["bm"]
    return m.rename(columns={"rebalance_date": "date"})[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
