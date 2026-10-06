"""Size effect: build the factor exposure into data/factors/<project>/.

Paper: Banz (1981), The relationship between return and market value of common stocks, Journal of Financial Economics 9(1)
Exposure: -log(company market equity) at each month end; value-weighted decile spread rebalanced monthly. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "SIZE",
    "title": "Size effect",
    "paper": "Banz (1981), The relationship between return and market value of common stocks, Journal of Financial Economics 9(1)",
    "description": "Small minus big; signal = -log(ME)",
    "exposure": "-log(company market equity) at each month end; value-weighted decile spread rebalanced monthly.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = -np.log(m["market_equity"].where(m["market_equity"].gt(0)))
    m = m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)]
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
