"""Short-term reversal: build the factor exposure into data/factors/<project>/.

Paper: Jegadeesh (1990), Evidence of predictable behavior of security returns, Journal of Finance 45(3); Lehmann (1990), Quarterly Journal of Economics
Exposure: Minus the stock's return in the formation month. Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "REV_1M",
    "title": "Short-term reversal",
    "paper": "Jegadeesh (1990), Evidence of predictable behavior of security returns, Journal of Finance 45(3); Lehmann (1990), Quarterly Journal of Economics",
    "description": "Prior one-month loser minus winner",
    "exposure": "Minus the stock's return in the formation month.",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = -m["ret"]
    m = m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)]
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
