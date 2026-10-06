"""Low volatility: build the factor exposure into data/factors/<project>/.

Paper: Ang, Hodrick, Xing & Zhang (2006), The cross-section of volatility and expected returns, Journal of Finance 61(1); Blitz & van Vliet (2007), The volatility effect, Journal of Portfolio Management
Exposure: Minus the standard deviation of the last 12 monthly returns (at least 8). Larger values are bought. Rebalanced monthly, January 2008 - November 2025.
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
    "name": "LOW_VOL_12M",
    "title": "Low volatility",
    "paper": "Ang, Hodrick, Xing & Zhang (2006), The cross-section of volatility and expected returns, Journal of Finance 61(1); Blitz & van Vliet (2007), The volatility effect, Journal of Portfolio Management",
    "description": "Low trailing 12-month volatility minus high volatility",
    "exposure": "Minus the standard deviation of the last 12 monthly returns (at least 8).",
    "frequency": "monthly",
    "report": {}
}


def build_exposure() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = -m.groupby("asset_id")["ret"].transform(lambda x: x.rolling(12, min_periods=8).std())
    m = m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)]
    return m[["date", "asset_id", "exposure"]]


if __name__ == "__main__":
    path = data.save_factor(HERE.name, build_exposure(), META)
    print(f"Wrote factor data to {path}")
