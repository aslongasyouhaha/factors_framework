"""Fama-French three factors: build MKT_RF, SMB, HML into data/factors/<project>/.

Paper: Fama & French (1993), Common risk factors in the returns on stocks and bonds, Journal of
Financial Economics 33(1).

Each June, stocks whose book equity has been announced are split at the NYSE median market equity (S / B) and at the NYSE 30th / 70th
book-to-market percentiles (L / M / H). The six portfolios are value-weighted by lagged market
equity every month from July to June. SMB = mean(SL, SM, SH) - mean(BL, BM, BH),
HML = mean(SH, BH) - mean(SL, BL), MKT_RF = value-weighted market return - RF.

Outputs: returns.parquet (date, MKT_RF, SMB, HML, MKT, RF), holdings.parquet,
characteristics.parquet (June size and book-to-market of every stock), diagnostics.parquet.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

from factorlab import FamaFrench3Builder, data, panels  # noqa: E402

START_YEAR, END_YEAR = panels.SAMPLE_YEARS  # June formations; returns July 2008 - December 2025

META = {
    "name": "FF3",
    "title": "Fama-French three factors",
    "paper": "Fama & French (1993), Common risk factors in the returns on stocks and bonds, Journal of Financial Economics 33(1)",
    "description": "MKT_RF, SMB and HML from 2x3 size / book-to-market portfolios, formed each June with NYSE breakpoints",
    "frequency": "monthly returns, annual formation",
    "tables": {
        "returns": "date, MKT_RF, SMB, HML, MKT, RF (decimal monthly returns)",
        "holdings": "June holdings of the six portfolios with target weights",
        "characteristics": "June market equity, book-to-market and exchange of every stock in the universe",
        "diagnostics": "number of stocks and weights per portfolio and formation date",
    },
}


def build() -> dict:
    chars = panels.annual_characteristics(START_YEAR, END_YEAR)
    returns = panels.crsp_monthly()[["date", "asset_id", "ret", "market_equity"]]
    factor = FamaFrench3Builder().build({"characteristics": chars, "returns": returns, "risk_free": panels.risk_free()})
    return {
        "returns": factor.returns,
        "holdings": factor.holdings,
        "characteristics": chars,
        "diagnostics": factor.diagnostics,
    }


if __name__ == "__main__":
    tables = build()
    path = data.save_factor(HERE.name, None, META, tables)
    print(tables["returns"].head().to_string(index=False))
    print(f"Wrote factor data to {path}")
