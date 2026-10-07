"""Betting-against-beta stock signal."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.paper_portfolios import bab_factor  # noqa: E402
META={"name":"BAB","title":"Betting against beta","paper":"Frazzini & Pedersen (2014), Betting Against Beta","description":"Five-year three-day correlation and one-year volatility beta, shrunk 60% toward one; rank-weighted low/high legs separately scaled to unit formation beta","frequency":"monthly","report":{}}
if __name__ == "__main__":
    exposure,detail,returns=bab_factor(); print(f"Wrote factor data to {data.save_factor(HERE.name,exposure,META,tables={'beta_detail':detail,'factor_returns':returns})}")
