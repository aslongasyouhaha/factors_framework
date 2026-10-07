"""Conservative-minus-aggressive investment characteristic."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.paper_portfolios import annual_characteristic_factor  # noqa: E402
META={"name":"CMA","title":"Conservative minus aggressive investment","paper":"Fama & French (2015), A five-factor asset pricing model","description":"Annual June independent 2x3 size/asset-growth sort; NYSE median and 30/70 breakpoints; value-weighted monthly returns","frequency":"monthly","report":{}}
if __name__ == "__main__":
    exposure,detail,returns=annual_characteristic_factor("cma")
    factor_returns=returns[["date","CMA"]].drop_duplicates()
    print(f"Wrote factor data to {data.save_factor(HERE.name,exposure,META,tables={'formation_detail':detail,'group_returns':returns.drop(columns='CMA'),'factor_returns':factor_returns})}")
