"""Net operating assets anomaly."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.paper_portfolios import annual_characteristic_factor  # noqa: E402
META={"name":"LOW_NOA","title":"Net operating assets","paper":"Hirshleifer, Hou, Teoh & Zhang (2004)","description":"NOA divided by lagged assets; annual June NYSE-breakpoint 2x3 size/NOA sort; long low NOA","frequency":"monthly","report":{}}
if __name__ == "__main__":
    exposure,detail,returns=annual_characteristic_factor("noa")
    factor_returns=returns[["date","LOW_NOA"]].drop_duplicates()
    print(f"Wrote factor data to {data.save_factor(HERE.name,exposure,META,tables={'formation_detail':detail,'group_returns':returns.drop(columns='LOW_NOA'),'factor_returns':factor_returns})}")
