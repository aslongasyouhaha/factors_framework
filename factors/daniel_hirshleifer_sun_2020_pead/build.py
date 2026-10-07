"""Post-earnings-announcement drift from observed Compustat report dates."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))
from factorlab import data  # noqa: E402
from factorlab.paper_portfolios import pead_factor  # noqa: E402
META = {"name":"PEAD","title":"Post-earnings-announcement drift","paper":"Daniel, Hirshleifer & Sun (2020)","description":"Four-day [-2,+1] market-adjusted announcement return; monthly independent 2x3 size/CAR sort with NYSE median and 20/80 breakpoints; value-weighted","frequency":"monthly","report":{}}
if __name__ == "__main__":
    exposure, detail, returns = pead_factor()
    factor_returns=returns[["date","PEAD"]].drop_duplicates()
    print(f"Wrote factor data to {data.save_factor(HERE.name, exposure, META, tables={'announcement_detail': detail,'group_returns':returns.drop(columns='PEAD'),'factor_returns':factor_returns})}")
