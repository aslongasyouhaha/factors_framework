from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.report import quantile_report,weekly_crsp_inputs
if __name__=="__main__":
    e,_=data.load_factor(HERE.name); r,m=weekly_crsp_inputs(e.index,set(e.columns.astype(int))); print(quantile_report(HERE.name,HERE/"result",returns=r,market_equity=m)["summary"].to_string())
