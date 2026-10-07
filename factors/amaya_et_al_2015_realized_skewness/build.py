from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.additional_factors import weekly_realized_moment
META={"name":"LOW_RSK","title":"Realized skewness","paper":"Amaya, Christoffersen, Jacobs & Vasquez (2015)","description":"Negative five-day mean realized skewness from 5-minute intraday returns; long low skewness","frequency":"weekly (Tuesday close)","report":{"n_groups":10,"weighting":"value","long_groups":["D10"],"short_groups":["D01"],"winsorize":[0.01,0.99],"periods_per_year":52,"rebalances_per_year":52}}
if __name__=="__main__": print(data.save_factor(HERE.name,weekly_realized_moment("rsk",-1),META))
