from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data
from factorlab.additional_factors import sue
META={"name":"SUE","title":"Standardized unexpected earnings","paper":"Jegadeesh & Livnat (2006)","description":"Seasonal change in split-adjusted quarterly EPS divided by the prior eight-surprise volatility; observed RDQ only","frequency":"monthly","report":{}}
if __name__=="__main__":
    exposure,detail=sue(); print(data.save_factor(HERE.name,exposure,META,tables={"detail":detail}))
