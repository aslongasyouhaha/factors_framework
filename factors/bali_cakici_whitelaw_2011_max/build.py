"""MAX lottery-demand anomaly."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.extended_factors import daily_characteristic  # noqa: E402
META={"name":"LOW_MAX","title":"Maximum daily return (MAX)","paper":"Bali, Cakici & Whitelaw (2011)","description":"Negative maximum daily return in the previous month","frequency":"monthly","report":{}}
if __name__ == "__main__": print(f"Wrote factor data to {data.save_factor(HERE.name,daily_characteristic('max_return',-1),META)}")
