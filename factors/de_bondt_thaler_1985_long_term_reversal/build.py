"""Long-term reversal from months t-60 through t-13."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.extended_factors import long_term_reversal  # noqa: E402
META={"name":"LTREV","title":"Long-term reversal","paper":"De Bondt & Thaler (1985); Jegadeesh & Titman (2001)","description":"Negative compounded return over months t-60 through t-13","frequency":"monthly","report":{}}
if __name__ == "__main__": print(f"Wrote factor data to {data.save_factor(HERE.name,long_term_reversal(),META)}")
