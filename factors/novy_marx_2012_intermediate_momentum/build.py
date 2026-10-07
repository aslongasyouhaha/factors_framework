"""Intermediate-horizon momentum."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab import data  # noqa: E402
from factorlab.extended_factors import intermediate_momentum  # noqa: E402
META={"name":"INTERM_MOM","title":"Intermediate momentum","paper":"Novy-Marx (2012), Is momentum really momentum?","description":"Compounded return over months t-12 through t-7","frequency":"monthly","report":{}}
if __name__ == "__main__": print(f"Wrote factor data to {data.save_factor(HERE.name,intermediate_momentum(),META)}")
