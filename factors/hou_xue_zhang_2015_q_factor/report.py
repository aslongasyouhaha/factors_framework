from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab.report import paper_factor_report
if __name__=="__main__":
    for c in ("r_ME","r_IA","r_ROE"): print(c); print(paper_factor_report(HERE.name,HERE/"result"/c,c)["summary"].to_string())
