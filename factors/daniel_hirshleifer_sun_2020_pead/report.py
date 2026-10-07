from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/"framework"))
from factorlab.report import paper_factor_report  # noqa: E402
if __name__ == "__main__": print(paper_factor_report(HERE.name,HERE/"result","PEAD")["summary"].to_string())
