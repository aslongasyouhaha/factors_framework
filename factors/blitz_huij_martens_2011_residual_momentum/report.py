from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))
from factorlab.report import quantile_report  # noqa: E402
if __name__ == "__main__":
    print(quantile_report(HERE.name, HERE / "result")["summary"].to_string())
