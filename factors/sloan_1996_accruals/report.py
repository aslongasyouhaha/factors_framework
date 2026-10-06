"""Standard result page: decile portfolios, long-short spread, turnover, IC and yearly performance."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

from factorlab.report import quantile_report  # noqa: E402

if __name__ == "__main__":
    out = quantile_report(HERE.name, HERE / "result")
    print(out["summary"].to_string())
