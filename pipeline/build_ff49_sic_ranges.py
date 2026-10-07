"""Parse Kenneth French's official Siccodes49.zip into data/base/ff49_sic_ranges.parquet."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from factorlab import data  # noqa: E402


def main() -> None:
    source = data.SOURCE / "Siccodes49.zip"
    if not source.exists():
        raise FileNotFoundError(f"Download https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip to {source}")
    with zipfile.ZipFile(source) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin1")
    industry = None
    rows = []
    for line in text.splitlines():
        head = re.match(r"\s*(\d+)\s+([A-Za-z0-9]+)\s+", line)
        if head:
            industry = f"{int(head.group(1)):02d}_{head.group(2)}"
            continue
        span = re.match(r"\s*(\d{4})-(\d{4})\s+", line)
        if span and industry:
            rows.append((int(span.group(1)), int(span.group(2)), industry))
    out = pd.DataFrame(rows, columns=["sic_low", "sic_high", "industry"])
    if out.empty or out["industry"].nunique() != 49:
        raise ValueError("Could not parse all 49 industries")
    path = data.save_base(out, "ff49_sic_ranges")
    print(f"Wrote {len(out)} SIC ranges for 49 industries to {path}")


if __name__ == "__main__":
    main()
