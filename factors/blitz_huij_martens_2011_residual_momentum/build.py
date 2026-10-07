"""FF3 residual momentum."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))
from factorlab import data  # noqa: E402
from factorlab.momentum import residual_momentum  # noqa: E402

META = {
    "name": "RES_MOM",
    "title": "Fama-French residual momentum",
    "paper": "Blitz, Huij & Martens (2011); reviewed in Baltussen et al. (2025)",
    "description": "12-2 rolling FF3 residual return standardised by residual volatility",
    "frequency": "monthly",
    "report": {},
}

if __name__ == "__main__":
    print(f"Wrote factor data to {data.save_factor(HERE.name, residual_momentum(), META)}")
