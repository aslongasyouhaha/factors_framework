"""Kenneth French data library -> data/base/rf_monthly.parquet (monthly risk-free rate).

RF is the one-month Treasury bill return from F-F_Research_Data_Factors (decimal per month),
the risk-free rate behind French's published factors.

    python pipeline/build_rf.py
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from common import SOURCE, save_base

KF_ZIP = SOURCE / "kenneth_french" / "F-F_Research_Data_Factors_CSV.zip"


def kenneth_french_monthly() -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RF from the French factor file (decimal returns)."""
    with zipfile.ZipFile(KF_ZIP) as z:
        lines = z.open(z.namelist()[0]).read().decode("latin-1").splitlines()
    rows = [l for l in lines if len(l.split(",")[0].strip()) == 6 and l.split(",")[0].strip().isdigit()]
    kf = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["yyyymm", "MKT_RF", "SMB", "HML", "RF"])
    kf["date"] = pd.to_datetime(kf["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    kf[["MKT_RF", "SMB", "HML", "RF"]] = kf[["MKT_RF", "SMB", "HML", "RF"]] / 100.0
    return kf[["date", "MKT_RF", "SMB", "HML", "RF"]]


def main() -> None:
    kf = kenneth_french_monthly()
    rf = kf[["date", "RF"]]
    print(f"Wrote {len(rf):,} months ({rf['date'].min():%Y-%m} to {rf['date'].max():%Y-%m}) to {save_base(rf, 'rf_monthly')}")
    print(f"Wrote French factors to {save_base(kf, 'french_factors_monthly')}")


if __name__ == "__main__":
    main()
