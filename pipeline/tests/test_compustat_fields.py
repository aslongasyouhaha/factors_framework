import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_compustat import ITEMS, add_derived_fields  # noqa: E402


def quarters(**overrides) -> pd.DataFrame:
    """Five consecutive quarters of one company with simple values."""
    q = pd.DataFrame({
        "gvkey": "1",
        "datadate": pd.to_datetime(["2009-12-31", "2010-03-31", "2010-06-30", "2010-09-30", "2010-12-31"]),
        "rdq": pd.to_datetime(["2010-02-10", "2010-05-05", None, "2010-09-01", "2011-02-15"]),
        "fyearq": [2009, 2010, 2010, 2010, 2010],
        "fqtr": [4, 1, 2, 3, 4],
    })
    for col in ITEMS:
        q[col] = 1.0
    q["capxy"] = [40.0, 10.0, 25.0, 30.0, 50.0]  # year-to-date
    for k, v in overrides.items():
        q[k] = v
    return q


def test_capex_is_converted_from_year_to_date_to_quarters():
    out = add_derived_fields(quarters())
    assert out["capx_q"].tolist()[1:] == [10.0, 15.0, 5.0, 20.0]
    assert out["capx_q_ttm"].iloc[-1] == 50.0  # four quarters of fiscal 2010 = full-year capex


def test_ttm_needs_four_consecutive_quarters():
    q = quarters().drop(index=2)  # June 2010 quarter missing
    out = add_derived_fields(q)
    assert out["saleq_ttm"].isna().all()
    full = add_derived_fields(quarters())
    assert full["saleq_ttm"].tolist()[3:] == [4.0, 4.0]


def test_available_date_falls_back_when_rdq_is_missing_or_too_early():
    out = add_derived_fields(quarters())
    # Quarter 3 (June 2010) has no rdq; quarter 4 (September 2010) reports before its own quarter end.
    assert out.loc[2, "available_date"] == pd.Timestamp("2010-06-30") + pd.Timedelta(days=90)
    assert out.loc[3, "available_date"] == pd.Timestamp("2010-09-30") + pd.Timedelta(days=90)
    assert out.loc[1, "available_date"] == pd.Timestamp("2010-05-05")
    assert out["rdq_imputed"].tolist() == [False, False, True, True, False]
    assert np.all(out["available_date"] >= out["datadate"])
