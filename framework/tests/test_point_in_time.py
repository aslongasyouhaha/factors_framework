import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import factorlab.data as data  # noqa: E402
from factorlab import panels  # noqa: E402
from factorlab.fama_french import annual_characteristics  # noqa: E402

D = pd.Timestamp


@pytest.fixture
def base(tmp_path, monkeypatch):
    """A tiny data/base: three companies, monthly CRSP rows for 2010-2011."""
    monkeypatch.setattr(data, "BASE", tmp_path)
    months = pd.date_range("2010-01-31", "2011-12-31", freq="ME")
    crsp = pd.DataFrame(
        [(d, a) for a in (1, 2, 3) for d in months], columns=["date", "asset_id"]
    ).assign(company_id=lambda x: x["asset_id"], ret=0.01, market_equity=100.0, dollar_volume=1.0,
             exchange_code=1.0, first_trade_date=D("2000-01-01"), ff_primary_share=True,
             siccd=lambda x: x["asset_id"].map({1: 3570.0, 2: 6020.0, 3: float("nan")}))
    rows = [
        # gvkey, datadate, available_date, fyear, fqtr, atq
        ("000001", "2010-03-31", "2010-05-10", 2010, 1, 10.0),
        ("000001", "2010-06-30", "2010-08-09", 2010, 2, 11.0),
        ("000001", "2010-09-30", "2010-11-08", 2010, 3, 12.0),
        ("000001", "2010-12-31", "2011-02-20", 2010, 4, 13.0),
        ("000001", "2011-03-31", "2011-05-10", 2011, 1, 14.0),
        # company 2: the September quarter is filed after the December quarter
        ("000002", "2010-09-30", "2011-01-25", 2010, 3, 50.0),
        ("000002", "2010-12-31", "2011-01-20", 2010, 4, 60.0),
        # company 3: reports once, then goes stale
        ("000003", "2010-03-31", "2010-05-10", 2010, 1, 7.0),
    ]
    q = pd.DataFrame(rows, columns=["gvkey", "datadate", "available_date", "fyear", "fqtr", "atq"])
    q[["datadate", "available_date"]] = q[["datadate", "available_date"]].apply(pd.to_datetime)
    q["book_equity"] = q["atq"]
    link = pd.DataFrame({"gvkey": ["000001", "000002", "000003"], "asset_id": [1, 2, 3],
                         "linkdt": D("1990-01-01"), "linkenddt": D("2262-04-11"), "LINKTYPE": "LC", "LINKPRIM": "P"})
    data.save_base(crsp, "crsp_monthly")
    data.save_base(q, "compustat_quarterly")
    data.save_base(link, "ccm_link")


def latest(panel: pd.DataFrame, asset: int, date: str):
    row = panel[(panel["asset_id"] == asset) & (panel["date"] == D(date))]
    return None if row.empty else row.iloc[0]


def test_only_announced_quarters_are_used(base):
    p = panels.monthly_accounting_panel("2010-01-31", "2011-12-31", year_ago_cols=["atq"])
    assert latest(p, 1, "2010-04-30") is None  # nothing public yet
    assert latest(p, 1, "2010-05-31")["datadate"] == D("2010-03-31")
    assert latest(p, 1, "2010-07-31")["datadate"] == D("2010-03-31")  # June quarter public only on Aug 9
    assert latest(p, 1, "2010-08-31")["datadate"] == D("2010-06-30")
    assert (p["available_date"] <= p["date"]).all()


def test_late_filing_of_an_older_quarter_does_not_replace_a_newer_one(base):
    p = panels.monthly_accounting_panel("2010-01-31", "2011-12-31")
    assert latest(p, 2, "2010-12-31") is None
    assert latest(p, 2, "2011-01-31")["datadate"] == D("2010-12-31")
    assert latest(p, 2, "2011-01-31")["atq"] == 60.0


def test_stale_quarters_are_dropped(base):
    p = panels.monthly_accounting_panel("2010-01-31", "2011-12-31")
    assert latest(p, 3, "2011-03-31")["datadate"] == D("2010-03-31")  # 365 days old
    assert latest(p, 3, "2011-04-30") is None  # 395 days old


def test_year_ago_values_come_from_the_same_quarter_last_year(base):
    p = panels.monthly_accounting_panel("2010-01-31", "2011-12-31", year_ago_cols=["atq"])
    row = latest(p, 1, "2011-05-31")
    assert row["datadate"] == D("2011-03-31") and row["yoy_atq"] == 10.0
    assert pd.isna(latest(p, 1, "2011-03-31")["yoy_atq"])  # December 2010 quarter has no year-ago row


def test_june_formation_skips_fiscal_years_not_yet_announced():
    monthly = pd.DataFrame({
        "date": [D("2010-12-31"), D("2011-06-30")] * 2,
        "asset_id": [1, 1, 2, 2],
        "company_id": [1, 1, 2, 2],
        "market_equity": 100.0,
        "exchange_code": 1.0,
        "first_trade_date": D("2000-01-01"),
        "dollar_volume": 1.0,
        "ff_primary_share": True,
    })
    be = pd.DataFrame({
        "asset_id": [1, 2],
        "fyear": [2010, 2010],
        "datadate": [D("2010-12-31"), D("2010-12-31")],
        "available_date": [D("2011-03-01"), D("2011-07-15")],  # company 2 announces after June 30
        "book_equity": [50.0, 50.0],
    })
    chars = annual_characteristics(monthly, be, 2011, 2011)
    assert chars["asset_id"].tolist() == [1]


def test_financial_firms_can_be_excluded(base):
    p = panels.monthly_accounting_panel("2010-01-31", "2011-12-31", exclude_financials=True)
    assert 2 not in set(p["asset_id"])  # SIC 6020: a bank
    assert {1, 3} <= set(p["asset_id"])  # unknown SIC is kept
    assert 2 in set(panels.monthly_accounting_panel("2010-01-31", "2011-12-31")["asset_id"])
