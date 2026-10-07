import pandas as pd

from factorlab import data
from factorlab.extended_factors import daily_characteristic
from factorlab.paper_portfolios import _buckets, _factor_from_groups
from factorlab.additional_factors import net_share_issuance


def test_daily_characteristic_applies_direction_and_sample(monkeypatch):
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2007-12-31", "2008-01-31", "2025-11-30", "2025-12-31"]),
            "asset_id": [1, 1, 2, 2],
            "max_return": [0.1, 0.2, 0.3, 0.4],
        }
    )
    monkeypatch.setattr(data, "load_base", lambda name, columns=None: source[columns].copy())
    got = daily_characteristic("max_return", direction=-1)
    assert got["date"].tolist() == [pd.Timestamp("2008-01-31"), pd.Timestamp("2025-11-30")]
    assert got["exposure"].tolist() == [-0.2, -0.3]


def test_paper_2x3_uses_nyse_breakpoints():
    frame = pd.DataFrame({
        "rebalance_date": pd.to_datetime(["2020-06-30"] * 6),
        "asset_id": range(6),
        "exchange_code": [1, 1, 1, 1, 2, 3],
        "market_equity": [10, 20, 30, 40, 15, 50],
        "characteristic": [1, 2, 3, 4, 0, 5],
    })
    got = _buckets(frame, "characteristic", 0.25, 0.75).set_index("asset_id")
    assert got.loc[4, "group"] == "SL"
    assert got.loc[5, "group"] == "BH"


def test_factor_is_average_high_minus_average_low():
    groups = pd.DataFrame({
        "date": pd.to_datetime(["2020-07-31"] * 6),
        "group": ["SL", "SM", "SH", "BL", "BM", "BH"],
        "ret": [0.01, 0.02, 0.05, 0.03, 0.04, 0.09],
    })
    got = _factor_from_groups(groups, "X")
    assert abs(got.loc[0, "X"] - 0.05) < 1e-12


def test_net_issuance_removes_ex_dividend_price_growth(monkeypatch):
    dates = pd.date_range("2020-01-31", periods=20, freq="ME")
    source = pd.DataFrame({"date": dates, "asset_id": 1, "ret": 0.01, "retx": 0.01,
                           "market_equity_security": 100 * 1.01 ** pd.RangeIndex(20)})
    monkeypatch.setattr("factorlab.additional_factors.panels.crsp_monthly", lambda: source.copy())
    got = net_share_issuance()
    assert got["exposure"].abs().max() < 1e-10
