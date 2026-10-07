import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "framework"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsj import (  # noqa: E402
    build_rsj_factor,
    make_weekly_rsj_panel,
    realized_measures_from_bars,
    realized_measures_from_minute_ohlcv,
)
from build_rsj_daily import filter_bad_price_bars  # noqa: E402


def test_bad_tick_filter_removes_isolated_spike_but_keeps_persistent_jump():
    bars = pd.DataFrame(
        {
            "asset_id": [1] * 8 + [2] * 8,
            "date_key": [20240102] * 16,
            "session_minute": list(range(8)) * 2,
            "daily_price": [10.0] * 8 + [20.0] * 8,
            "open": [10, 10, 10, 16, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
            "close": [10, 10, 10, 16, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
        }
    )
    out, reference_dropped, local_dropped = filter_bad_price_bars(
        bars, min_reference_ratio=0.5, max_reference_ratio=2.0, max_local_ratio=1.5
    )
    assert reference_dropped == 0
    assert local_dropped == 1
    assert 3 not in set(out.loc[out["asset_id"].eq(1), "session_minute"])
    assert len(out[out["asset_id"].eq(2)]) == 8


def test_one_minute_ohlcv_aggregates_to_78_returns_and_uses_1600_open():
    rows = []
    price = 100.0
    for minute in range(390):
        close = 101.0
        rows.append(
            {
                "asset_id": 1,
                "date": 20240102,
                "session_minute": minute,
                "open": price if minute == 0 else close,
                "close": close,
            }
        )
        price = close
    rows.append(
        {
            "asset_id": 1,
            "date": 20240102,
            "session_minute": 390,
            "open": 99.0,
            "close": 50.0,
        }
    )
    out = realized_measures_from_minute_ohlcv(pd.DataFrame(rows), min_intraday_returns=78)
    assert len(out) == 1
    assert out.iloc[0]["n_intraday_returns"] == 78
    assert out.iloc[0]["rv_plus"] == pytest.approx(np.log(101.0 / 100.0) ** 2)
    assert out.iloc[0]["rv_minus"] == pytest.approx(np.log(99.0 / 101.0) ** 2)


def test_missing_five_minute_intervals_are_previous_tick_filled():
    bars = pd.DataFrame(
        {
            "asset_id": [1, 1, 1],
            "date": [20240102] * 3,
            "session_minute": [0, 10, 390],
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 999.0],
        }
    )
    out = realized_measures_from_minute_ohlcv(bars, min_intraday_returns=1)
    assert out.iloc[0]["n_intraday_returns"] == 78
    assert out.iloc[0]["n_observed_minutes"] == 2
    expected = np.log(101 / 100) ** 2 + np.log(102 / 101) ** 2
    assert out.iloc[0]["rv_plus"] == pytest.approx(expected)


def test_auction_only_stock_day_is_ignored_without_key_error():
    bars = pd.DataFrame(
        {
            "asset_id": [1, 2, 2],
            "date": [20240102] * 3,
            "session_minute": [390, 0, 390],
            "open": [10.0, 20.0, 21.0],
            "close": [11.0, 20.5, 999.0],
        }
    )
    out = realized_measures_from_minute_ohlcv(bars, min_intraday_returns=1)
    assert set(out["asset_id"]) == {2}


def test_daily_realized_measures_match_definition_and_exclude_overnight():
    times = pd.to_datetime(
        [
            "2024-01-08 09:30",
            "2024-01-08 09:35",
            "2024-01-08 09:40",
            "2024-01-09 09:30",
            "2024-01-09 09:35",
        ]
    )
    bars = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "A",
            "price": [100.0, 100.0 * np.exp(0.1), 100.0 * np.exp(0.1) * np.exp(-0.2), 500.0, 500.0 * np.exp(0.3)],
        }
    )
    out = realized_measures_from_bars(bars, min_intraday_returns=1)
    d1 = out.iloc[0]
    assert d1["rv_plus"] == pytest.approx(0.1**2)
    assert d1["rv_minus"] == pytest.approx(0.2**2)
    assert d1["rsj"] == pytest.approx((0.1**2 - 0.2**2) / (0.1**2 + 0.2**2))
    # The 500/previous-day-close move is not included; only the within-day 0.3 return is.
    assert out.iloc[1]["rv"] == pytest.approx(0.3**2)


def test_weekly_signal_is_mean_of_daily_rsj_and_uses_only_history():
    dates = pd.bdate_range("2024-01-03", periods=6)  # sixth observation is Wednesday
    daily = pd.DataFrame(
        {
            "date": dates,
            "asset_id": "A",
            "rv": [1.0, 1.0, 1.0, 1.0, 100.0, 1.0],
            "rsj": [-1.0, -0.5, 0.0, 0.5, 1.0, -1.0],
            "rsk": 0.0,
            "rkt": 3.0,
        }
    )
    out = make_weekly_rsj_panel(daily)
    assert len(out) == 1
    assert out.iloc[0]["rebalance_date"] == pd.Timestamp("2024-01-09")
    # Mean daily RSJ, not ratio/sum weighted by the very large fifth-day RV.
    assert out.iloc[0]["signal_value"] == pytest.approx(0.0)
    assert out.iloc[0]["hold_start"] == pd.Timestamp("2024-01-10")
    assert out.iloc[0]["hold_end"] == pd.Timestamp("2024-01-16")


def test_weekly_signal_does_not_bridge_a_missing_market_date():
    market_dates = pd.bdate_range("2024-01-02", periods=7)
    daily = pd.DataFrame(
        {
            "date": list(market_dates) + list(market_dates.delete(2)),
            "asset_id": ["COMPLETE"] * 7 + ["GAP"] * 6,
            "rv": 1.0,
            "rsj": 0.1,
            "rsk": 0.0,
            "rkt": 3.0,
        }
    )
    out = make_weekly_rsj_panel(daily)
    assert "COMPLETE" in set(out["asset_id"])
    assert "GAP" not in set(out["asset_id"])


def test_low_rsj_is_long_and_high_rsj_is_short():
    assets = [f"A{i}" for i in range(10)]
    panel = pd.DataFrame(
        {
            "rebalance_date": pd.Timestamp("2024-01-09"),
            "hold_start": pd.Timestamp("2024-01-10"),
            "hold_end": pd.Timestamp("2024-01-16"),
            "asset_id": assets,
            "signal_value": np.arange(10, dtype=float),
        }
    )
    factor, spread = build_rsj_factor(panel, weighting="equal", n_groups=5)
    assert factor.metadata["direction"] == "long low RSJ, short high RSJ"
    long_assets = set(spread.loc[spread["target_weight"].gt(0), "asset_id"])
    short_assets = set(spread.loc[spread["target_weight"].lt(0), "asset_id"])
    assert long_assets == {"A0", "A1"}
    assert short_assets == {"A8", "A9"}
    assert spread.loc[spread["target_weight"].gt(0), "target_weight"].sum() == pytest.approx(1.0)
    assert spread.loc[spread["target_weight"].lt(0), "target_weight"].sum() == pytest.approx(-1.0)

