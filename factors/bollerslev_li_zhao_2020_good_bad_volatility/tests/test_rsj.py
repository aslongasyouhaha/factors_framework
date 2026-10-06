import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "framework"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsj import build_rsj_factor, make_weekly_rsj_panel, realized_measures_from_bars  # noqa: E402


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

