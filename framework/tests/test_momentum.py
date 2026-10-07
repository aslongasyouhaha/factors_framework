import numpy as np
import pandas as pd

from factorlab.momentum import cross_sectional_zscore, rolling_compound


def test_rolling_compound_respects_skip_and_asset_boundaries():
    ret = pd.Series([0.10, 0.20, 0.30, 0.40, 0.50, 0.10, 0.20, 0.30])
    asset = pd.Series([1, 1, 1, 1, 1, 2, 2, 2])
    got = rolling_compound(ret, asset, skip=1, window=2, min_periods=2)
    assert np.isnan(got.iloc[0]) and np.isnan(got.iloc[1])
    assert np.isclose(got.iloc[2], 1.10 * 1.20 - 1.0)
    assert np.isclose(got.iloc[4], 1.30 * 1.40 - 1.0)
    assert np.isnan(got.iloc[5]) and np.isnan(got.iloc[6])
    assert np.isclose(got.iloc[7], 1.10 * 1.20 - 1.0)


def test_cross_sectional_zscore_is_date_local():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 20 + [pd.Timestamp("2024-02-29")] * 20,
            "exposure": list(range(20)) + list(range(100, 120)),
        }
    )
    z = cross_sectional_zscore(frame)
    by_date = z.groupby(frame["date"])
    assert np.allclose(by_date.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(by_date.std().to_numpy(), 1.0, atol=1e-12)
