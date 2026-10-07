import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_extended_daily_characteristics import _corr_numpy  # noqa: E402


def test_bounded_memory_rolling_corr_matches_pandas():
    x = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0, 8.0])
    y = pd.Series([2.0, 1.0, 3.0, 5.0, 7.0, 9.0])
    expected = x.rolling(4, min_periods=3).corr(y).to_numpy()
    got = _corr_numpy(x.to_numpy(), y.to_numpy(), 4, 3)
    np.testing.assert_allclose(got, expected, equal_nan=True)
