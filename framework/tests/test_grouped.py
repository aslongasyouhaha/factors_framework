import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorlab import FactorBuildTools, NeutralizationRule, OutlierRule, fill_missing, neutralize_rows  # noqa: E402
from factorlab.grouped import group_bounds, group_codes, group_median, group_quantiles, residualize  # noqa: E402


def grouped_sample(seed=0, n=5000, groups=40):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, groups, size=n)
    values = rng.standard_t(3, size=n)
    values[rng.random(n) < 0.1] = np.nan
    values[: n // 10] = np.round(values[: n // 10], 1)  # ties
    return values, codes, groups


@pytest.mark.parametrize("qs", [(0.01, 0.99), (0.25, 0.75), (0.3, 0.7), (0.0, 1.0)])
def test_group_quantiles_bit_identical_to_pandas(qs):
    values, codes, groups = grouped_sample()
    got = group_quantiles(values, codes, groups, qs)
    s = pd.Series(values)
    for g in range(groups):
        expected = s[codes == g].quantile(list(qs)).to_numpy()
        assert (got[:, g] == expected).all()


def test_group_median_bit_identical_to_pandas():
    values, codes, groups = grouped_sample(1)
    got = group_median(values, codes, groups)
    s = pd.Series(values)
    for g in range(groups):
        assert got[g] == s[codes == g].median()


@pytest.mark.parametrize("method", ["quantile", "std", "mad", "iqr"])
def test_group_bounds_match_series_outlier_bounds(method):
    values, codes, groups = grouped_sample(2)
    rule = OutlierRule(cols=("x",), method=method)
    lo, hi = group_bounds(values, codes, groups, method, rule.lower, rule.upper, rule.n_std, rule.n_mad, rule.iqr_multiplier)
    s = pd.Series(values)
    for g in range(groups):
        elo, ehi = FactorBuildTools.outlier_bounds(s[codes == g], rule)
        assert lo[g] == pytest.approx(elo, rel=1e-14, abs=1e-14)
        assert hi[g] == pytest.approx(ehi, rel=1e-14, abs=1e-14)


def test_group_codes_missing_keys():
    codes, n = group_codes(["b", None, "a", "b"], [1, 1, 1, 2])
    assert n == 3 and codes[1] == -1 and codes[0] != codes[3]


def _lstsq_residuals(y, x, cats):
    cols = [np.ones(len(y))] + ([x] if x is not None else [])
    for c in cats:
        d = pd.get_dummies(pd.Series(c).astype(str)).iloc[:, 1:].to_numpy(dtype=float)
        cols.append(d)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


@pytest.mark.parametrize("n_cats", [0, 1, 2])
def test_residualize_matches_dummy_regression(n_cats):
    rng = np.random.default_rng(3)
    n, groups = 3000, 6
    codes = rng.integers(0, groups, size=n)
    x = rng.normal(size=(n, 2)) * [1.0, 1e6]  # very different scales
    cats = [rng.integers(0, 5, size=n), rng.integers(0, 3, size=n)][:n_cats]
    y = x @ [0.5, 2e-6] + rng.normal(size=n) + sum(c * 0.3 for c in cats)
    got = residualize(y, x, codes, groups, category_codes=cats, min_obs=5)
    for g in range(groups):
        m = codes == g
        expected = _lstsq_residuals(y[m], x[m], [c[m] for c in cats])
        np.testing.assert_allclose(got[m], expected, atol=1e-9)


def test_residualize_min_obs_and_missing():
    codes = np.array([0] * 5 + [1] * 30)
    y = np.arange(35, dtype=float)
    y[7] = np.nan
    got = residualize(y, None, codes, 2, min_obs=10)
    assert np.isnan(got[:5]).all() and np.isnan(got[7])
    assert np.nansum(got[5:]) == pytest.approx(0.0, abs=1e-12)


def test_long_neutralize_uses_group_keys():
    rng = np.random.default_rng(4)
    dates = np.repeat(pd.date_range("2020-01-31", periods=3, freq="ME"), 50)
    panel = pd.DataFrame(
        {"date": dates, "asset_id": np.tile(np.arange(50), 3), "y": rng.normal(size=150), "x": rng.normal(size=150)}
    )
    out = FactorBuildTools.neutralize(FactorBuildTools.to_panel_index(panel), NeutralizationRule(y_col="y", x_cols=("x",)))
    for _, g in out.groupby(level="date"):
        np.testing.assert_allclose(g["y_neutral"], _lstsq_residuals(g["y"].to_numpy(), g["x"].to_numpy(), []), atol=1e-12)


def test_neutralize_rows_matches_long():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame(rng.normal(size=(4, 60)), index=idx)
    x = pd.DataFrame(rng.normal(size=(4, 60)), index=idx)
    cat = pd.DataFrame(rng.integers(0, 4, size=(4, 60)), index=idx)
    y.iloc[1, 3] = np.nan
    wide = neutralize_rows(y, {"x": x}, categories=cat)
    for t in range(4):
        m = y.iloc[t].notna().to_numpy()
        expected = _lstsq_residuals(y.iloc[t].to_numpy()[m], x.iloc[t].to_numpy()[m], [cat.iloc[t].to_numpy()[m]])
        np.testing.assert_allclose(wide.iloc[t].to_numpy()[m], expected, atol=1e-12)


@pytest.mark.parametrize("limit", [None, 1, 3])
def test_fill_missing_ffill_counts_universe_rows(limit):
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    x = pd.DataFrame(rng.normal(size=(30, 8)), index=idx)
    x[rng.random(x.shape) < 0.4] = np.nan
    universe = pd.DataFrame(rng.random(x.shape) > 0.2, index=idx)
    got = fill_missing(x, "ffill", universe=universe, limit=limit)
    long = x.where(universe).stack(dropna=False)[universe.stack()]
    expected = long.groupby(level=1).ffill(limit=limit)
    got_long = got.stack(dropna=False)[universe.stack()]
    pd.testing.assert_series_equal(got_long, expected, check_names=False)
    assert got.where(~universe).isna().all().all()
