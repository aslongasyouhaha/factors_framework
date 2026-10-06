import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorlab import (  # noqa: E402
    FactorBacktester,
    fama_macbeth,
    ic_decay,
    ic_summary,
    information_coefficient,
    newey_west,
    newey_west_lags,
    performance_by_year,
)

sm = pytest.importorskip("statsmodels.api")


def ar1(n=240, phi=0.4, mu=0.01, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.05, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x + mu, index=pd.date_range("2000-01-31", periods=n, freq="ME"))


@pytest.mark.parametrize("lags", [0, 3, 12, None])
def test_newey_west_matches_statsmodels_hac(lags):
    x = ar1()
    out = newey_west(x, lags)
    used = newey_west_lags(len(x)) if lags is None else lags
    ref = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": used, "use_correction": False})
    assert out["lags"] == used
    assert out["se"] == pytest.approx(ref.bse[0], rel=1e-10)
    assert out["t"] == pytest.approx(ref.tvalues[0], rel=1e-10)


def test_newey_west_lag_rule():
    assert newey_west_lags(100) == 4 and newey_west_lags(102) == 4 and newey_west_lags(6500) == 10


def test_ols_hac_matches_statsmodels():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y = pd.Series(0.1 + x @ [0.5, -0.2] + ar1(300, seed=2).to_numpy())
    out = FactorBacktester.ols(y, x, hac_lags=5)
    ref = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    np.testing.assert_allclose([out["alpha_t"], out["a_t"], out["b_t"]], ref.tvalues, rtol=1e-9)
    classic = FactorBacktester.ols(y, x)
    np.testing.assert_allclose([classic["alpha_t"], classic["a_t"], classic["b_t"]], sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit().tvalues, rtol=1e-9)


def panel(t=36, n=200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=t, freq="ME", name="date")
    size = pd.DataFrame(rng.normal(size=(t, n)), index=idx)
    value = pd.DataFrame(rng.normal(size=(t, n)) * 1e3, index=idx)  # different scale on purpose
    industry = pd.DataFrame(np.broadcast_to(rng.integers(0, 5, n), (t, n)), index=idx)
    ret = 0.002 - 0.01 * size + 2e-6 * value + 0.003 * industry + rng.normal(0, 0.05, (t, n))
    ret = ret.mask(rng.random((t, n)) < 0.05)
    return ret, size, value, industry


def test_fama_macbeth_matches_per_date_ols():
    ret, size, value, _ = panel()
    res = fama_macbeth(ret, {"size": size, "value": value}, min_obs=10, lags=3)
    for date in ret.index[[0, 17, 35]]:
        d = pd.DataFrame({"y": ret.loc[date], "size": size.loc[date], "value": value.loc[date]}).dropna()
        ref = sm.OLS(d["y"], sm.add_constant(d[["size", "value"]])).fit()
        np.testing.assert_allclose(res.coefficients.loc[date, ["const", "size", "value"]], ref.params, rtol=1e-8, atol=1e-12)
        assert res.r2.loc[date] == pytest.approx(ref.rsquared, rel=1e-9)
        assert res.n_assets.loc[date] == len(d)
    for term in ["const", "size", "value"]:
        assert res.summary.loc[term, "t"] == pytest.approx(newey_west(res.coefficients[term], 3)["t"], rel=1e-12)
    assert res.summary.loc["size", "t"] < -5


def test_fama_macbeth_industry_effects_match_dummies():
    ret, size, _, industry = panel()
    res = fama_macbeth(ret, {"size": size}, categories=industry, min_obs=10)
    assert "const" not in res.coefficients.columns
    date = ret.index[5]
    d = pd.DataFrame({"y": ret.loc[date], "size": size.loc[date], "ind": industry.loc[date]}).dropna()
    x = pd.concat([d[["size"]], pd.get_dummies(d["ind"], prefix="i", drop_first=True).astype(float)], axis=1)
    ref = sm.OLS(d["y"], sm.add_constant(x)).fit()
    assert res.coefficients.loc[date, "size"] == pytest.approx(ref.params["size"], rel=1e-8)
    assert res.r2.loc[date] == pytest.approx(ref.rsquared, rel=1e-8)


def test_ic_decay_lags_shift_the_return_period():
    ret, size, _, _ = panel()
    decay = ic_decay(size, ret, max_lag=3, lags=2)
    lag1 = ic_summary(information_coefficient(size, ret), lags=2)
    lag3 = ic_summary(information_coefficient(size, ret.shift(-2)), lags=2)
    assert decay.loc[1, "mean_rank_ic"] == pytest.approx(lag1.loc["rank_ic", "mean"])
    assert decay.loc[3, "rank_ic_t_nw"] == pytest.approx(lag3.loc["rank_ic", "t_nw"])
    assert decay.loc[1, "mean_rank_ic"] < -0.05 and abs(decay.loc[2, "mean_rank_ic"]) < 0.05
    assert decay.attrs["half_life"] == 2


def test_performance_by_year_compounds_within_year():
    r = ar1(36)
    out = performance_by_year(r, periods_per_year=12)
    assert list(out.index) == [2000, 2001, 2002]
    for year in out.index:
        assert out.loc[year, "total_return"] == pytest.approx((1 + r[r.index.year == year]).prod() - 1)
    wide = performance_by_year(pd.DataFrame({"A": r, "B": -r}), periods_per_year=12)
    assert wide.index.names == ["portfolio", "year"] and len(wide) == 6


def test_summary_reports_newey_west_t():
    r = ar1()
    s = FactorBacktester(periods_per_year=12).summary(r, nw_lags=6)
    assert s["t_nw"] == pytest.approx(newey_west(r, 6)["t"])
    assert s["sharpe"] == pytest.approx(np.sqrt(12) * r.mean() / r.std())
