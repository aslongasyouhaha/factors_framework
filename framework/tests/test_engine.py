import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorlab import (  # noqa: E402
    FactorBacktester,
    LayerBook,
    WeightBook,
    forward_returns,
    holding_segments,
    quantile_codes,
    simulate,
    winsorize_rows,
)


def make_returns(t=60, n=40, seed=0, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.02, size=(t, n))
    r[rng.random((t, n)) < nan_frac] = np.nan
    idx = pd.bdate_range("2020-01-01", periods=t, name="date")
    return pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(n)])


def test_segments_follow_close_to_close_timing():
    idx = pd.bdate_range("2020-01-01", periods=10)
    reb = pd.DatetimeIndex([idx[0], idx[4], idx[7]])
    segs = holding_segments(idx, reb)
    assert segs.tolist() == [[1, 5], [5, 8], [8, 10]]
    # A rebalance on a non-trading day maps to the next row after it.
    segs = holding_segments(idx, pd.DatetimeIndex([idx[2] + pd.Timedelta(hours=12)]))
    assert segs.tolist() == [[3, 10]]


@pytest.mark.parametrize("n_groups", [3, 5, 10])
def test_quantile_codes_match_qcut(n_groups):
    rng = np.random.default_rng(1)
    sizes = list(range(n_groups, 400)) + [2711, 3619]
    width = max(sizes)
    x = np.full((len(sizes), width), np.nan)
    for i, size in enumerate(sizes):
        x[i, :size] = rng.normal(size=size)
        x[i, : size // 7] = np.round(x[i, : size // 7], 1)  # ties
    wide = pd.DataFrame(x)
    codes = quantile_codes(wide, n_groups)
    for i in range(len(sizes)):
        s = wide.iloc[i].dropna()
        expected = pd.qcut(s.rank(method="first"), n_groups, labels=False).to_numpy()
        assert (codes[i, s.index] == expected).all(), sizes[i]
        assert (codes[i, ~wide.iloc[i].notna()] == -1).all()


def test_quantile_codes_thin_rows_are_unassigned():
    wide = pd.DataFrame([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]])
    assert (quantile_codes(wide, 3)[0] == -1).all()
    assert quantile_codes(wide, 3)[1].tolist() == [0, 1, 2]


def test_winsorize_rows_matches_pandas_quantile():
    r = make_returns(t=20, n=50)
    out = winsorize_rows(r, 0.05, 0.95)
    for date, row in r.iterrows():
        s = row.dropna()
        expected = s.clip(s.quantile(0.05), s.quantile(0.95))
        pd.testing.assert_series_equal(out.loc[date, s.index], expected, check_names=False)


def test_fixed_weights_equal_row_dot_product():
    r = make_returns()
    reb = r.index[[0, 20, 40]]
    w = pd.DataFrame(np.random.default_rng(2).normal(size=(3, r.shape[1])), index=reb, columns=r.columns)
    sim = simulate(WeightBook(w), r)
    for t in range(1, len(r)):
        k = np.searchsorted(reb, r.index[t]) - 1
        expected = np.nansum(w.iloc[k].to_numpy() * r.iloc[t].to_numpy())
        assert sim.returns.iloc[t - 1, 0] == pytest.approx(expected, abs=1e-14)


def test_drift_matches_share_by_share_buy_and_hold():
    r = make_returns(nan_frac=0.0)
    reb = r.index[[0, 25]]
    rng = np.random.default_rng(3)
    long_only = rng.random((2, r.shape[1]))
    long_only /= long_only.sum(axis=1, keepdims=True)
    w = pd.DataFrame(long_only, index=reb, columns=r.columns)
    sim = simulate(WeightBook(w), r, drift=True, cost_bps=10.0)

    nav, values, prev_w, expected = 1.0, None, np.zeros(r.shape[1]), []
    for t in range(1, len(r)):
        if r.index[t - 1] in reb:
            target = w.loc[r.index[t - 1]].to_numpy()
            cost = np.abs(target - prev_w).sum() * 10.0 / 10_000
            values = nav * target
        else:
            cost = 0.0
        new_values = values * (1 + r.iloc[t].to_numpy())
        ret = new_values.sum() / values.sum() - 1 - cost
        expected.append(ret)
        nav, values = new_values.sum(), new_values
        prev_w = values / values.sum()
    np.testing.assert_allclose(sim.returns.iloc[:, 0].to_numpy(), expected, atol=1e-13)


def test_layer_book_layers_and_spread():
    r = make_returns(n=30)
    reb = r.index[[0, 30]]
    rng = np.random.default_rng(4)
    signal = pd.DataFrame(rng.normal(size=(2, 30)), index=reb, columns=r.columns)
    codes = quantile_codes(signal, 3)
    base = rng.random((2, 30))
    book = LayerBook(reb, r.columns, codes, base, ["L1", "L2", "L3"], spreads={"S": (("L3",), ("L1",))})
    w = book.weights_at(0)
    np.testing.assert_allclose(w[:, :3].sum(axis=0), 1.0)
    np.testing.assert_allclose(w[:, 3], w[:, 2] - w[:, 0])
    sim = simulate(book, r)
    np.testing.assert_allclose(sim.returns["S"], sim.returns["L3"] - sim.returns["L1"], atol=1e-15)


def test_turnover_and_long_exports():
    r = make_returns(n=30)
    reb = r.index[[0, 20, 40]]
    rng = np.random.default_rng(7)
    signal = pd.DataFrame(rng.normal(size=(3, 30)), index=reb, columns=r.columns)
    book = LayerBook(reb, r.columns, quantile_codes(signal, 3), rng.random((3, 30)), ["L1", "L2", "L3"], spreads={"S": (("L3",), ("L1",))})
    bt = FactorBacktester()
    turn = bt.compute_turnover(book, "S")
    w = np.stack([book.weights_at(k)[:, 3] for k in range(3)])
    np.testing.assert_allclose(turn["turnover"], 0.5 * np.abs(np.diff(np.vstack([np.zeros(30), w]), axis=0)).sum(axis=1))
    np.testing.assert_allclose(turn["gross"], 2.0)
    with pytest.raises(ValueError):
        bt.compute_turnover(book)  # several portfolios: must name one

    holdings = book.to_long(extras={"signal": signal})
    assert set(holdings["portfolio"]) == {"L1", "L2", "L3", "S"}
    row = holdings.iloc[0]
    assert row["signal"] == signal.loc[row["rebalance_date"], row["asset_id"]]

    sim = simulate(book, r)
    long = sim.to_long(["L1", "S"])
    assert set(long["portfolio"]) == {"L1", "S"} and (long["n_assets"] > 0).all()
    np.testing.assert_allclose(long.set_index(["date", "portfolio"])["ret"].unstack()["S"], sim.returns["S"].loc[long["date"].unique()])


def test_run_factor_and_wide_only_inputs():
    from factorlab import FactorConfig, QuantileSignalFactorBuilder

    r = make_returns(t=80, n=60)
    reb = r.index[::20]
    signal = pd.DataFrame(np.random.default_rng(8).normal(size=(len(reb), 60)), index=reb, columns=r.columns)
    cfg = FactorConfig(name="LS", n_groups=5, weighting="equal", long_groups=("D05",), short_groups=("D01",))
    wf = QuantileSignalFactorBuilder(config=cfg, winsorize=False).build_wide(signal)
    res = FactorBacktester().run_factor(wf, r, make_plots=False)
    assert list(res.returns.columns) == ["LS"] and res.ic is not None and res.by_year is not None
    expected_ic = FactorBacktester().ic_wide(wf.signal, forward_returns(r, reb))["rank_ic"]
    np.testing.assert_allclose(res.ic["rank_ic"], expected_ic)
    with pytest.raises(TypeError):
        FactorBacktester().simulate(wf.book, r.reset_index())  # long data must go through to_wide()


def test_forward_returns_compound_each_holding_period():
    r = make_returns(nan_frac=0.0)
    reb = r.index[[0, 10]]
    fwd = forward_returns(r, reb)
    expected0 = (1 + r.iloc[1:11]).prod() - 1
    expected1 = (1 + r.iloc[11:]).prod() - 1
    np.testing.assert_allclose(fwd.iloc[0], expected0, atol=1e-14)
    np.testing.assert_allclose(fwd.iloc[1], expected1, atol=1e-14)


def test_ic_wide_matches_pandas():
    bt = FactorBacktester()
    s = make_returns(t=15, n=60, seed=5)
    f = make_returns(t=15, n=60, seed=6)
    out = bt.ic_wide(s, f)
    for date in s.index:
        both = pd.concat([s.loc[date], f.loc[date]], axis=1).dropna()
        assert out.loc[date, "ic"] == pytest.approx(both.iloc[:, 0].corr(both.iloc[:, 1]), abs=1e-12)
        assert out.loc[date, "rank_ic"] == pytest.approx(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"), abs=1e-12)
    r2 = bt.r2_wide(s, f)
    np.testing.assert_allclose(r2["r2"], out["ic"] ** 2, atol=1e-12)
