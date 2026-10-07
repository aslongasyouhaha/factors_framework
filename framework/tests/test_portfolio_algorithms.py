import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorlab.bayesian_sdf import continuous_spike_slab_sdf, psi_from_prior_sharpe
from factorlab.risk_parity import _symmetric_power, agnostic_risk_parity_backtest


def test_symmetric_inverse_sqrt_whitens_matrix():
    matrix = np.array([[2.0, 0.4], [0.4, 1.0]])
    inverse_half = _symmetric_power(matrix, -0.5)
    np.testing.assert_allclose(inverse_half @ matrix @ inverse_half, np.eye(2), atol=1e-10)


def test_arp_backtest_outputs_all_comparators_without_lookahead_rows():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-31", periods=90, freq="M")
    returns = pd.DataFrame(rng.normal(0.004, 0.04, (90, 5)), index=dates)
    result = agnostic_risk_parity_backtest(returns, min_window=36, covariance_window=48)
    assert list(result.returns.columns) == ["ARP", "equal_risk_signal", "markowitz_clean", "markowitz_raw"]
    assert len(result.returns) == 54
    assert result.returns.index.min() == dates[36]
    assert set(result.eigenrisk["strategy"]) == set(result.returns.columns)
    np.testing.assert_allclose(
        result.eigenrisk.groupby(["date", "strategy"])["risk_share"].sum().to_numpy(),
        1.0,
    )


def test_bayesian_sdf_sampler_shapes_and_probabilities():
    rng = np.random.default_rng(11)
    dates = pd.date_range("1990-01-31", periods=90, freq="M")
    factors = pd.DataFrame(rng.normal(size=(90, 2)), index=dates, columns=["strong", "noise"])
    assets = pd.DataFrame(
        np.outer(factors["strong"], np.linspace(0.2, 1.0, 6)) + rng.normal(0, 0.5, (90, 6)),
        index=dates,
    )
    psi = psi_from_prior_sharpe(factors, assets, prior_sharpe=0.05)
    result = continuous_spike_slab_sdf(
        factors, assets, simulations=40, burn=10, psi0=psi, seed=9
    )
    assert result.gamma_path.shape == (40, 2)
    assert result.bma_sdf.shape == (90,)
    assert result.inclusion_probability.between(0, 1).all()
    assert np.isfinite(result.lambda_quantiles.to_numpy()).all()
