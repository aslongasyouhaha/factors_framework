"""Bayesian SDF selection following Bryzgalova, Huang and Julliard (2023)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import invwishart, invgamma


@dataclass
class BayesianSDFResult:
    inclusion_probability: pd.Series
    lambda_mean: pd.Series
    lambda_quantiles: pd.DataFrame
    model_size: pd.Series
    bma_sdf: pd.Series
    gamma_path: np.ndarray


def psi_from_prior_sharpe(
    factors: pd.DataFrame,
    test_assets: pd.DataFrame,
    prior_sharpe: float = 0.10,
    aw: float = 1.0,
    bw: float = 1.0,
) -> float:
    """Map an economically interpretable prior Sharpe ratio to psi (eq. 27)."""
    joined = pd.concat({"f": factors, "r": test_assets}, axis=1).dropna()
    f = joined["f"].to_numpy(float)
    r = joined["r"].to_numpy(float)
    mean_r = r.mean(axis=0)
    max_sr2 = float(mean_r @ np.linalg.pinv(np.cov(r, rowvar=False, ddof=1)) @ mean_r)
    corr_rf = np.corrcoef(r, f, rowvar=False)[: r.shape[1], r.shape[1] :]
    demeaned = corr_rf - corr_rf.mean(axis=0, keepdims=True)
    eta = (aw / (aw + bw)) * float(np.sum(demeaned**2)) / r.shape[1]
    denominator = (max_sr2 - prior_sharpe**2) * eta
    if denominator <= 0:
        raise ValueError("prior Sharpe must be below the test assets' maximum Sharpe ratio")
    return prior_sharpe**2 / denominator


def continuous_spike_slab_sdf(
    factors: pd.DataFrame,
    test_assets: pd.DataFrame,
    simulations: int = 3000,
    burn: int = 1000,
    psi0: float = 1.0,
    spike_ratio: float = 0.001,
    aw: float = 1.0,
    bw: float = 1.0,
    seed: int = 202310,
) -> BayesianSDFResult:
    """Continuous spike-and-slab BMA-SDF, OLS version with an intercept.

    This follows equations (28)--(31) and the official ``continuous_ss_sdf``
    implementation. All inputs must be monthly returns on the same dates.
    """
    joined = pd.concat({"f": factors, "r": test_assets}, axis=1).dropna()
    f = joined["f"].to_numpy(float)
    r_assets = joined["r"].to_numpy(float)
    dates = joined.index
    t, k = f.shape
    n = r_assets.shape[1]
    if k >= n:
        raise ValueError("The number of test assets must exceed the number of factors")
    y = np.column_stack([f, r_assets])
    p = y.shape[1]
    covariance = np.cov(y, rowvar=False, ddof=1)
    mean = y.mean(axis=0)
    sd = y.std(axis=0, ddof=1)
    corr = covariance / np.outer(sd, sd)
    beta0 = np.column_stack([np.ones(n), corr[k:, :k]])
    a0 = mean[k:] / sd[k:]
    initial_lambda = np.linalg.pinv(beta0.T @ beta0) @ beta0.T @ a0
    sigma2 = max(float(np.mean((a0 - beta0 @ initial_lambda) ** 2)), 1e-8)

    rho = corr[k:, :k]
    rho_demean = rho - rho.mean(axis=0, keepdims=True)
    psi = psi0 * np.sum(rho_demean**2, axis=0)
    psi = np.maximum(psi, 1e-10)
    rng = np.random.default_rng(seed)
    omega = np.full(k, 0.5)
    gamma = rng.binomial(1, omega)
    gamma_path = np.empty((simulations, k), dtype=np.int8)
    lambda_path = np.empty((simulations, k + 1))
    sdf_sum = np.zeros(t)

    scale = t * covariance
    for draw in range(simulations):
        sigma_y = invwishart.rvs(df=t - 1, scale=scale, random_state=rng)
        sigma_y = np.atleast_2d(sigma_y)
        mu = mean + rng.multivariate_normal(np.zeros(p), sigma_y / t)
        sd_y = np.sqrt(np.maximum(np.diag(sigma_y), 1e-16))
        corr_y = sigma_y / np.outer(sd_y, sd_y)
        beta = np.column_stack([np.ones(n), corr_y[k:, :k]])
        a = mu[k:] / sd_y[k:]

        r_gamma = np.where(gamma == 1, 1.0, spike_ratio)
        dmat = np.diag(np.r_[1.0 / 100000.0, 1.0 / (r_gamma * psi)])
        precision = beta.T @ beta + dmat
        covariance_lambda = sigma2 * np.linalg.pinv(precision)
        lambda_hat = np.linalg.pinv(precision) @ beta.T @ a
        lam = rng.multivariate_normal(lambda_hat, covariance_lambda)

        log_odds = (
            np.log(np.clip(omega, 1e-12, 1) / np.clip(1 - omega, 1e-12, 1))
            + 0.5 * np.log(spike_ratio)
            + 0.5 * lam[1:] ** 2 * (1.0 / spike_ratio - 1.0) / (sigma2 * psi)
        )
        gamma = rng.binomial(1, expit(np.minimum(log_odds, np.log(1000.0))))
        omega = rng.beta(aw + gamma, bw + 1 - gamma)
        residual = a - beta @ lam
        shape = (n + k + 1) / 2.0
        inv_scale = (residual @ residual + lam @ dmat @ lam) / 2.0
        sigma2 = float(invgamma.rvs(a=shape, scale=max(inv_scale, 1e-16), random_state=rng))

        gamma_path[draw] = gamma
        lambda_path[draw] = lam
        if draw >= burn:
            lambda_f = lam[1:] / f.std(axis=0, ddof=1)
            sdf = 1.0 - f @ lambda_f
            sdf_sum += sdf + (1.0 - sdf.mean())

    if burn >= simulations:
        raise ValueError("burn must be smaller than simulations")
    kept_gamma = gamma_path[burn:]
    kept_lambda = lambda_path[burn:, 1:]
    columns = factors.columns
    quantiles = pd.DataFrame(
        np.quantile(kept_lambda, [0.05, 0.50, 0.95], axis=0).T,
        index=columns,
        columns=["p05", "p50", "p95"],
    )
    sizes, counts = np.unique(kept_gamma.sum(axis=1), return_counts=True)
    return BayesianSDFResult(
        inclusion_probability=pd.Series(kept_gamma.mean(axis=0), index=columns, name="inclusion_probability"),
        lambda_mean=pd.Series(kept_lambda.mean(axis=0), index=columns, name="lambda_mean"),
        lambda_quantiles=quantiles,
        model_size=pd.Series(counts / counts.sum(), index=sizes, name="probability"),
        bma_sdf=pd.Series(sdf_sum / (simulations - burn), index=dates, name="BMA_SDF"),
        gamma_path=gamma_path,
    )
