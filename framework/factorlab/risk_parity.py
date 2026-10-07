"""Agnostic Risk Parity and comparable signal-weighting rules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _symmetric_power(matrix: np.ndarray, power: float, floor: float = 1e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors * (values**power)) @ vectors.T


def _scale_to_risk(weights: np.ndarray, covariance: np.ndarray, target_monthly_vol: float) -> np.ndarray:
    risk = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    return weights * (target_monthly_vol / risk) if risk > 1e-12 else np.zeros_like(weights)


@dataclass
class ARPResult:
    returns: pd.DataFrame
    gross_returns: pd.DataFrame
    turnover: pd.DataFrame
    weights: pd.DataFrame
    eigenrisk: pd.DataFrame


def agnostic_risk_parity_backtest(
    returns: pd.DataFrame,
    covariance_window: int = 60,
    signal_window: int = 12,
    min_window: int = 36,
    target_vol: float = 0.10,
    cost_bps: float = 10.0,
) -> ARPResult:
    """Compare ARP with equal-risk signal and Markowitz allocations.

    Information through month t-1 sets positions for month t. Factor returns are
    volatility-normalized, and the trend indicator is the flat moving average of
    the previous ``signal_window`` normalized returns. Ledoit-Wolf covariance
    cleaning is used as a transparent finite-sample substitute for the paper's RIE.
    """
    x = returns.sort_index().replace([np.inf, -np.inf], np.nan)
    common = x.dropna()
    if len(common) < min_window + 2:
        raise ValueError("Not enough complete monthly observations for ARP")
    x = common
    names = list(x.columns)
    strategies = ["equal_risk_signal", "markowitz_raw", "markowitz_clean", "ARP"]
    weights, gross, costs, eigenrows = [], [], [], []
    previous = {s: np.zeros(len(names)) for s in strategies}
    target_monthly = target_vol / np.sqrt(12.0)

    for pos in range(min_window, len(x)):
        hist = x.iloc[max(0, pos - covariance_window):pos]
        sigma = hist.std(ddof=1).to_numpy()
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, np.nan)
        if np.isnan(sigma).any():
            continue
        z = hist.to_numpy() / sigma
        raw_corr = np.cov(z, rowvar=False, ddof=1)
        clean_corr = LedoitWolf(assume_centered=False).fit(z).covariance_
        signal = z[-signal_window:].mean(axis=0)
        raw_rules = {
            "equal_risk_signal": signal,
            "markowitz_raw": np.linalg.pinv(raw_corr, rcond=1e-6) @ signal,
            "markowitz_clean": np.linalg.pinv(clean_corr, rcond=1e-6) @ signal,
            "ARP": _symmetric_power(clean_corr, -0.5) @ signal,
        }
        raw_cov = np.diag(sigma) @ clean_corr @ np.diag(sigma)
        date = x.index[pos]
        realized = x.iloc[pos].to_numpy()
        for strategy, normalized_weight in raw_rules.items():
            physical = normalized_weight / sigma
            physical = _scale_to_risk(physical, raw_cov, target_monthly)
            turn = float(np.abs(physical - previous[strategy]).sum())
            pnl = float(physical @ realized)
            fee = turn * cost_bps / 10000.0
            gross.append((date, strategy, pnl))
            costs.append((date, strategy, turn, fee))
            weights.extend((date, strategy, n, w) for n, w in zip(names, physical))
            previous[strategy] = physical

        eigval, eigvec = np.linalg.eigh(clean_corr)
        for strategy, normalized_weight in raw_rules.items():
            exposure = eigvec.T @ normalized_weight
            contributions = eigval * exposure**2
            total = contributions.sum()
            for rank, value in enumerate(contributions / total if total > 0 else contributions, start=1):
                eigenrows.append((date, strategy, rank, value))

    gross_df = pd.DataFrame(gross, columns=["date", "strategy", "ret"]).pivot(index="date", columns="strategy", values="ret")
    turnover = pd.DataFrame(costs, columns=["date", "strategy", "turnover", "cost"])
    cost_wide = turnover.pivot(index="date", columns="strategy", values="cost")
    net = gross_df - cost_wide
    return ARPResult(
        returns=net,
        gross_returns=gross_df,
        turnover=turnover,
        weights=pd.DataFrame(weights, columns=["date", "strategy", "factor", "weight"]),
        eigenrisk=pd.DataFrame(eigenrows, columns=["date", "strategy", "eigenmode", "risk_share"]),
    )
