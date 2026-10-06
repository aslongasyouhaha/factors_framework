from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .grouped import group_ols
from .panel import _category_codes

# Factor evaluation: Newey-West inference, IC statistics and decay, Fama-MacBeth regressions
# and calendar-year breakdowns. Cross-sectional inputs are date x asset matrices.


# ------------------------------------------------------------------ Newey-West


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) automatic bandwidth floor(4 * (n / 100) ** (2 / 9))."""
    return int(np.floor(4 * (max(n, 1) / 100) ** (2 / 9)))


def newey_west(x: pd.Series | np.ndarray, lags: int | None = None) -> pd.Series:
    """Mean of a time series with its Newey-West (Bartlett kernel) standard error and t-stat.

    Var(mean) = (gamma_0 + 2 * sum_l (1 - l / (L + 1)) * gamma_l) / T with gamma_l the
    lag-l autocovariance divided by T; this is statsmodels HAC with use_correction=False.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    t = len(v)
    lags = newey_west_lags(t) if lags is None else int(lags)
    if t < 2:
        return pd.Series({"mean": v.mean() if t else np.nan, "se": np.nan, "t": np.nan, "lags": lags, "n": t})
    d = v - v.mean()
    s = d @ d / t
    for lag in range(1, min(lags, t - 1) + 1):
        s += 2.0 * (1.0 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / t
    se = np.sqrt(s / t) if s > 0 else np.nan
    return pd.Series({"mean": v.mean(), "se": se, "t": v.mean() / se if se else np.nan, "lags": lags, "n": t})


def newey_west_table(frame: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
    """newey_west applied to every column; one row per column."""
    return pd.DataFrame({col: newey_west(frame[col], lags) for col in frame.columns}).T


# ------------------------------------------------------------------ IC


def _row_stats(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Pairwise-complete cross-sectional moments of each row of x against y."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum(axis=1)
    x = np.where(mask, x, 0.0)
    y = np.where(mask, y, 0.0)
    safe_n = np.maximum(n, 1)
    mx = x.sum(axis=1) / safe_n
    my = y.sum(axis=1) / safe_n
    dx = np.where(mask, x - mx[:, None], 0.0)
    dy = np.where(mask, y - my[:, None], 0.0)
    return {
        "n": n,
        "mx": mx,
        "my": my,
        "sxx": (dx * dx).sum(axis=1),
        "syy": (dy * dy).sum(axis=1),
        "sxy": (dx * dy).sum(axis=1),
        "xx": (x * x).sum(axis=1),
        "yy": (y * y).sum(axis=1),
        "xy": (x * y).sum(axis=1),
        "mask": mask,
    }


def _safe_div(num: np.ndarray, den: np.ndarray, ok: np.ndarray) -> np.ndarray:
    ok = ok & (den > 0)
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def information_coefficient(signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Per-date Pearson, rank and cosine IC between two date x asset matrices."""
    forward = forward.reindex(index=signal.index, columns=signal.columns)
    x = signal.to_numpy(dtype=float)
    y = forward.to_numpy(dtype=float)
    st = _row_stats(x, y)
    ok = st["n"] >= 2
    rx = pd.DataFrame(np.where(st["mask"], x, np.nan)).rank(axis=1).to_numpy()
    ry = pd.DataFrame(np.where(st["mask"], y, np.nan)).rank(axis=1).to_numpy()
    rk = _row_stats(rx, ry)
    return pd.DataFrame(
        {
            "ic": _safe_div(st["sxy"], np.sqrt(st["sxx"] * st["syy"]), ok),
            "rank_ic": _safe_div(rk["sxy"], np.sqrt(rk["sxx"] * rk["syy"]), ok),
            "cos_ic": _safe_div(st["xy"], np.sqrt(st["xx"] * st["yy"]), ok),
            "n_assets": st["n"],
        },
        index=pd.DatetimeIndex(signal.index, name="date"),
    )


def univariate_r2(signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Per-date univariate cross-sectional regression forward ~ alpha + beta * signal."""
    forward = forward.reindex(index=signal.index, columns=signal.columns)
    st = _row_stats(signal.to_numpy(dtype=float), forward.to_numpy(dtype=float))
    ok = st["n"] >= 3
    beta = _safe_div(st["sxy"], st["sxx"], ok)
    return pd.DataFrame(
        {
            "r2": _safe_div(st["sxy"] ** 2, st["sxx"] * st["syy"], ok),
            "alpha": st["my"] - beta * st["mx"],
            "beta": beta,
            "n_assets": st["n"],
        },
        index=pd.DatetimeIndex(signal.index, name="date"),
    )


def ic_summary(ic: pd.DataFrame, lags: int | None = None, cols: Sequence[str] = ("ic", "rank_ic", "cos_ic")) -> pd.DataFrame:
    """Mean, volatility, IR, Newey-West t and hit rate for each IC series."""
    rows = {}
    for col in cols:
        if col not in ic.columns:
            continue
        s = ic[col].dropna()
        nw = newey_west(s, lags)
        std = s.std(ddof=1)
        rows[col] = {
            "mean": nw["mean"],
            "std": std,
            "ir": nw["mean"] / std if std else np.nan,
            "t_nw": nw["t"],
            "pct_positive": (s > 0).mean() if len(s) else np.nan,
            "n_periods": nw["n"],
            "nw_lags": nw["lags"],
        }
    return pd.DataFrame(rows).T


def ic_decay(
    signal: pd.DataFrame,
    period_returns: pd.DataFrame,
    max_lag: int = 12,
    lags: int | None = None,
) -> pd.DataFrame:
    """IC of the signal at t against the return of holding period t + h - 1, h = 1..max_lag.

    `period_returns` holds, on each signal date, the return of the period that starts there
    (e.g. engine.forward_returns(returns, signal.index), or next-day returns for a daily signal).
    The `half_life` attribute is the first lag whose |mean rank IC| falls below half of lag 1's.
    """
    period_returns = period_returns.reindex(index=signal.index, columns=signal.columns)
    rows = {}
    for h in range(1, max_lag + 1):
        ic = information_coefficient(signal, period_returns.shift(-(h - 1)))
        summ = ic_summary(ic, lags, cols=("ic", "rank_ic"))
        rows[h] = {
            "mean_ic": summ.loc["ic", "mean"],
            "ic_t_nw": summ.loc["ic", "t_nw"],
            "mean_rank_ic": summ.loc["rank_ic", "mean"],
            "rank_ic_ir": summ.loc["rank_ic", "ir"],
            "rank_ic_t_nw": summ.loc["rank_ic", "t_nw"],
            "n_periods": summ.loc["rank_ic", "n_periods"],
        }
    out = pd.DataFrame(rows).T.rename_axis("lag")
    first = abs(out["mean_rank_ic"].iloc[0])
    below = out.index[out["mean_rank_ic"].abs() < first / 2]
    out.attrs["half_life"] = int(below[0]) if len(below) and np.isfinite(first) else None
    return out


# ------------------------------------------------------------------ Fama-MacBeth


@dataclass
class FamaMacBethResult:
    coefficients: pd.DataFrame  # date x term
    summary: pd.DataFrame  # term x (mean, se, t, ...)
    r2: pd.Series
    n_assets: pd.Series


def fama_macbeth(
    forward: pd.DataFrame,
    characteristics: dict[str, pd.DataFrame],
    categories=None,
    add_intercept: bool = True,
    min_obs: int = 30,
    lags: int | None = None,
) -> FamaMacBethResult:
    """Fama-MacBeth: per-date cross-sectional OLS, then Newey-West inference on the slopes.

    forward: date x asset returns earned after each date (dependent variable).
    characteristics: name -> date x asset matrix known at each date (regressors).
    categories: optional date x asset label matrix (or list) for fixed effects such as industry;
    their dummies are absorbed, so no intercept is reported in that case.
    """
    names = list(characteristics)
    idx = forward.index
    cols = forward.columns
    t, n = forward.shape
    x = np.column_stack(
        [characteristics[k].reindex(index=idx, columns=cols).to_numpy(dtype=float).ravel() for k in names]
    ) if names else None
    rows = np.repeat(np.arange(t, dtype=np.int64), n)
    fit = group_ols(
        forward.to_numpy(dtype=float).ravel(),
        x,
        rows,
        t,
        category_codes=_category_codes(forward, categories),
        add_intercept=add_intercept,
        min_obs=min_obs,
    )
    coef = pd.DataFrame(fit["beta"], index=pd.DatetimeIndex(idx, name="date"), columns=names)
    if add_intercept and categories is None:
        coef.insert(0, "const", fit["intercept"])
    valid = fit["n"] > 0
    coef = coef[valid]
    summary = newey_west_table(coef, lags)
    summary["pct_positive"] = (coef > 0).mean()
    summary["std"] = coef.std(ddof=1)
    return FamaMacBethResult(
        coefficients=coef,
        summary=summary[["mean", "se", "t", "std", "pct_positive", "n", "lags"]],
        r2=pd.Series(fit["r2"][valid], index=coef.index, name="r2"),
        n_assets=pd.Series(fit["n"][valid], index=coef.index, name="n_assets"),
    )


# ------------------------------------------------------------------ performance


def performance_summary(returns: pd.Series, periods_per_year: float, nw_lags: int | None = None) -> pd.Series:
    """Annualised return / vol / Sharpe, drawdown, hit rate and Newey-West t of the mean."""
    s = pd.Series(returns).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    vol = s.std(ddof=1)
    wealth = (1 + s).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    nw = newey_west(s, nw_lags)
    return pd.Series(
        {
            "n": len(s),
            "ann_return": periods_per_year * s.mean(),
            "ann_vol": np.sqrt(periods_per_year) * vol,
            "sharpe": np.nan if vol == 0 else np.sqrt(periods_per_year) * s.mean() / vol,
            "t_nw": nw["t"],
            "total_return": wealth.iloc[-1] - 1,
            "max_drawdown": drawdown.min(),
            "hit_rate": (s > 0).mean(),
            "min": s.min(),
            "max": s.max(),
        }
    )


def performance_by_year(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: float,
    nw_lags: int | None = None,
) -> pd.DataFrame:
    """performance_summary for each calendar year (and each column of a DataFrame).

    total_return is the compounded return within the year; ann_* use periods_per_year.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    parts = {}
    for col in frame.columns:
        s = frame[col].dropna()
        for year, g in s.groupby(pd.DatetimeIndex(s.index).year):
            parts[(col, year)] = performance_summary(g, periods_per_year, nw_lags)
    out = pd.DataFrame(parts).T
    out.index.names = ["portfolio", "year"]
    if isinstance(returns, pd.Series):
        out = out.droplevel("portfolio")
    return out


def ic_by_year(ic: pd.DataFrame, cols: Sequence[str] = ("ic", "rank_ic")) -> pd.DataFrame:
    """Mean IC, IC IR and hit rate per calendar year."""
    years = pd.DatetimeIndex(ic.index).year
    out = {}
    for col in cols:
        g = ic[col].groupby(years)
        out[f"mean_{col}"] = g.mean()
        out[f"{col}_ir"] = g.mean() / g.std(ddof=1)
        out[f"{col}_pct_positive"] = g.apply(lambda s: (s.dropna() > 0).mean())
    out["n_periods"] = ic[cols[0]].groupby(years).count()
    return pd.DataFrame(out).rename_axis("year")
