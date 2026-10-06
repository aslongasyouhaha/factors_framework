from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import WideFactor
from .engine import Book, LayerBook, SimulationResult, forward_returns, simulate
from .evaluation import (
    FamaMacBethResult,
    fama_macbeth,
    ic_by_year,
    ic_decay,
    ic_summary,
    information_coefficient,
    newey_west,
    performance_by_year,
    performance_summary,
    univariate_r2,
)


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    summary: pd.Series | pd.DataFrame
    turnover: pd.DataFrame
    layer_returns: pd.DataFrame | None = None
    ic: pd.DataFrame | None = None
    r2: pd.DataFrame | None = None
    figures: dict[str, Any] = field(default_factory=dict)
    simulation: SimulationResult | None = None
    by_year: pd.DataFrame | None = None
    ic_summary: pd.DataFrame | None = None
    ic_by_year: pd.DataFrame | None = None


def _require_wide(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a date x asset matrix with a DatetimeIndex; convert long data with to_wide().")
    return returns


@dataclass(frozen=True)
class FactorBacktester:
    periods_per_year: int = 252

    # ------------------------------------------------------------------ wide (primary) API

    def simulate(
        self,
        book: Book,
        returns: pd.DataFrame,
        drift: bool = False,
        cost_bps: float = 0.0,
        weight_base: pd.DataFrame | None = None,
    ) -> SimulationResult:
        """Simulate every portfolio in `book` on a date x asset return matrix (see engine.simulate)."""
        return simulate(book, _require_wide(returns), drift=drift, cost_bps=cost_bps, weight_base=weight_base)

    def ic_wide(self, signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
        """Per-date Pearson, rank and cosine IC between two aligned date x asset matrices."""
        out = information_coefficient(signal, forward)
        for col in ["ic", "rank_ic", "cos_ic"]:
            out[f"cum_{col}"] = out[col].fillna(0.0).cumsum()
            vol = out[col].std(ddof=1)
            out[f"{col}_ir"] = np.nan if vol == 0 or pd.isna(vol) else out[col].mean() / vol
        return out

    def r2_wide(self, signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
        """Per-date univariate cross-sectional regression forward ~ alpha + beta * signal."""
        return univariate_r2(signal, forward)

    # ------------------------------------------------------------------ evaluation

    def newey_west(self, x: pd.Series, lags: int | None = None) -> pd.Series:
        """Mean, Newey-West standard error and t-stat of a time series."""
        return newey_west(x, lags)

    def ic_summary(self, ic: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
        """Mean, std, IR, Newey-West t and hit rate of ic / rank_ic / cos_ic."""
        return ic_summary(ic, lags)

    def ic_decay(self, signal: pd.DataFrame, period_returns: pd.DataFrame, max_lag: int = 12, lags: int | None = None) -> pd.DataFrame:
        """IC against the return h periods ahead, h = 1..max_lag (see evaluation.ic_decay)."""
        return ic_decay(signal, period_returns, max_lag=max_lag, lags=lags)

    def fama_macbeth(
        self,
        forward: pd.DataFrame,
        characteristics: dict[str, pd.DataFrame],
        categories=None,
        add_intercept: bool = True,
        min_obs: int = 30,
        lags: int | None = None,
    ) -> FamaMacBethResult:
        """Per-date cross-sectional regressions with Newey-West t-stats on the average slopes."""
        return fama_macbeth(forward, characteristics, categories, add_intercept, min_obs, lags)

    def performance_by_year(self, returns: pd.Series | pd.DataFrame, nw_lags: int | None = None) -> pd.DataFrame:
        """summary statistics for each calendar year."""
        return performance_by_year(returns, self.periods_per_year, nw_lags)

    def ic_by_year(self, ic: pd.DataFrame) -> pd.DataFrame:
        return ic_by_year(ic)

    def compute_turnover(self, book: Book, portfolio: str | None = None) -> pd.DataFrame:
        """Per-rebalance turnover (0.5 * sum|w_k - w_{k-1}|, target to target), gross, net and
        number of holdings for one portfolio of the book."""
        names = book.names
        if portfolio is None:
            if len(names) != 1:
                raise ValueError(f"Book holds {names}; pass portfolio=.")
            portfolio = names[0]
        p = names.index(portfolio)
        w = np.stack([book.weights_at(k)[:, p] for k in range(len(book.dates))]) if len(book.dates) else np.zeros((0, len(book.assets)))
        prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        return pd.DataFrame(
            {
                "rebalance_date": book.dates,
                "turnover": 0.5 * np.abs(w - prev).sum(axis=1),
                "gross": np.abs(w).sum(axis=1),
                "net": w.sum(axis=1),
                "n_assets": (w != 0).sum(axis=1).astype(int),
            }
        )

    # ------------------------------------------------------------------ statistics

    def summary(
        self,
        returns: pd.Series,
        turnover: pd.DataFrame | pd.Series | None = None,
        turnover_periods_per_year: float | None = None,
        nw_lags: int | None = None,
    ) -> pd.Series:
        """Annualised performance plus t_nw, the Newey-West t-stat of the mean return."""
        out = performance_summary(returns, self.periods_per_year, nw_lags)
        if out.empty:
            return out
        out = out.to_dict()
        if isinstance(turnover, pd.DataFrame):
            turnover = turnover["turnover"] if "turnover" in turnover.columns else None
        if turnover is not None and not turnover.empty:
            turnover_ppy = self.periods_per_year if turnover_periods_per_year is None else turnover_periods_per_year
            out["avg_turnover"] = turnover.mean()
            out["ann_turnover"] = turnover.mean() * turnover_ppy
        return pd.Series(out)

    def summary_table(
        self,
        sim: SimulationResult,
        turnover_periods_per_year: float | None = None,
    ) -> pd.DataFrame:
        rows = {
            name: self.summary(sim.returns[name], sim.turnover[name], turnover_periods_per_year)
            for name in sim.returns.columns
        }
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------ plotting

    @staticmethod
    def _plotly_x(values: pd.Series | pd.Index) -> np.ndarray:
        return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]")

    def plot_performance(self, returns: pd.DataFrame | pd.Series, date_col: str = "date", ret_col: str = "ret"):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise ImportError("plotly is required for plotting. Install plotly first.") from exc

        if isinstance(returns, pd.Series):
            returns = returns.rename(ret_col).rename_axis(date_col).reset_index()
        r = returns.sort_values(date_col).copy()
        r["cum_return"] = (1 + r[ret_col]).cumprod() - 1
        wealth = 1 + r["cum_return"]
        r["drawdown"] = wealth / wealth.cummax() - 1
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        x = self._plotly_x(r[date_col])
        fig.add_trace(go.Scatter(x=x, y=r["cum_return"].to_numpy(), name="Cumulative Return"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=r["drawdown"].to_numpy(), name="Drawdown", fill="tozeroy"), row=2, col=1)
        fig.update_yaxes(tickformat=".2%", row=1, col=1)
        fig.update_yaxes(tickformat=".2%", row=2, col=1)
        fig.update_layout(template="plotly_white", height=650, title="Factor Performance")
        return fig

    def plot_layers(self, layer_returns: pd.DataFrame, date_col: str = "date", group_col: str = "group"):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for plotting. Install plotly first.") from exc

        if group_col in layer_returns.columns:
            wide = layer_returns.pivot(index=date_col, columns=group_col, values="ret").sort_index()
        else:
            wide = layer_returns.sort_index()
        cum = (1 + wide).cumprod() - 1
        fig = go.Figure()
        x = self._plotly_x(cum.index)
        for group in cum.columns:
            fig.add_trace(go.Scatter(x=x, y=cum[group].to_numpy(), name=str(group)))
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(template="plotly_white", height=500, title="Layer Portfolio Returns")
        return fig

    # ------------------------------------------------------------------ regressions

    @staticmethod
    def ols(y: pd.Series, x: pd.DataFrame, hac_lags: int | None = None) -> pd.Series:
        """OLS with intercept; t-stats are classical, or Newey-West with hac_lags (0 = White)."""
        data = pd.concat([y.rename("y"), x], axis=1).dropna()
        if data.empty:
            return pd.Series(dtype=float)
        yv = data["y"].to_numpy(dtype=float)
        xv = data.drop(columns="y").to_numpy(dtype=float)
        xv = np.column_stack([np.ones(len(xv)), xv])
        beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        fitted = xv @ beta
        resid = yv - fitted
        bread = np.linalg.pinv(xv.T @ xv)
        if hac_lags is None:
            dof = max(len(yv) - xv.shape[1], 1)
            cov = float(resid.T @ resid / dof) * bread
        else:
            # Newey-West sandwich with Bartlett weights (statsmodels HAC, no df correction).
            u = xv * resid[:, None]
            meat = u.T @ u
            for lag in range(1, min(hac_lags, len(yv) - 1) + 1):
                g = u[lag:].T @ u[:-lag]
                meat += (1.0 - lag / (hac_lags + 1)) * (g + g.T)
            cov = bread @ meat @ bread
        se = np.sqrt(np.diag(cov))
        names = ["alpha", *data.drop(columns="y").columns.tolist()]
        out = {}
        for name, b, s in zip(names, beta, se):
            out[name] = b
            out[f"{name}_t"] = np.nan if s == 0 else b / s
        denom = ((yv - yv.mean()) ** 2).sum()
        out["r2"] = np.nan if denom == 0 else 1 - float((resid**2).sum() / denom)
        out["n"] = len(yv)
        return pd.Series(out)

    def regress_on_factors(
        self,
        asset_returns: pd.DataFrame,
        factors: pd.DataFrame,
        ret_col: str = "ret",
        date_col: str = "date",
        factor_cols: tuple[str, ...] = ("MKT_RF", "SMB", "HML"),
        rf_col: str = "RF",
        hac_lags: int | None = None,
    ) -> pd.Series:
        data = asset_returns.merge(factors[[date_col, rf_col, *factor_cols]], on=date_col, how="inner")
        y = data[ret_col] - data[rf_col].fillna(0.0)
        x = data[list(factor_cols)]
        result = self.ols(y, x, hac_lags=hac_lags)
        if "alpha" in result:
            result["alpha_ann"] = self.periods_per_year * result["alpha"]
        return result

    def batch_regress(
        self,
        asset_panel: pd.DataFrame,
        factors: pd.DataFrame,
        asset_col: str = "asset_id",
        ret_col: str = "ret",
        date_col: str = "date",
    ) -> pd.DataFrame:
        rows = []
        for asset, g in asset_panel.groupby(asset_col):
            result = self.regress_on_factors(g, factors, ret_col=ret_col, date_col=date_col)
            if not result.empty:
                row = result.to_dict()
                row[asset_col] = asset
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ one-shot runs

    def run(
        self,
        book: Book,
        returns: pd.DataFrame,
        signal: pd.DataFrame | None = None,
        forward: pd.DataFrame | None = None,
        cost_bps: float = 0.0,
        drift: bool = False,
        weight_base: pd.DataFrame | None = None,
        make_plots: bool = True,
    ) -> BacktestResult:
        """Simulate a book and evaluate it.

        returns: date x asset matrix. signal / forward: rebalance-date x asset matrices of the
        ranking signal and the return earned over the following holding period (see
        engine.forward_returns); IC statistics are computed when both are given.
        """
        sim = self.simulate(book, returns, drift=drift, cost_bps=cost_bps, weight_base=weight_base)
        layer_names = book.labels if isinstance(book, LayerBook) else []
        main = [c for c in sim.returns.columns if c not in layer_names] or list(sim.returns.columns)
        ic = r2 = None
        if signal is not None and forward is not None:
            ic = self.ic_wide(signal, forward)
            r2 = self.r2_wide(signal, forward)
        figures = {}
        if make_plots:
            figures["performance"] = self.plot_performance(sim.returns[main[0]])
            if layer_names:
                figures["layers"] = self.plot_layers(sim.returns[layer_names])
        return BacktestResult(
            returns=sim.returns[main],
            summary=self.summary_table(sim),
            turnover=sim.turnover,
            layer_returns=sim.returns[layer_names] if layer_names else None,
            ic=ic,
            r2=r2,
            figures=figures,
            simulation=sim,
            by_year=self.performance_by_year(sim.returns[main]),
            ic_summary=self.ic_summary(ic) if ic is not None else None,
            ic_by_year=self.ic_by_year(ic) if ic is not None else None,
        )

    def run_factor(self, factor: WideFactor, returns: pd.DataFrame, **kwargs) -> BacktestResult:
        """run() for a factor from build_wide(): its signal is evaluated against the return over
        each of its holding periods."""
        book = factor.book
        forward = forward_returns(_require_wide(returns), book.dates, book.start_dates, book.end_dates)
        return self.run(book, returns, signal=factor.signal, forward=forward, **kwargs)
