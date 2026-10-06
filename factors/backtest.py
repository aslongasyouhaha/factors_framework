from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import Factor
from .engine import Book, LayerBook, SimulationResult, WeightBook, simulate
from .panel import to_wide


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


def _wide_returns(
    returns: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Accept either a date x asset matrix or a long (date, asset, ret) panel."""
    if isinstance(returns.index, pd.DatetimeIndex) and ret_col not in returns.columns:
        return returns
    frame = returns
    if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names[:2]) == ["date", "asset_id"]:
        frame = frame.reset_index().rename(columns={"date": date_col, "asset_id": asset_col})
    return to_wide(frame, ret_col, date_col=date_col, asset_col=asset_col)


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
        return simulate(book, _wide_returns(returns), drift=drift, cost_bps=cost_bps, weight_base=weight_base)

    def ic_wide(self, signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
        """Per-date Pearson, rank and cosine IC between two aligned date x asset matrices."""
        forward = forward.reindex(index=signal.index, columns=signal.columns)
        x = signal.to_numpy(dtype=float)
        y = forward.to_numpy(dtype=float)
        st = _row_stats(x, y)
        ok = st["n"] >= 2
        rx = pd.DataFrame(np.where(st["mask"], x, np.nan)).rank(axis=1).to_numpy()
        ry = pd.DataFrame(np.where(st["mask"], y, np.nan)).rank(axis=1).to_numpy()
        rk = _row_stats(rx, ry)
        out = pd.DataFrame(
            {
                "ic": _safe_div(st["sxy"], np.sqrt(st["sxx"] * st["syy"]), ok),
                "rank_ic": _safe_div(rk["sxy"], np.sqrt(rk["sxx"] * rk["syy"]), ok),
                "cos_ic": _safe_div(st["xy"], np.sqrt(st["xx"] * st["yy"]), ok),
                "n_assets": st["n"],
            },
            index=pd.DatetimeIndex(signal.index, name="date"),
        )
        for col in ["ic", "rank_ic", "cos_ic"]:
            out[f"cum_{col}"] = out[col].fillna(0.0).cumsum()
            vol = out[col].std(ddof=1)
            out[f"{col}_ir"] = np.nan if vol == 0 or pd.isna(vol) else out[col].mean() / vol
        return out

    def r2_wide(self, signal: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
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

    # ------------------------------------------------------------------ long-format adapters

    @staticmethod
    def _long_portfolio(sim: SimulationResult, name: str, date_col: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "ret": sim.returns[name],
                "gross_exposure": sim.gross[name],
                "net_exposure": sim.net[name],
                "n_assets": sim.n_assets[name].astype(int),
            }
        )
        out = out[out["n_assets"].gt(0)]
        return out.rename_axis(date_col).reset_index()

    def compute_turnover(
        self,
        holdings: pd.DataFrame,
        rebalance_col: str = "rebalance_date",
        asset_col: str = "asset_id",
        weight_col: str = "target_weight",
    ) -> pd.DataFrame:
        book = WeightBook.from_holdings(
            holdings, weight_col=weight_col, rebalance_col=rebalance_col, asset_col=asset_col,
            hold_start_col=None, hold_end_col=None,
        )
        w = book._weights[:, :, 0]
        prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        return pd.DataFrame(
            {
                rebalance_col: book.dates,
                "turnover": 0.5 * np.abs(w - prev).sum(axis=1),
                "gross": np.abs(w).sum(axis=1),
                "net": w.sum(axis=1),
                "n_assets": (w != 0).sum(axis=1).astype(int),
            }
        )

    def portfolio_returns(
        self,
        holdings: pd.DataFrame,
        returns: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
        ret_col: str = "ret",
        mode: str = "factor",
        transaction_cost_bps: float = 0.0,
        drift: bool = False,
    ) -> pd.DataFrame:
        h = holdings
        if mode == "long_only":
            h = h[h["leg"].eq("long")].copy()
            h["target_weight"] = h["target_weight"].abs()
            h["target_weight"] = h["target_weight"] / h.groupby("rebalance_date")["target_weight"].transform("sum")
        elif mode != "factor":
            raise ValueError("mode must be 'factor' or 'long_only'.")
        book = WeightBook.from_holdings(h, asset_col=asset_col)
        r = _wide_returns(returns, date_col=date_col, asset_col=asset_col, ret_col=ret_col)
        sim = simulate(book, r, drift=drift, cost_bps=transaction_cost_bps)
        out = self._long_portfolio(sim, "portfolio", date_col)
        if transaction_cost_bps:
            cost = sim.cost["portfolio"]
            first_rows = book.segments(pd.DatetimeIndex(r.index))[:, 0]
            charged = pd.Series(0.0, index=sim.returns.index)
            for lo, c in zip(first_rows, cost.to_numpy()):
                if lo < len(r.index) and r.index[lo] in charged.index:
                    charged[r.index[lo]] += c
            out["cost"] = charged.reindex(out[date_col]).to_numpy()
            out["ret_before_cost"] = out["ret"] + out["cost"]
        return out

    def layer_portfolio_returns(
        self,
        holdings: pd.DataFrame,
        returns: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
        ret_col: str = "ret",
        group_col: str = "group",
        drift: bool = False,
    ) -> pd.DataFrame:
        book = LayerBook.from_holdings(holdings, group_col=group_col, asset_col=asset_col)
        r = _wide_returns(returns, date_col=date_col, asset_col=asset_col, ret_col=ret_col)
        sim = simulate(book, r, drift=drift)
        ret = sim.returns.stack().rename("ret")
        n = sim.n_assets.stack().rename("n_assets").astype(int)
        out = pd.concat([ret, n], axis=1).rename_axis([date_col, group_col]).reset_index()
        return out[out["n_assets"].gt(0)].sort_values([date_col, group_col]).reset_index(drop=True)

    def ic_analysis(
        self,
        panel: pd.DataFrame,
        signal_col: str = "factor_value",
        forward_ret_col: str = "forward_ret",
        date_col: str = "date",
        asset_col: str = "asset_id",
    ) -> pd.DataFrame:
        signal = to_wide(panel, signal_col, date_col=date_col, asset_col=asset_col)
        forward = to_wide(panel, forward_ret_col, date_col=date_col, asset_col=asset_col)
        return self.ic_wide(signal, forward).rename_axis(date_col).reset_index()

    def r2_analysis(
        self,
        panel: pd.DataFrame,
        signal_col: str = "factor_value",
        forward_ret_col: str = "forward_ret",
        date_col: str = "date",
        asset_col: str = "asset_id",
    ) -> pd.DataFrame:
        signal = to_wide(panel, signal_col, date_col=date_col, asset_col=asset_col)
        forward = to_wide(panel, forward_ret_col, date_col=date_col, asset_col=asset_col)
        return self.r2_wide(signal, forward).rename_axis(date_col).reset_index()

    # ------------------------------------------------------------------ statistics

    def summary(
        self,
        returns: pd.Series,
        turnover: pd.DataFrame | pd.Series | None = None,
        turnover_periods_per_year: float | None = None,
    ) -> pd.Series:
        s = pd.Series(returns).dropna()
        if s.empty:
            return pd.Series(dtype=float)
        vol = s.std(ddof=1)
        wealth = (1 + s).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        out = {
            "n": len(s),
            "ann_return": self.periods_per_year * s.mean(),
            "ann_vol": np.sqrt(self.periods_per_year) * vol,
            "sharpe": np.nan if vol == 0 else np.sqrt(self.periods_per_year) * s.mean() / vol,
            "total_return": wealth.iloc[-1] - 1,
            "max_drawdown": drawdown.min(),
            "hit_rate": (s > 0).mean(),
            "min": s.min(),
            "max": s.max(),
        }
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
    def ols(y: pd.Series, x: pd.DataFrame) -> pd.Series:
        data = pd.concat([y.rename("y"), x], axis=1).dropna()
        if data.empty:
            return pd.Series(dtype=float)
        yv = data["y"].to_numpy(dtype=float)
        xv = data.drop(columns="y").to_numpy(dtype=float)
        xv = np.column_stack([np.ones(len(xv)), xv])
        beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        fitted = xv @ beta
        resid = yv - fitted
        dof = max(len(yv) - xv.shape[1], 1)
        sigma2 = float(resid.T @ resid / dof)
        cov = sigma2 * np.linalg.pinv(xv.T @ xv)
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
    ) -> pd.Series:
        data = asset_returns.merge(factors[[date_col, rf_col, *factor_cols]], on=date_col, how="inner")
        y = data[ret_col] - data[rf_col].fillna(0.0)
        x = data[list(factor_cols)]
        result = self.ols(y, x)
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
        holdings: pd.DataFrame | Book,
        returns: pd.DataFrame,
        signal_panel: pd.DataFrame | None = None,
        mode: str = "factor",
        transaction_cost_bps: float = 0.0,
        make_plots: bool = True,
        drift: bool = False,
        forward: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run a full backtest.

        With a Book, `returns` is a date x asset matrix, `signal_panel` a date x asset signal
        matrix and `forward` the matching forward returns (see engine.forward_returns).
        With a long holdings table the legacy long-format path is used.
        """
        if isinstance(holdings, Book):
            return self._run_book(holdings, returns, signal_panel, forward, transaction_cost_bps, make_plots, drift)

        turnover = self.compute_turnover(holdings)
        port_ret = self.portfolio_returns(
            holdings, returns, mode=mode, transaction_cost_bps=transaction_cost_bps, drift=drift
        )
        summ = self.summary(port_ret["ret"], turnover=turnover)
        layers = self.layer_portfolio_returns(holdings, returns, drift=drift)
        ic = self.ic_analysis(signal_panel) if signal_panel is not None else None
        r2 = self.r2_analysis(signal_panel) if signal_panel is not None else None
        figures = {}
        if make_plots:
            figures["performance"] = self.plot_performance(port_ret)
            figures["layers"] = self.plot_layers(layers)
        return BacktestResult(
            returns=port_ret, summary=summ, turnover=turnover, layer_returns=layers, ic=ic, r2=r2, figures=figures
        )

    def _run_book(
        self,
        book: Book,
        returns: pd.DataFrame,
        signal: pd.DataFrame | None,
        forward: pd.DataFrame | None,
        cost_bps: float,
        make_plots: bool,
        drift: bool,
    ) -> BacktestResult:
        sim = self.simulate(book, returns, drift=drift, cost_bps=cost_bps)
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
        )

    def run_factor(
        self,
        factor: Factor,
        returns: pd.DataFrame,
        signal_panel: pd.DataFrame | None = None,
        mode: str = "factor",
        transaction_cost_bps: float = 0.0,
        make_plots: bool = True,
        drift: bool = False,
    ) -> BacktestResult:
        return self.run(
            factor.holdings,
            returns,
            signal_panel=signal_panel,
            mode=mode,
            transaction_cost_bps=transaction_cost_bps,
            make_plots=make_plots,
            drift=drift,
        )
