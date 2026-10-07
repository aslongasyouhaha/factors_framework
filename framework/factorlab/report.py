from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data, panels
from .backtest import FactorBacktester
from .base import FactorConfig
from .engine import forward_returns
from .evaluation import ic_decay, ic_summary
from .evaluation import performance_by_year, performance_summary
from .quantile import QuantileSignalFactorBuilder

# Standard result page for a factor stored with data.save_factor: decile layers, the long-short
# spread, turnover, IC (with Newey-West t-stats), IC decay and calendar-year performance.
# Every project's result/ folder is produced by this module so results are comparable.

DEFAULT_REPORT = {
    "n_groups": 10,
    "weighting": "value",
    "long_groups": ["D10"],
    "short_groups": ["D01"],
    "winsorize": [0.01, 0.99],
    "periods_per_year": 12,
    "rebalances_per_year": 12,
}


def paper_factor_report(project: str, out_dir: Path, factor_column: str) -> dict[str, Any]:
    """Report a paper-specific traded factor stored in factor_returns.parquet."""
    _, meta = data.load_factor(project)
    returns = data.load_factor_table(project, "factor_returns")
    returns["date"] = pd.to_datetime(returns["date"])
    series = returns.set_index("date")[factor_column].dropna().sort_index()
    summary = performance_summary(series, 12)
    by_year = performance_by_year(series, 12)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out_dir / "factor_returns.csv", index=False)
    summary.to_frame().T.to_csv(out_dir / "summary.csv")
    by_year.to_csv(out_dir / "by_year.csv")
    if (data.factor_dir(project) / "group_returns.parquet").exists():
        data.load_factor_table(project, "group_returns").to_csv(out_dir / "group_returns.csv", index=False)
    head = summary[["n", "ann_return", "ann_vol", "sharpe", "t_nw", "max_drawdown", "hit_rate"]].to_frame().T
    head.index.name = "factor"
    lines = [
        f"# {meta.get('title', factor_column)}", "", f"- Paper: {meta.get('paper', '')}",
        f"- Construction: {meta.get('description', '')}", "", "## Factor return",
        _fmt_table(head, pct=("ann_return", "ann_vol", "max_drawdown", "hit_rate")), "", "## By year",
        _fmt_table(by_year[["n", "total_return", "sharpe", "max_drawdown", "hit_rate"]], pct=("total_return", "max_drawdown", "hit_rate")), ""
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary, "by_year": by_year, "returns": returns}


def weekly_crsp_inputs(rebalance_dates: pd.DatetimeIndex, assets: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compound CRSP daily returns into consecutive weekly formation periods."""
    from .panel import to_wide
    rebalances = pd.DatetimeIndex(sorted(pd.to_datetime(rebalance_dates).unique()))
    returns, weights = [], []
    for file in sorted((data.BASE / "crsp_daily").glob("????.parquet")):
        x = pd.read_parquet(file, columns=["date","asset_id","ret","market_equity"])
        x["date"] = pd.to_datetime(x["date"])
        x = x[x["asset_id"].isin(assets)].sort_values(["asset_id","date"])
        weights.append(x[x["date"].isin(rebalances)][["date","asset_id","market_equity"]])
        pos = rebalances.searchsorted(x["date"], side="left")
        valid = pos < len(rebalances)
        x = x[valid].copy(); x["date"] = rebalances[pos[valid]].to_numpy(); x["gross"] = 1 + x["ret"].fillna(0)
        returns.append(x.groupby(["date","asset_id"], as_index=False)["gross"].prod())
    weekly = pd.concat(returns).groupby(["date","asset_id"], as_index=False)["gross"].prod()
    weekly["ret"] = weekly["gross"] - 1
    me = pd.concat(weights).drop_duplicates(["date","asset_id"], keep="last")
    return to_wide(weekly,"ret"), to_wide(me,"market_equity")


def _next_period_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return of the first return period after each date (the IC target)."""
    pos = returns.index.searchsorted(dates, side="right")
    ends = returns.index[np.minimum(pos, len(returns.index) - 1)]
    ends = ends.where(pos < len(returns.index), dates)  # no later period: empty window -> NaN
    return forward_returns(returns, dates, end_dates=ends)


def _fmt_table(frame: pd.DataFrame, pct: tuple[str, ...] = (), digits: int = 3) -> str:
    out = frame.copy()
    for col in out.columns:
        if col in pct:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{100 * v:.2f}%")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join([out.index.name or ""] + [str(c) for c in out.columns]) + " |"
    rule = "|" + "---|" * (len(out.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(out.index, out.to_numpy())]
    return "\n".join([header, rule, *rows])


def quantile_report(
    project: str,
    out_dir: Path,
    returns: pd.DataFrame | None = None,
    market_equity: pd.DataFrame | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Build the decile portfolios of a stored exposure and write the standard result files.

    returns / market_equity default to the CRSP monthly matrices from data/base.
    """
    exposure, meta = data.load_factor(project)
    settings = {**DEFAULT_REPORT, **meta.get("report", {})}
    name = meta.get("name", project)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    returns = panels.monthly_matrix("ret") if returns is None else returns
    if settings["weighting"] == "value" and market_equity is None:
        market_equity = panels.monthly_matrix("market_equity")

    winsor = settings.get("winsorize")
    cfg = FactorConfig(
        name=name,
        n_groups=settings["n_groups"],
        weighting=settings["weighting"],
        long_groups=tuple(settings["long_groups"]),
        short_groups=tuple(settings["short_groups"]),
    )
    builder = QuantileSignalFactorBuilder(
        config=cfg,
        winsorize=bool(winsor),
        winsor_lower=winsor[0] if winsor else 0.01,
        winsor_upper=winsor[1] if winsor else 0.99,
    )
    weight_base = market_equity.reindex(index=exposure.index, columns=exposure.columns) if market_equity is not None else None
    factor = builder.build_wide(exposure, weight_base)
    book = factor.book
    bt = FactorBacktester(periods_per_year=settings["periods_per_year"])

    sim = bt.simulate(book, returns)
    layer_returns = sim.to_long(builder.group_labels, name="group")[["date", "group", "ret", "n_assets"]]
    spread_returns = sim.to_long([name]).drop(columns="portfolio")
    turnover = bt.compute_turnover(book, name)
    summary = bt.summary(spread_returns["ret"], turnover, turnover_periods_per_year=settings["rebalances_per_year"]).rename(name)
    summary["description"] = meta.get("description", "")
    layer_summary = pd.DataFrame({g: bt.summary(sim.returns[g]) for g in builder.group_labels}).T.rename_axis("group")

    # IC of the (winsorised) signal of held stocks against the next period's return.
    held = book.codes >= 0
    signal = factor.signal.where(held)[held.any(axis=1)]
    next_period = _next_period_returns(returns, signal.index)
    ic = bt.ic_wide(signal, next_period).reset_index()
    r2 = bt.r2_wide(signal, next_period).reset_index()
    ic_row = pd.Series(
        {
            "mean_ic": ic["ic"].mean(),
            "mean_rank_ic": ic["rank_ic"].mean(),
            "mean_cos_ic": ic["cos_ic"].mean(),
            "ic_ir": ic["ic"].mean() / ic["ic"].std(ddof=1),
            "rank_ic_ir": ic["rank_ic"].mean() / ic["rank_ic"].std(ddof=1),
            "cos_ic_ir": ic["cos_ic"].mean() / ic["cos_ic"].std(ddof=1),
            "mean_r2": r2["r2"].mean(),
        },
        name=name,
    )
    ic_stats = ic_summary(ic.set_index("date"))
    decay = None
    if settings["rebalances_per_year"] == settings["periods_per_year"]:
        decay = ic_decay(signal, next_period, max_lag=12)
    by_year = bt.performance_by_year(spread_returns.set_index("date")["ret"])

    layer_returns.assign(factor=name).to_csv(out_dir / "layer_returns.csv", index=False)
    spread_returns.assign(factor=name).to_csv(out_dir / "spread_returns.csv", index=False)
    turnover.to_csv(out_dir / "turnover.csv", index=False)
    ic.assign(factor=name).to_csv(out_dir / "ic.csv", index=False)
    r2.assign(factor=name).to_csv(out_dir / "r2.csv", index=False)
    summary.to_frame().T.to_csv(out_dir / "summary.csv")
    ic_row.to_frame().T.to_csv(out_dir / "ic_summary.csv")
    ic_stats.to_csv(out_dir / "ic_stats.csv")
    layer_summary.to_csv(out_dir / "layer_summary.csv")
    by_year.to_csv(out_dir / "by_year.csv")
    if decay is not None:
        decay.to_csv(out_dir / "ic_decay.csv")

    if make_plots:
        _write_plots(bt, out_dir, name, layer_returns, spread_returns, ic)
    _write_markdown(out_dir, meta, summary, layer_summary, ic_stats, decay, by_year, settings)
    return {
        "summary": summary,
        "ic_summary": ic_row,
        "ic_stats": ic_stats,
        "layer_summary": layer_summary,
        "by_year": by_year,
        "ic_decay": decay,
        "simulation": sim,
        "factor": factor,
    }


def _write_plots(bt, out_dir: Path, name: str, layer_returns, spread_returns, ic) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    fig = bt.plot_layers(layer_returns)
    fig.update_layout(title=f"{name} decile layer returns")
    fig.write_html(out_dir / "layers.html", include_plotlyjs="cdn")
    fig = bt.plot_performance(spread_returns[["date", "ret"]])
    fig.update_layout(title=f"{name} long-short spread")
    fig.write_html(out_dir / "spread.html", include_plotlyjs="cdn")
    fig = go.Figure()
    for col in ["cum_ic", "cum_rank_ic", "cum_cos_ic"]:
        fig.add_trace(go.Scatter(x=pd.to_datetime(ic["date"]).to_numpy(dtype="datetime64[ns]"), y=ic[col].to_numpy(), name=col))
    fig.update_layout(template="plotly_white", height=500, title=f"{name} cumulative IC")
    fig.write_html(out_dir / "cumulative_ic.html", include_plotlyjs="cdn")


def _write_markdown(out_dir: Path, meta, summary, layer_summary, ic_stats, decay, by_year, settings) -> None:
    long_leg, short_leg = "+".join(settings["long_groups"]), "+".join(settings["short_groups"])
    head = summary[["n", "ann_return", "ann_vol", "sharpe", "t_nw", "max_drawdown", "hit_rate", "avg_turnover"]].astype(float).to_frame().T
    head.index.name = "spread"
    lines = [
        f"# {meta.get('title', meta.get('name', ''))}",
        "",
        f"- Paper: {meta.get('paper', '')}",
        f"- Signal: {meta.get('description', '')}",
        f"- Portfolios: {settings['n_groups']} {settings['weighting']}-weighted groups, {long_leg} minus {short_leg}; "
        f"winsorised at {settings.get('winsorize')}",
        f"- Sample: {summary.get('n', 0):.0f} periods; t_nw is the Newey-West t-stat of the mean",
        "",
        "## Long-short spread",
        _fmt_table(head, pct=("ann_return", "ann_vol", "max_drawdown", "hit_rate", "avg_turnover")),
        "",
        "## Decile portfolios",
        _fmt_table(layer_summary[["ann_return", "ann_vol", "sharpe", "t_nw"]], pct=("ann_return", "ann_vol")),
        "",
        "## IC (signal vs next-period return)",
        _fmt_table(ic_stats[["mean", "std", "ir", "t_nw", "pct_positive", "n_periods"]].astype(float), pct=("pct_positive",)),
        "",
    ]
    if decay is not None:
        lines += [
            "## Rank IC decay (lag = periods ahead)",
            _fmt_table(decay[["mean_rank_ic", "rank_ic_t_nw"]].T.rename_axis("lag")),
            "",
        ]
    lines += [
        "## By year",
        _fmt_table(by_year[["n", "total_return", "sharpe", "max_drawdown", "hit_rate"]], pct=("total_return", "max_drawdown", "hit_rate")),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
