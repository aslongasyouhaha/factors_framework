from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(CODE_ROOT))

from factors import FactorBacktester, FactorConfig, QuantileSignalFactorBuilder, WeightBook, forward_returns, to_wide


RESULTS = PROJECT_ROOT / "results"
DEMO = RESULTS / "backtest_demo"
FACTOR_NAMES = ["MKT_RF", "SMB", "HML"]
LAYERS = [f"D{i:02d}" for i in range(1, 11)]


def make_ff3_long_short_holdings(holdings: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    specs = {
        "SMB": {"SL": 1 / 3, "SM": 1 / 3, "SH": 1 / 3, "BL": -1 / 3, "BM": -1 / 3, "BH": -1 / 3},
        "HML": {"SH": 0.5, "BH": 0.5, "SL": -0.5, "BL": -0.5},
    }
    h = holdings[holdings["group"].isin(specs[factor_name])].copy()
    h["target_weight"] = h["target_weight"] * h["group"].map(specs[factor_name]).astype(float)
    h["leg"] = np.where(h["target_weight"].ge(0), "long", "short")
    h["factor_name"] = factor_name
    return h


def ff3_spread_book(holdings: pd.DataFrame, factor_name: str) -> WeightBook:
    return WeightBook.from_holdings(make_ff3_long_short_holdings(holdings, factor_name), hold_start_col=None, hold_end_col=None)


def market_turnover(monthly: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    m = monthly[monthly["ff_primary_share"]].copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"].between(start_date, end_date)].sort_values(["asset_id", "date"])
    m["lag_market_equity"] = m.groupby("asset_id")["market_equity"].shift(1)
    m = m[m["lag_market_equity"].gt(0)].copy()
    m["target_weight"] = m["lag_market_equity"] / m.groupby("date")["lag_market_equity"].transform("sum")
    book = WeightBook.from_holdings(m.rename(columns={"date": "rebalance_date"}), hold_start_col=None, hold_end_col=None)
    return FactorBacktester().compute_turnover(book)


def summary_table(bt: FactorBacktester, factor_returns: pd.DataFrame, ff3_holdings: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start_date = factor_returns["date"].min()
    end_date = factor_returns["date"].max()
    turnover_by_factor = {
        "MKT_RF": market_turnover(monthly, start_date, end_date),
        "SMB": bt.compute_turnover(ff3_spread_book(ff3_holdings, "SMB")),
        "HML": bt.compute_turnover(ff3_spread_book(ff3_holdings, "HML")),
    }
    for name in FACTOR_NAMES:
        turnover_ppy = 12 if name == "MKT_RF" else 1
        s = bt.summary(factor_returns[name], turnover_by_factor[name], turnover_periods_per_year=turnover_ppy).rename(name)
        rows.append(s)
        turnover_by_factor[name].to_csv(DEMO / f"{name.lower()}_turnover.csv", index=False)
    return pd.concat(rows, axis=1).T


def write_factor_plots(bt: FactorBacktester, factor_returns: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plotting.") from exc

    for name in FACTOR_NAMES:
        port = factor_returns[["date", name]].rename(columns={name: "ret"})
        fig = bt.plot_performance(port)
        fig.update_layout(title=f"{name} performance")
        fig.write_html(DEMO / f"{name.lower()}_performance.html", include_plotlyjs="cdn")
        port.to_csv(DEMO / f"{name.lower()}_returns.csv", index=False)

    fig = go.Figure()
    for name in FACTOR_NAMES:
        r = factor_returns[["date", name]].dropna().copy()
        r["cum_return"] = (1.0 + r[name]).cumprod() - 1.0
        fig.add_trace(go.Scatter(x=pd.to_datetime(r["date"]).to_numpy(dtype="datetime64[ns]"), y=r["cum_return"].to_numpy(), name=name))
    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(template="plotly_white", height=560, title="FF3 factor cumulative returns")
    fig.write_html(DEMO / "ff3_factor_curves.html", include_plotlyjs="cdn")


def decile_factor(chars: pd.DataFrame, signal_col: str, name: str, direction: str):
    """Value-weighted, winsorised deciles of a positive characteristic, held for one year."""
    c = chars.dropna(subset=[signal_col, "market_equity"])
    c = c[c[signal_col].gt(0) & c["market_equity"].gt(0)]
    signal = to_wide(c, signal_col, date_col="rebalance_date")
    market_equity = to_wide(c, "market_equity", date_col="rebalance_date")
    long, short = (("D01",), ("D10",)) if direction == "low_minus_high" else (("D10",), ("D01",))
    cfg = FactorConfig(name=name, n_groups=10, weighting="value", long_groups=long, short_groups=short)
    factor = QuantileSignalFactorBuilder(config=cfg, winsorize=True, winsor_lower=0.01, winsor_upper=0.99).build_wide(
        signal,
        market_equity,
        start_dates=signal.index + pd.offsets.Day(1),
        end_dates=signal.index + pd.DateOffset(years=1),
    )
    return factor, signal, market_equity


def decile_summary(bt: FactorBacktester, layer_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in layer_returns.groupby("group", observed=True):
        s = bt.summary(g["ret"])
        s["group"] = group
        rows.append(s)
    return pd.DataFrame(rows).set_index("group").sort_index()


def long_short_from_layers(layer_returns: pd.DataFrame, high: str = "D10", low: str = "D01", direction: str = "high_minus_low") -> pd.DataFrame:
    wide = layer_returns.pivot(index="date", columns="group", values="ret")
    ret = wide[low] - wide[high] if direction == "low_minus_high" else wide[high] - wide[low]
    return ret.rename("ret").reset_index()


def write_ic_outputs(bt: FactorBacktester, name: str, factor, returns: pd.DataFrame) -> pd.Series:
    """IC of the winsorised signal of held stocks against the following month's return."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plotting.") from exc

    held = factor.book.codes >= 0
    signal = factor.signal.where(held)[held.any(axis=1)]
    next_month = forward_returns(returns, signal.index, end_dates=signal.index + pd.offsets.MonthEnd(1))
    ic = bt.ic_wide(signal, next_month).reset_index()
    r2 = bt.r2_wide(signal, next_month).reset_index()
    ic.to_csv(DEMO / f"{name}_winsor_ic.csv", index=False)
    r2.to_csv(DEMO / f"{name}_winsor_r2.csv", index=False)

    fig = go.Figure()
    for col in ["cum_ic", "cum_rank_ic", "cum_cos_ic"]:
        fig.add_trace(go.Scatter(x=pd.to_datetime(ic["date"]).to_numpy(dtype="datetime64[ns]"), y=ic[col].to_numpy(), name=col))
    fig.update_layout(template="plotly_white", height=500, title=f"{name} cumulative IC")
    fig.write_html(DEMO / f"{name}_winsor_cumulative_ic.html", include_plotlyjs="cdn")

    return pd.Series({
        "mean_ic": ic["ic"].mean(),
        "mean_rank_ic": ic["rank_ic"].mean(),
        "mean_cos_ic": ic["cos_ic"].mean(),
        "ic_ir": ic["ic"].mean() / ic["ic"].std(ddof=1),
        "rank_ic_ir": ic["rank_ic"].mean() / ic["rank_ic"].std(ddof=1),
        "cos_ic_ir": ic["cos_ic"].mean() / ic["cos_ic"].std(ddof=1),
        "mean_r2": r2["r2"].mean(),
    }, name=name)


def build_decile_demo(bt: FactorBacktester) -> None:
    chars = pd.read_parquet(RESULTS / "ff3_characteristics_2017_2025.parquet")
    monthly = pd.read_parquet(RESULTS / "crsp_monthly_permco_2017_2025.parquet")
    returns = to_wide(monthly[monthly["ff_primary_share"]], "ret")

    specs = [
        ("bm", "bm", "BM winsorized deciles", "high_minus_low"),
        ("size_me", "market_equity", "Size/ME winsorized deciles", "low_minus_high"),
    ]
    ls_rows = []
    ic_rows = []
    for name, col, title, direction in specs:
        factor, raw_signal, market_equity = decile_factor(chars, col, name, direction)
        sim = bt.simulate(factor.book, returns)
        layers = sim.to_long(LAYERS, name="group")[["date", "group", "ret", "n_assets"]]
        summary = decile_summary(bt, layers)
        ls = long_short_from_layers(layers, direction=direction)
        spread_turnover = bt.compute_turnover(factor.book, name)
        ls_summary = bt.summary(ls["ret"], spread_turnover, turnover_periods_per_year=1).rename(name)
        ls_rows.append(ls_summary)
        ic_rows.append(write_ic_outputs(bt, name, factor, returns))

        holdings = factor.book.to_long(extras={"signal_raw": raw_signal, "signal_winsor": factor.signal, "market_equity": market_equity})
        holdings.insert(0, "factor_name", name)
        decile_holdings = holdings[holdings["portfolio"].ne(name)].rename(columns={"portfolio": "group"})
        spread_holdings = holdings[holdings["portfolio"].eq(name)].drop(columns="portfolio")
        spread_holdings.insert(2, "leg", np.where(spread_holdings["target_weight"].ge(0), "long", "short"))
        decile_holdings.to_parquet(DEMO / f"{name}_winsor_decile_holdings.parquet", index=False, compression="zstd")
        spread_holdings.to_parquet(DEMO / f"{name}_winsor_decile_spread_holdings.parquet", index=False, compression="zstd")
        spread_turnover.to_csv(DEMO / f"{name}_winsor_decile_turnover.csv", index=False)
        layers.to_csv(DEMO / f"{name}_winsor_decile_layer_returns.csv", index=False)
        summary.to_csv(DEMO / f"{name}_winsor_decile_summary.csv")
        ls.to_csv(DEMO / f"{name}_winsor_decile_long_short.csv", index=False)

        fig = bt.plot_layers(layers)
        fig.update_layout(title=title)
        fig.write_html(DEMO / f"{name}_winsor_decile_layers.html", include_plotlyjs="cdn")

        ls_fig = bt.plot_performance(ls)
        ls_title = f"{name} decile spread ({'D01-D10' if direction == 'low_minus_high' else 'D10-D01'})"
        ls_fig.update_layout(title=ls_title)
        ls_fig.write_html(DEMO / f"{name}_winsor_decile_long_short.html", include_plotlyjs="cdn")

    pd.concat(ls_rows, axis=1).T.to_csv(DEMO / "winsor_decile_long_short_summary.csv")
    pd.concat(ic_rows, axis=1).T.to_csv(DEMO / "winsor_signal_ic_r2_summary.csv")


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    bt = FactorBacktester(periods_per_year=12)
    ff3 = pd.read_csv(RESULTS / "ff3_factors_2017_2025.csv", parse_dates=["date"])
    holdings = pd.read_parquet(RESULTS / "ff3_holdings_2017_2025.parquet")
    monthly = pd.read_parquet(RESULTS / "crsp_monthly_permco_2017_2025.parquet")
    summary = summary_table(bt, ff3, holdings, monthly)
    summary.to_csv(DEMO / "ff3_factor_summary.csv")
    write_factor_plots(bt, ff3)
    build_decile_demo(bt)
    cols = ["ann_return", "ann_vol", "sharpe", "total_return", "max_drawdown", "hit_rate", "avg_turnover"]
    print("Wrote demo outputs under", DEMO)
    print(summary[cols].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
