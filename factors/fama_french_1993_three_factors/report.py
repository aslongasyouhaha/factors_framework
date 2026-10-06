"""FF3 result page: factor performance and turnover, and a comparison with Kenneth French's data."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "framework"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorlab import FactorBacktester, WeightBook, data, panels  # noqa: E402
from factorlab.report import _fmt_table  # noqa: E402

RESULT = HERE / "result"
FACTOR_NAMES = ["MKT_RF", "SMB", "HML"]
SPREADS = {
    "SMB": {"SL": 1 / 3, "SM": 1 / 3, "SH": 1 / 3, "BL": -1 / 3, "BM": -1 / 3, "BH": -1 / 3},
    "HML": {"SH": 0.5, "BH": 0.5, "SL": -0.5, "BL": -0.5},
}


def spread_book(holdings: pd.DataFrame, factor_name: str) -> WeightBook:
    """Target weights of SMB or HML at each June formation."""
    h = holdings[holdings["group"].isin(SPREADS[factor_name])].copy()
    h["target_weight"] = h["target_weight"] * h["group"].map(SPREADS[factor_name]).astype(float)
    return WeightBook.from_holdings(h, hold_start_col=None, hold_end_col=None)


def market_turnover(monthly: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Turnover of the value-weighted market portfolio, re-weighted monthly by lagged market equity."""
    m = monthly[monthly["ff_primary_share"]].copy()
    m = m[m["date"].between(start_date, end_date)].sort_values(["asset_id", "date"])
    m["lag_market_equity"] = m.groupby("asset_id")["market_equity"].shift(1)
    m = m[m["lag_market_equity"].gt(0)].copy()
    m["target_weight"] = m["lag_market_equity"] / m.groupby("date")["lag_market_equity"].transform("sum")
    book = WeightBook.from_holdings(m.rename(columns={"date": "rebalance_date"}), hold_start_col=None, hold_end_col=None)
    return FactorBacktester().compute_turnover(book)


def kenneth_french_monthly() -> pd.DataFrame:
    """French's monthly MKT_RF, SMB, HML, RF (built into data/base by pipeline/build_rf.py)."""
    kf = data.load_base("french_factors_monthly")
    kf["date"] = pd.to_datetime(kf["date"])
    return kf


def compare_with_kenneth_french(ours: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["MKT_RF", "SMB", "HML", "RF"]
    merged = ours.rename(columns={c: f"{c}_ours" for c in cols}).merge(
        kenneth_french_monthly().rename(columns={c: f"{c}_ff" for c in cols}), on="date", how="inner"
    )
    rows = []
    for c in cols:
        diff = merged[f"{c}_ours"] - merged[f"{c}_ff"]
        merged[f"{c}_diff"] = diff
        rows.append(
            {
                "factor": c,
                "n": len(diff),
                "corr": merged[f"{c}_ours"].corr(merged[f"{c}_ff"]),
                "ann_mean_diff_pct": 12 * diff.mean() * 100,
                "tracking_error_pct": np.sqrt(12) * diff.std(ddof=1) * 100,
                "mae_pct": diff.abs().mean() * 100,
                "max_abs_diff_pct": diff.abs().max() * 100,
            }
        )
    return merged, pd.DataFrame(rows).set_index("factor")


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    project = HERE.name
    bt = FactorBacktester(periods_per_year=12)
    ff3 = data.load_factor_table(project, "returns")
    ff3["date"] = pd.to_datetime(ff3["date"])
    holdings = data.load_factor_table(project, "holdings")
    monthly = panels.crsp_monthly(primary_only=False)

    turnover = {
        "MKT_RF": market_turnover(monthly, ff3["date"].min(), ff3["date"].max()),
        "SMB": bt.compute_turnover(spread_book(holdings, "SMB")),
        "HML": bt.compute_turnover(spread_book(holdings, "HML")),
    }
    rows = []
    for name in FACTOR_NAMES:
        rows.append(bt.summary(ff3[name], turnover[name], turnover_periods_per_year=12 if name == "MKT_RF" else 1).rename(name))
        turnover[name].to_csv(RESULT / f"{name.lower()}_turnover.csv", index=False)
    summary = pd.concat(rows, axis=1).T
    summary.to_csv(RESULT / "factor_summary.csv")
    ff3.to_csv(RESULT / "ff3_returns.csv", index=False)
    by_year = bt.performance_by_year(ff3.set_index("date")[FACTOR_NAMES])
    by_year.to_csv(RESULT / "by_year.csv")

    monthly_cmp, kf_summary = compare_with_kenneth_french(ff3[["date", "MKT_RF", "SMB", "HML", "MKT", "RF"]])
    monthly_cmp.to_csv(RESULT / "vs_kenneth_french_monthly.csv", index=False)
    kf_summary.to_csv(RESULT / "vs_kenneth_french_summary.csv")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for name in FACTOR_NAMES:
            r = ff3[["date", name]].dropna()
            fig.add_trace(go.Scatter(x=r["date"].to_numpy(dtype="datetime64[ns]"), y=((1 + r[name]).cumprod() - 1).to_numpy(), name=name))
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(template="plotly_white", height=560, title="FF3 factor cumulative returns")
        fig.write_html(RESULT / "ff3_factor_curves.html", include_plotlyjs="cdn")
        for name in FACTOR_NAMES:
            fig = bt.plot_performance(ff3[["date", name]].rename(columns={name: "ret"}))
            fig.update_layout(title=f"{name} performance")
            fig.write_html(RESULT / f"{name.lower()}_performance.html", include_plotlyjs="cdn")
    except ImportError:
        pass

    cols = ["ann_return", "ann_vol", "sharpe", "t_nw", "max_drawdown", "hit_rate", "avg_turnover"]
    md = [
        "# Fama-French three factors",
        "",
        "- Paper: Fama & French (1993), Common risk factors in the returns on stocks and bonds, JFE 33(1)",
        f"- Sample: {ff3['date'].min():%Y-%m} to {ff3['date'].max():%Y-%m}, {len(ff3)} months",
        "",
        "## Factor performance",
        _fmt_table(summary[cols].rename_axis("factor"), pct=("ann_return", "ann_vol", "max_drawdown", "hit_rate", "avg_turnover")),
        "",
        "## Comparison with Kenneth French's factors (ours - French, monthly)",
        _fmt_table(kf_summary[["n", "corr", "ann_mean_diff_pct", "tracking_error_pct", "mae_pct", "max_abs_diff_pct"]].astype(float)),
        "",
        "## By year",
        _fmt_table(by_year[["total_return", "sharpe", "max_drawdown"]], pct=("total_return", "max_drawdown")),
        "",
    ]
    (RESULT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print(kf_summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
