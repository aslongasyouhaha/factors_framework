"""Create the ARP comparison report."""
from pathlib import Path
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from factorlab.evaluation import performance_by_year, performance_summary  # noqa: E402


if __name__ == "__main__":
    source = ROOT / "data" / "factors" / HERE.name
    out = HERE / "result"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    returns = pd.read_parquet(source / "portfolio_returns.parquet").set_index("date")
    gross = pd.read_parquet(source / "gross_returns.parquet").set_index("date")
    turnover = pd.read_parquet(source / "turnover.parquet")
    eigenrisk = pd.read_parquet(source / "eigenrisk.parquet")
    rows = []
    for strategy in returns:
        s = performance_summary(returns[strategy], 12)
        s["gross_ann_return"] = performance_summary(gross[strategy], 12)["ann_return"]
        s["avg_monthly_turnover"] = turnover.loc[turnover.strategy == strategy, "turnover"].mean()
        s["ann_cost"] = turnover.loc[turnover.strategy == strategy, "cost"].mean() * 12
        rows.append(s.rename(strategy))
    summary = pd.DataFrame(rows)
    by_year = pd.concat(
        {strategy: performance_by_year(returns[strategy], 12) for strategy in returns},
        names=["strategy", "year"],
    )
    eigen_hhi = eigenrisk.groupby(["strategy", "date"])["risk_share"].apply(lambda x: float((x**2).sum()))
    eigen_hhi_summary = eigen_hhi.groupby("strategy").agg(["mean", "median", "max"])
    summary.to_csv(out / "summary.csv")
    by_year.to_csv(out / "by_year.csv")
    returns.reset_index().to_csv(out / "portfolio_returns.csv", index=False)
    turnover.to_csv(out / "turnover.csv", index=False)
    eigenrisk.to_csv(out / "arp_eigenrisk.csv", index=False)
    eigen_hhi_summary.to_csv(out / "eigenrisk_hhi.csv")
    cols = ["n", "ann_return", "ann_vol", "sharpe", "max_drawdown", "gross_ann_return", "avg_monthly_turnover", "ann_cost"]
    md = [
        "# Agnostic Risk Parity on the local factor zoo",
        "",
        f"- Universe: {meta['n_factors']} locally constructed factor return series.",
        "- Signal: trailing 12-month mean of volatility-normalized factor returns.",
        "- Covariance: trailing 60 months (minimum 36); Ledoit-Wolf cleaning.",
        "- All strategies use a 10% ex-ante annual volatility target and 10 bp per unit turnover.",
        "- ARP uses cleaned C^(-1/2)p; Markowitz uses C^(-1)p.",
        "- This is an algorithm application, not a reproduction of the paper's proprietary 110-futures dataset.",
        "",
        "## Net performance",
        "",
        summary[cols].to_markdown(floatfmt=".4f"),
        "",
        "## Eigenrisk diagnostic",
        "",
        eigen_hhi_summary.to_markdown(floatfmt=".4f"),
        "",
        "Lower HHI means realized risk is spread across more covariance eigenmodes.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary[cols].to_string())
