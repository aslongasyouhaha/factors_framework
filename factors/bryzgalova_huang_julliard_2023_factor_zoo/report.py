"""Report local BSftFT posterior selection and pricing diagnostics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


if __name__ == "__main__":
    source = ROOT / "data" / "factors" / HERE.name
    out = HERE / "result"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    inclusion = pd.read_parquet(source / "inclusion_probability.parquet").set_index("factor")
    lambdas = pd.read_parquet(source / "lambda_mean.parquet").set_index("factor")
    quantiles = pd.read_parquet(source / "lambda_quantiles.parquet").set_index("factor")
    ranking = inclusion.join(lambdas).join(quantiles).sort_values("inclusion_probability", ascending=False)
    model_size = pd.read_parquet(source / "model_size.parquet").set_index("model_size")
    factors = pd.read_parquet(source / "candidate_factors.parquet").set_index("date")
    assets = pd.read_parquet(source / "test_assets.parquet").set_index("date")
    sdf = pd.read_parquet(source / "bma_sdf.parquet").set_index("date")["BMA_SDF"]
    common = assets.join(sdf).dropna()
    raw_error = common[assets.columns].mean(axis=0)
    bma_error = common[assets.columns].mul(common["BMA_SDF"], axis=0).mean(axis=0)
    raw_error_centered = raw_error - raw_error.mean()
    bma_error_centered = bma_error - bma_error.mean()
    pricing = pd.DataFrame(
        {
            "raw_mean": raw_error,
            "bma_pricing_error": bma_error,
            "raw_error_ex_intercept": raw_error_centered,
            "bma_error_ex_intercept": bma_error_centered,
        }
    )
    gamma = pd.read_parquet(source / "gamma_draws.parquet").iloc[meta["burn"] :]
    half = len(gamma) // 2
    split_difference = (gamma.iloc[:half].mean() - gamma.iloc[half:].mean()).abs()
    raw_rmse = np.sqrt(np.mean(raw_error_centered**2))
    bma_rmse = np.sqrt(np.mean(bma_error_centered**2))
    diagnostics = pd.Series(
        {
            "raw_ann_rmse_ex_intercept": raw_rmse * 12,
            "bma_ann_rmse_ex_intercept": bma_rmse * 12,
            "pricing_error_reduction": 1 - bma_rmse / raw_rmse,
            "mean_model_size": float((model_size.index * model_size["probability"]).sum()),
            "bma_sdf_monthly_vol": sdf.std(ddof=1),
            "max_split_inclusion_difference": split_difference.max(),
        }
    )
    ranking.to_csv(out / "factor_posterior.csv")
    model_size.to_csv(out / "model_size.csv")
    pricing.to_csv(out / "pricing_errors.csv")
    diagnostics.to_frame("value").to_csv(out / "diagnostics.csv")
    sdf.rename("BMA_SDF").to_csv(out / "bma_sdf.csv")
    md = [
        "# Bayesian Solutions for the Factor Zoo — local application",
        "",
        f"- Candidate factors: {meta['n_factors']}; FF49 industry test portfolios: {meta['n_test_assets']}; months: {meta['n_months']}.",
        f"- MCMC: {meta['simulations']} draws, {meta['burn']} burn-in; prior model Sharpe ratio 0.10.",
        "- Estimator: official continuous spike-and-slab B-SDF equations, OLS and common intercept.",
        "- Scope: the authors' method is reproduced on local inputs; this does not claim to match their original 51-factor empirical tables.",
        "",
        "## Highest posterior inclusion probabilities",
        "",
        ranking.head(15).to_markdown(floatfmt=".4f"),
        "",
        "## Pricing diagnostics",
        "",
        diagnostics.to_frame("value").to_markdown(floatfmt=".4f"),
        "",
        "## Posterior model size",
        "",
        model_size.to_markdown(floatfmt=".4f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(ranking.head(15).to_string())
    print(diagnostics.to_string())
