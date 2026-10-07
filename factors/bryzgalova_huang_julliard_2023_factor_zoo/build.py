"""Run the BSftFT continuous spike-and-slab SDF on the local factor zoo."""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from factorlab.bayesian_sdf import continuous_spike_slab_sdf, psi_from_prior_sharpe  # noqa: E402
from factorlab.factor_universe import ff49_industry_returns, local_factor_returns  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=3000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202310)
    args = parser.parse_args()

    factors = local_factor_returns()
    assets = ff49_industry_returns()
    common = pd.concat({"f": factors, "r": assets}, axis=1).dropna()
    factors, assets = common["f"], common["r"]
    psi = psi_from_prior_sharpe(factors, assets, prior_sharpe=0.10)
    result = continuous_spike_slab_sdf(
        factors,
        assets,
        simulations=args.simulations,
        burn=args.burn,
        psi0=psi,
        seed=args.seed,
    )

    out = ROOT / "data" / "factors" / HERE.name
    out.mkdir(parents=True, exist_ok=True)
    factors.reset_index(names="date").to_parquet(out / "candidate_factors.parquet", index=False)
    assets.reset_index(names="date").to_parquet(out / "test_assets.parquet", index=False)
    result.inclusion_probability.rename_axis("factor").reset_index().to_parquet(out / "inclusion_probability.parquet", index=False)
    result.lambda_mean.rename_axis("factor").reset_index().to_parquet(out / "lambda_mean.parquet", index=False)
    result.lambda_quantiles.rename_axis("factor").reset_index().to_parquet(out / "lambda_quantiles.parquet", index=False)
    result.model_size.rename_axis("model_size").reset_index().to_parquet(out / "model_size.parquet", index=False)
    result.bma_sdf.rename_axis("date").reset_index().to_parquet(out / "bma_sdf.parquet", index=False)
    pd.DataFrame(result.gamma_path, columns=factors.columns).to_parquet(out / "gamma_draws.parquet", index=False)
    meta = {
        "paper": "Bryzgalova, Huang & Julliard (2023), Bayesian Solutions for the Factor Zoo",
        "method": "continuous spike-and-slab BMA-SDF, OLS with common intercept",
        "prior_sharpe": 0.10,
        "psi": psi,
        "simulations": args.simulations,
        "burn": args.burn,
        "seed": args.seed,
        "n_factors": factors.shape[1],
        "n_test_assets": assets.shape[1],
        "n_months": len(common),
        "scope": "local-data application; official method, not the paper's original 51-factor input panel",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote Bayesian factor-zoo experiment to {out}")
