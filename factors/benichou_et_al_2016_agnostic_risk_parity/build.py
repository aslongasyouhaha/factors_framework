"""Apply Agnostic Risk Parity to the local factor return universe."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from factorlab.factor_universe import local_factor_returns  # noqa: E402
from factorlab.risk_parity import agnostic_risk_parity_backtest  # noqa: E402


if __name__ == "__main__":
    factor_returns = local_factor_returns()
    result = agnostic_risk_parity_backtest(factor_returns)
    out = ROOT / "data" / "factors" / HERE.name
    out.mkdir(parents=True, exist_ok=True)
    factor_returns.reset_index(names="date").to_parquet(out / "factor_returns.parquet", index=False)
    result.returns.reset_index(names="date").to_parquet(out / "portfolio_returns.parquet", index=False)
    result.gross_returns.reset_index(names="date").to_parquet(out / "gross_returns.parquet", index=False)
    result.turnover.to_parquet(out / "turnover.parquet", index=False)
    result.weights.to_parquet(out / "weights.parquet", index=False)
    result.eigenrisk.to_parquet(out / "eigenrisk.parquet", index=False)
    meta = {
        "paper": "Benichou et al. (2016), Agnostic Risk Parity",
        "application": "Local monthly factor zoo, 12-month trend signal",
        "covariance": "60-month rolling Ledoit-Wolf cleaning",
        "target_vol": 0.10,
        "cost_bps": 10.0,
        "n_factors": factor_returns.shape[1],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {factor_returns.shape[1]}-factor ARP experiment to {out}")
