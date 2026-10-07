"""Load a comparable monthly return panel from the local factor projects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FACTORS = ROOT / "factors"
DATA_FACTORS = ROOT / "data" / "factors"


_SPECIAL_COLUMNS = {
    "daniel_hirshleifer_sun_2020_pead": ["PEAD"],
    "fama_french_2015_cma": ["CMA"],
    "fama_french_2015_rmw": ["RMW"],
    "frazzini_pedersen_2014_bab": ["BAB"],
    "hirshleifer_et_al_2004_net_operating_assets": ["LOW_NOA"],
    "piotroski_2000_fscore": ["F_SCORE"],
}


def _monthly(series: pd.Series) -> pd.Series:
    """Compound a dated return series to calendar months."""
    x = pd.DataFrame({"ret": pd.to_numeric(series, errors="coerce")}, index=pd.to_datetime(series.index))
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x["month"] = x.index.to_period("M").to_timestamp("M")
    return x.groupby("month")["ret"].apply(lambda z: np.prod(1.0 + z) - 1.0)


def local_factor_returns(min_observations: int = 120) -> pd.DataFrame:
    """Return one monthly series per implemented factor, without duplicate MKT-RF.

    Paper-specific factor returns take precedence over generic decile spreads.  The
    q-factor project contributes r_ME, r_IA and r_ROE; its duplicate MKT-RF is omitted.
    """
    series: dict[str, pd.Series] = {}
    ff3 = FACTORS / "fama_french_1993_three_factors" / "result" / "ff3_returns.csv"
    if ff3.exists():
        x = pd.read_csv(ff3, parse_dates=["date"]).set_index("date")
        for col in ["MKT_RF", "SMB", "HML"]:
            series[col] = _monthly(x[col])

    q_file = DATA_FACTORS / "hou_xue_zhang_2015_q_factor" / "factor_returns.parquet"
    if q_file.exists():
        x = pd.read_parquet(q_file).set_index("date")
        for col in ["r_ME", "r_IA", "r_ROE"]:
            if col in x:
                series[col] = _monthly(x[col])

    skip = {"fama_french_1993_three_factors", "hou_xue_zhang_2015_q_factor"}
    for project in sorted(p for p in FACTORS.iterdir() if p.is_dir() and p.name not in skip):
        result = project / "result"
        specific = result / "factor_returns.csv"
        spread = result / "spread_returns.csv"
        if specific.exists() and project.name in _SPECIAL_COLUMNS:
            x = pd.read_csv(specific, parse_dates=["date"]).set_index("date")
            for col in _SPECIAL_COLUMNS[project.name]:
                if col in x:
                    series[col] = _monthly(x[col])
        elif spread.exists():
            x = pd.read_csv(spread, parse_dates=["date"])
            if x.empty:
                continue
            label = str(x["factor"].dropna().iloc[0]) if "factor" in x else project.name
            label = label if label not in series else project.name
            series[label] = _monthly(x.set_index("date")["ret"])

    panel = pd.concat(series, axis=1).sort_index()
    keep = panel.count() >= min_observations
    return panel.loc[:, keep].astype(float)


DEFAULT_TEST_ASSET_PROJECTS = (
    "banz_1981_size",
    "fama_french_1992_book_to_market",
    "jegadeesh_titman_1993_momentum",
    "ang_hodrick_xing_zhang_2006_volatility",
    "novy_marx_2013_gross_profitability",
    "cooper_gulen_schill_2008_asset_growth",
)


def characteristic_test_assets(
    projects: tuple[str, ...] = DEFAULT_TEST_ASSET_PROJECTS,
) -> pd.DataFrame:
    """Load ten value-weighted characteristic portfolios from each project."""
    out: dict[str, pd.Series] = {}
    for project in projects:
        path = FACTORS / project / "result" / "layer_returns.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run the report first: {path}")
        x = pd.read_csv(path, parse_dates=["date"])
        for group, g in x.groupby("group"):
            out[f"{project}:{group}"] = _monthly(g.set_index("date")["ret"])
    return pd.concat(out, axis=1).sort_index().astype(float)


def ff49_industry_returns() -> pd.DataFrame:
    """Construct lagged-market-cap-weighted FF49 industry portfolios from CRSP."""
    crsp = pd.read_parquet(
        ROOT / "data" / "base" / "crsp_monthly.parquet",
        columns=["date", "asset_id", "ret", "market_equity", "siccd", "ff_primary_share"],
    )
    ranges = pd.read_parquet(ROOT / "data" / "base" / "ff49_sic_ranges.parquet")
    sic_map = np.full(10000, "", dtype=object)
    for row in ranges.itertuples(index=False):
        low, high = max(int(row.sic_low), 0), min(int(row.sic_high), 9999)
        sic_map[low : high + 1] = row.industry
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp = crsp[crsp["ff_primary_share"].fillna(False)].sort_values(["asset_id", "date"])
    crsp["lag_me"] = crsp.groupby("asset_id")["market_equity"].shift()
    sic = pd.to_numeric(crsp["siccd"], errors="coerce")
    valid_sic = sic.between(0, 9999, inclusive="both")
    crsp["industry"] = ""
    crsp.loc[valid_sic, "industry"] = sic_map[sic[valid_sic].astype(int).to_numpy()]
    crsp = crsp[
        crsp["industry"].ne("")
        & crsp["ret"].notna()
        & crsp["lag_me"].gt(0)
    ].copy()
    crsp["weighted_ret"] = crsp["ret"] * crsp["lag_me"]
    grouped = crsp.groupby(["date", "industry"])[["weighted_ret", "lag_me"]].sum()
    grouped["ret"] = grouped["weighted_ret"] / grouped["lag_me"]
    panel = grouped["ret"].unstack("industry").sort_index()
    ordered = sorted(panel.columns)
    return panel[ordered].astype(float)
