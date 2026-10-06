from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factors import FactorBuildTools

from .ff3_builder import value_weighted_by_date


@dataclass(frozen=True)
class FamaFrenchConfig:
    date_col: str = "date"
    asset_col: str = "permno"
    ret_col: str = "ret"
    me_col: str = "me"
    bm_col: str = "bm"
    exchange_col: str = "exchange_code"
    rf_col: str = "rf"
    formation_col: str = "formation_date"


@dataclass
class FamaFrench3Factor:
    """Construct MKT, SMB, and HML from a labeled return panel."""

    config: FamaFrenchConfig = field(default_factory=FamaFrenchConfig)
    tools: FactorBuildTools = field(default_factory=FactorBuildTools)

    def label_characteristics(self, characteristics: pd.DataFrame) -> pd.DataFrame:
        """Assign S/B and L/M/H buckets within each formation date."""
        cfg = self.config
        required = {cfg.formation_col, cfg.asset_col, cfg.me_col, cfg.bm_col, cfg.exchange_col}
        missing = required.difference(characteristics.columns)
        if missing:
            raise ValueError(f"Missing characteristic columns: {sorted(missing)}")

        eligible = characteristics.dropna(subset=[cfg.formation_col, cfg.me_col, cfg.bm_col, cfg.exchange_col])
        if eligible.empty:
            return pd.DataFrame(columns=list(characteristics.columns) + ["size_bucket", "bm_bucket"])
        return self.tools.assign_2x3_by_date(
            eligible.sort_values(cfg.formation_col, kind="stable").reset_index(drop=True),
            date_col=cfg.formation_col,
            me_col=cfg.me_col,
            bm_col=cfg.bm_col,
            exchange_col=cfg.exchange_col,
        )

    def attach_labels(
        self,
        returns: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge formation-date labels onto the holding-period return panel."""
        cfg = self.config
        key = [cfg.formation_col, cfg.asset_col]
        cols = key + ["size_bucket", "bm_bucket"]
        if cfg.me_col in labels.columns:
            cols.append(cfg.me_col)
        return returns.merge(labels[cols], on=key, how="inner", suffixes=("", "_formation"))

    def build_portfolios(self, labeled_panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        panel = labeled_panel.copy()
        if "weight" not in panel.columns:
            if cfg.me_col not in panel.columns:
                raise ValueError("Need either weight or market equity column for value weights.")
            panel["weight"] = panel[cfg.me_col]

        return self.tools.portfolio_returns_2x3(
            panel,
            date_col=cfg.date_col,
            ret_col=cfg.ret_col,
            weight_col="weight",
        )

    def build_factors(
        self,
        labeled_panel: pd.DataFrame,
        risk_free: pd.DataFrame | None = None,
        market_returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build daily or monthly MKT, SMB, and HML.

        labeled_panel must contain date, return, market equity or weight,
        size_bucket, and bm_bucket. Returns should include dividends.
        """
        cfg = self.config
        panel = labeled_panel.copy()
        if "weight" not in panel.columns:
            panel["weight"] = panel[cfg.me_col]

        portfolios = self.build_portfolios(panel)
        wide = portfolios.pivot(index="date", columns="portfolio", values="ret")
        wide = wide.reindex(columns=["SL", "SM", "SH", "BL", "BM", "BH"])

        factors = pd.DataFrame(index=wide.index)
        small = wide[["SL", "SM", "SH"]].mean(axis=1)
        big = wide[["BL", "BM", "BH"]].mean(axis=1)
        high = wide[["SH", "BH"]].mean(axis=1)
        low = wide[["SL", "BL"]].mean(axis=1)
        factors["SMB"] = small - big
        factors["HML"] = high - low

        if market_returns is not None:
            mkt = market_returns.rename(columns={cfg.date_col: "date"}).copy()
            if "MKT" not in mkt.columns:
                raise ValueError("market_returns must contain an MKT column.")
            factors = factors.reset_index().merge(mkt[["date", "MKT"]], on="date", how="left").set_index("date")
        else:
            factors["MKT"] = value_weighted_by_date(panel, cfg.date_col, cfg.ret_col, "weight")

        if risk_free is not None:
            rf = risk_free.rename(columns={cfg.date_col: "date", cfg.rf_col: "RF"}).copy()
            factors = factors.reset_index().merge(rf[["date", "RF"]], on="date", how="left").set_index("date")
            factors["MKT_RF"] = factors["MKT"] - factors["RF"].fillna(0.0)
        else:
            factors["RF"] = np.nan
            factors["MKT_RF"] = factors["MKT"]

        ordered = ["MKT_RF", "SMB", "HML", "MKT", "RF"]
        return factors.reset_index()[["date", *ordered]].sort_values("date")

    def build(
        self,
        returns: pd.DataFrame,
        characteristics: pd.DataFrame,
        risk_free: pd.DataFrame | None = None,
        market_returns: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convenience wrapper: label, attach labels, and build factors."""
        labels = self.label_characteristics(characteristics)
        panel = self.attach_labels(returns, labels)
        factors = self.build_factors(panel, risk_free=risk_free, market_returns=market_returns)
        return factors, panel, labels

