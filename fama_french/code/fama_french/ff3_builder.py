from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factors import Factor, FactorBuilder, FactorConfig, FactorBuildTools, LayerBook, simulate, to_wide


def value_weighted_by_date(panel: pd.DataFrame, date_col: str, ret_col: str, weight_col: str) -> pd.Series:
    """Per-date value-weighted return over rows with a return and a positive weight."""
    ret = pd.to_numeric(panel[ret_col], errors="coerce")
    weight = pd.to_numeric(panel[weight_col], errors="coerce")
    usable = ret.notna() & weight.notna() & weight.gt(0)
    parts = pd.DataFrame({"num": (ret * weight).where(usable), "den": weight.where(usable)})
    sums = parts.groupby(panel[date_col]).sum(min_count=1)
    return (sums["num"] / sums["den"]).rename_axis("date")


@dataclass
class FamaFrench3Builder(FactorBuilder):
    """Build Fama-French 2x3 size-value portfolio holdings and factor returns."""

    config: FactorConfig = field(
        default_factory=lambda: FactorConfig(
            name="FamaFrench3",
            frequency="monthly",
            weighting="value",
            asset_col="asset_id",
            rebalance_col="rebalance_date",
        )
    )
    tools: FactorBuildTools = field(default_factory=FactorBuildTools)
    required_inputs: tuple[str, ...] = ("characteristics",)

    def prepare_universe(self, data: dict[str, object]) -> pd.DataFrame:
        characteristics = data["characteristics"]
        if not isinstance(characteristics, pd.DataFrame):
            raise TypeError("characteristics must be a DataFrame")
        c = characteristics.copy()
        if isinstance(c.index, pd.MultiIndex):
            c = c.reset_index()
        c = c.rename(columns={"date": "rebalance_date", "permno": "asset_id", "me": "market_equity"})
        required = {"rebalance_date", "asset_id", "market_equity", "bm", "exchange_code"}
        missing = required.difference(c.columns)
        if missing:
            raise ValueError(f"Missing FF3 characteristic columns: {sorted(missing)}")
        c["rebalance_date"] = pd.to_datetime(c["rebalance_date"])
        c["market_equity"] = pd.to_numeric(c["market_equity"], errors="coerce")
        c["bm"] = pd.to_numeric(c["bm"], errors="coerce")
        c["exchange_code"] = pd.to_numeric(c["exchange_code"], errors="coerce")
        c = c.dropna(subset=["rebalance_date", "asset_id", "market_equity", "bm", "exchange_code"])
        c = c[(c["market_equity"] > 0) & (c["bm"] > 0)]
        c["hold_start"] = c["rebalance_date"] + pd.offsets.Day(1)
        c["hold_end"] = c["rebalance_date"] + pd.DateOffset(years=1)
        return c

    def compute_signal(self, universe: pd.DataFrame) -> pd.DataFrame:
        out = universe.copy()
        out["signal_value"] = out["bm"]
        return out

    def assign_groups(self, signal_panel: pd.DataFrame) -> pd.DataFrame:
        if signal_panel.empty:
            return pd.DataFrame()
        labeled = self.tools.assign_2x3_by_date(
            signal_panel.sort_values("rebalance_date", kind="stable").reset_index(drop=True),
            date_col="rebalance_date",
            me_col="market_equity",
            bm_col="bm",
            exchange_col="exchange_code",
        )
        labeled["group"] = labeled["size_bucket"].astype(str) + labeled["bm_bucket"].astype(str)
        labeled["leg"] = "portfolio"
        return labeled

    def compute_weights(self, grouped_panel: pd.DataFrame) -> pd.DataFrame:
        if grouped_panel.empty:
            return pd.DataFrame(columns=["rebalance_date", "hold_start", "hold_end", "asset_id", "group", "leg", "target_weight"])
        out = grouped_panel.copy()
        denom = out.groupby(["rebalance_date", "group"])["market_equity"].transform("sum")
        out["target_weight"] = out["market_equity"] / denom
        cols = [
            "rebalance_date",
            "hold_start",
            "hold_end",
            "asset_id",
            "group",
            "leg",
            "target_weight",
            "signal_value",
            "market_equity",
            "bm",
            "exchange_code",
        ]
        return out[cols].sort_values(["rebalance_date", "group", "asset_id"])

    def build(self, data: dict[str, object]) -> Factor:
        factor = super().build(data)
        returns = data.get("returns")
        risk_free = data.get("risk_free")
        market_returns = data.get("market_returns")
        if isinstance(returns, pd.DataFrame):
            factor.returns = self.factor_returns(factor.holdings, returns, risk_free=risk_free, market_returns=market_returns)
        return factor

    def group_returns(
        self,
        holdings: pd.DataFrame,
        returns: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
        ret_col: str = "ret",
    ) -> pd.DataFrame:
        r = returns.copy()
        if isinstance(r.index, pd.MultiIndex):
            r = r.reset_index()
        r = r.rename(columns={"permno": asset_col, "me": "market_equity"})
        r[date_col] = pd.to_datetime(r[date_col])
        if "market_equity" in r.columns:
            r = r.sort_values([asset_col, date_col])
            r["lag_market_equity"] = r.groupby(asset_col)["market_equity"].shift(1)

        if holdings.empty:
            return pd.DataFrame(columns=[date_col, "group", "ret"])
        # Each group is re-weighted every period by lagged market equity (FF value weights).
        book = LayerBook.from_holdings(holdings, group_col="group", asset_col="asset_id")
        wide_ret = to_wide(r, ret_col, date_col=date_col, asset_col=asset_col)
        base = None
        if "lag_market_equity" in r.columns and r["lag_market_equity"].notna().any():
            base = to_wide(r, "lag_market_equity", date_col=date_col, asset_col=asset_col)
        sim = simulate(book, wide_ret, weight_base=base)
        out = pd.concat(
            [sim.returns.stack().rename("ret"), sim.n_assets.stack().rename("n_assets").astype(int)], axis=1
        )
        out = out.rename_axis([date_col, "group"]).reset_index()
        return out[out["n_assets"].gt(0)].sort_values([date_col, "group"]).reset_index(drop=True)

    def factor_returns(
        self,
        holdings: pd.DataFrame,
        returns: pd.DataFrame,
        risk_free: pd.DataFrame | None = None,
        market_returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        group_ret = self.group_returns(holdings, returns)
        if group_ret.empty:
            return pd.DataFrame(columns=["date", "MKT_RF", "SMB", "HML", "MKT", "RF"])
        wide = group_ret.pivot(index="date", columns="group", values="ret")
        wide = wide.reindex(columns=["SL", "SM", "SH", "BL", "BM", "BH"])

        factors = pd.DataFrame(index=wide.index)
        factors["SMB"] = wide[["SL", "SM", "SH"]].mean(axis=1) - wide[["BL", "BM", "BH"]].mean(axis=1)
        factors["HML"] = wide[["SH", "BH"]].mean(axis=1) - wide[["SL", "BL"]].mean(axis=1)

        if market_returns is not None:
            mkt = market_returns.copy()
            if isinstance(mkt.index, pd.MultiIndex):
                mkt = mkt.reset_index()
            mkt = mkt.rename(columns={"ret": "MKT"})
            mkt["date"] = pd.to_datetime(mkt["date"])
            factors = factors.reset_index().merge(mkt[["date", "MKT"]], on="date", how="left").set_index("date")
        else:
            r = returns.copy()
            if isinstance(r.index, pd.MultiIndex):
                r = r.reset_index()
            r = r.rename(columns={"permno": "asset_id", "me": "market_equity"})
            if "market_equity" in r.columns:
                r = r.sort_values(["asset_id", "date"])
                r["lag_market_equity"] = r.groupby("asset_id")["market_equity"].shift(1)
                r = r[r["lag_market_equity"].gt(0)].copy()
                mkt = value_weighted_by_date(r, "date", "ret", "lag_market_equity").rename("MKT")
                factors = factors.reset_index().merge(mkt.reset_index(), on="date", how="left").set_index("date")
            else:
                factors["MKT"] = np.nan

        if risk_free is not None:
            rf = risk_free.copy()
            if isinstance(rf.index, pd.MultiIndex):
                rf = rf.reset_index()
            rf = rf.rename(columns={"rf": "RF"})
            rf["date"] = pd.to_datetime(rf["date"])
            factors = factors.reset_index().merge(rf[["date", "RF"]], on="date", how="left").set_index("date")
        else:
            factors["RF"] = 0.0
        factors["RF"] = factors["RF"].fillna(0.0)
        factors["MKT_RF"] = factors["MKT"] - factors["RF"]
        return factors.reset_index()[["date", "MKT_RF", "SMB", "HML", "MKT", "RF"]].sort_values("date")

