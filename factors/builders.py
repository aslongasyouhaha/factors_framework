from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .grouped import group_bounds, group_codes, group_median, group_quantiles, residualize


@dataclass(frozen=True)
class OutlierRule:
    cols: tuple[str, ...]
    method: str = "quantile"
    action: str = "winsorize"
    by_date: bool = True
    lower: float | None = 0.01
    upper: float | None = 0.99
    n_std: float = 3.0
    n_mad: float = 3.5
    iqr_multiplier: float = 1.5
    suffix: str = "_outlier"


@dataclass(frozen=True)
class MissingValueRule:
    cols: tuple[str, ...]
    method: str = "drop"
    by: tuple[str, ...] = ("date",)
    fill_value: float | str | None = None
    max_ffill_periods: int | None = None
    missing_suffix: str = "_was_missing"


@dataclass(frozen=True)
class StandardizationRule:
    cols: tuple[str, ...]
    method: str = "zscore"
    by: tuple[str, ...] = ("date",)
    suffix: str = "_z"
    overwrite: bool = False
    min_std: float = 1e-12


@dataclass(frozen=True)
class NeutralizationRule:
    y_col: str
    x_cols: tuple[str, ...] = ()
    category_cols: tuple[str, ...] = ()
    by: tuple[str, ...] = ("date",)
    output_col: str | None = None
    add_intercept: bool = True
    min_obs: int = 20


@dataclass(frozen=True)
class FactorBuildTools:
    """Reusable methods for constructing Fama-French style portfolios."""

    nyse_exchange_codes: tuple[int, ...] = (1,)

    @staticmethod
    def to_panel_index(
        frame: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
        sort: bool = True,
    ) -> pd.DataFrame:
        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col, asset_col])
        out = out.set_index([date_col, asset_col])
        out.index = out.index.set_names(["date", "asset_id"])
        return out.sort_index() if sort else out

    @staticmethod
    def from_panel_index(
        panel: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
    ) -> pd.DataFrame:
        out = panel.reset_index()
        return out.rename(columns={"date": date_col, "asset_id": asset_col})

    @staticmethod
    def ensure_panel_index(
        frame: pd.DataFrame,
        date_col: str = "date",
        asset_col: str = "asset_id",
    ) -> pd.DataFrame:
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names[:2]) == ["date", "asset_id"]:
            return frame.sort_index()
        return FactorBuildTools.to_panel_index(frame, date_col=date_col, asset_col=asset_col)

    @staticmethod
    def clean_frame(
        frame: pd.DataFrame,
        date_cols: tuple[str, ...] = ("date", "rebalance_date", "hold_start", "hold_end", "formation_date"),
        drop_duplicated_columns: bool = True,
    ) -> pd.DataFrame:
        out = frame.copy()
        if drop_duplicated_columns:
            out = out.loc[:, ~out.columns.duplicated()].copy()
        for col in date_cols:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out

    def clean_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for name, value in data.items():
            cleaned[name] = self.clean_frame(value) if isinstance(value, pd.DataFrame) else value
        return cleaned

    @staticmethod
    def filter_valid_panel(
        panel: pd.DataFrame,
        required_cols: tuple[str, ...],
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        return out.dropna(subset=[col for col in required_cols if col in out.columns])

    @staticmethod
    def filter_listing_age(
        panel: pd.DataFrame,
        min_days: int = 180,
        first_trade_col: str = "first_trade_date",
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        if first_trade_col not in out.columns:
            first_dates = out.reset_index().groupby("asset_id")["date"].min()
            out = out.join(first_dates.rename(first_trade_col), on="asset_id")
        out[first_trade_col] = pd.to_datetime(out[first_trade_col], errors="coerce")
        age = out.index.get_level_values("date") - out[first_trade_col]
        return out[age.dt.days >= min_days]

    @staticmethod
    def filter_active_trading(
        panel: pd.DataFrame,
        ret_col: str = "ret",
        volume_col: str = "volume",
        require_positive_volume: bool = True,
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        mask = out[ret_col].notna() if ret_col in out.columns else pd.Series(True, index=out.index)
        if require_positive_volume and volume_col in out.columns:
            mask &= out[volume_col].fillna(0).gt(0)
        return out[mask]

    @staticmethod
    def filter_liquidity(
        panel: pd.DataFrame,
        min_dollar_volume: float | None = None,
        min_turnover: float | None = None,
        dollar_volume_col: str = "dollar_volume",
        turnover_col: str = "turnover",
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        mask = pd.Series(True, index=out.index)
        if min_dollar_volume is not None and dollar_volume_col in out.columns:
            mask &= out[dollar_volume_col].ge(min_dollar_volume)
        if min_turnover is not None and turnover_col in out.columns:
            mask &= out[turnover_col].ge(min_turnover)
        return out[mask]

    @staticmethod
    def filter_positive(
        panel: pd.DataFrame,
        cols: tuple[str, ...] = ("me",),
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        mask = pd.Series(True, index=out.index)
        for col in cols:
            if col in out.columns:
                mask &= out[col].gt(0)
        return out[mask]

    @staticmethod
    def outlier_bounds(
        s: pd.Series,
        rule: OutlierRule,
    ) -> tuple[float, float]:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return -np.inf, np.inf

        method = rule.method.lower()
        if method in {"std", "mean_std", "zscore"}:
            mean = x.mean()
            std = x.std(ddof=1)
            if pd.isna(std) or std == 0:
                return -np.inf, np.inf
            return float(mean - rule.n_std * std), float(mean + rule.n_std * std)

        if method in {"mad", "median_mad", "absolute_median"}:
            med = x.median()
            mad = (x - med).abs().median()
            if pd.isna(mad) or mad == 0:
                return -np.inf, np.inf
            scaled = 1.4826 * mad
            return float(med - rule.n_mad * scaled), float(med + rule.n_mad * scaled)

        if method in {"iqr", "box", "boxplot", "box_plot"}:
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                return -np.inf, np.inf
            return float(q1 - rule.iqr_multiplier * iqr), float(q3 + rule.iqr_multiplier * iqr)

        if method in {"quantile", "percentile"}:
            lower = 0.0 if rule.lower is None else rule.lower
            upper = 1.0 if rule.upper is None else rule.upper
            return float(x.quantile(lower)), float(x.quantile(upper))

        raise ValueError(f"Unknown outlier method: {rule.method}")

    @staticmethod
    def handle_outliers(
        panel: pd.DataFrame,
        rule: OutlierRule,
    ) -> pd.DataFrame:
        """Detect and handle outliers with drop, winsorize, clip, or flag action."""
        out = FactorBuildTools.ensure_panel_index(panel).copy()
        action = rule.action.lower()
        if action not in {"drop", "winsorize", "clip", "flag", "none"}:
            raise ValueError(f"Unknown outlier action: {rule.action}")

        if rule.by_date:
            codes, n_groups = group_codes(out.index.get_level_values("date"))
        else:
            codes, n_groups = np.zeros(len(out), dtype=np.int64), 1
        drop_mask = np.zeros(len(out), dtype=bool)
        for col in rule.cols:
            if col not in out.columns:
                continue
            x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            lo, hi = group_bounds(
                x, codes, n_groups, rule.method, rule.lower, rule.upper, rule.n_std, rule.n_mad, rule.iqr_multiplier
            )
            lo_x, hi_x = lo[codes], hi[codes]
            with np.errstate(invalid="ignore"):
                mask = (x < lo_x) | (x > hi_x)
            if action == "drop":
                drop_mask |= mask
                continue
            out[f"{col}{rule.suffix}"] = mask
            if action in {"winsorize", "clip"}:
                out[col] = out[col].clip(lower=lo_x, upper=hi_x)

        if action == "drop":
            out = out[~drop_mask]
        return out

    @staticmethod
    def handle_outlier_rules(
        panel: pd.DataFrame,
        rules: tuple[OutlierRule, ...],
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        for rule in rules:
            out = FactorBuildTools.handle_outliers(out, rule)
        return out

    @staticmethod
    def _group_keys_for_panel(panel: pd.DataFrame, by: tuple[str, ...]) -> list[pd.Series | pd.Index]:
        keys: list[pd.Series | pd.Index] = []
        for key in by:
            if key == "date":
                keys.append(panel.index.get_level_values("date"))
            elif key == "asset_id":
                keys.append(panel.index.get_level_values("asset_id"))
            elif key in panel.columns:
                keys.append(panel[key])
            else:
                raise ValueError(f"Missing grouping key for missing-value rule: {key}")
        return keys

    @staticmethod
    def handle_missing_values(
        panel: pd.DataFrame,
        rule: MissingValueRule,
    ) -> pd.DataFrame:
        """Handle missing values with drop, mean, median, zero, value, ffill, or flag."""
        out = FactorBuildTools.ensure_panel_index(panel).copy()
        method = rule.method.lower()
        if method not in {"drop", "mean", "median", "zero", "value", "ffill", "bfill", "flag"}:
            raise ValueError(f"Unknown missing-value method: {rule.method}")

        cols = [col for col in rule.cols if col in out.columns]
        if not cols:
            return out

        for col in cols:
            out[f"{col}{rule.missing_suffix}"] = out[col].isna()

        if method == "flag":
            return out

        if method == "drop":
            return out.dropna(subset=cols)

        if method == "zero":
            out[cols] = out[cols].fillna(0.0)
            return out

        if method == "value":
            out[cols] = out[cols].fillna(rule.fill_value)
            return out

        if method in {"ffill", "bfill"}:
            group_by = rule.by or ("asset_id",)
            keys = FactorBuildTools._group_keys_for_panel(out, group_by)
            for col in cols:
                grouped = out[col].groupby(keys, sort=False)
                if method == "ffill":
                    out[col] = grouped.ffill(limit=rule.max_ffill_periods)
                else:
                    out[col] = grouped.bfill(limit=rule.max_ffill_periods)
            return out

        keys = FactorBuildTools._group_keys_for_panel(out, rule.by)
        for col in cols:
            grouped = out[col].groupby(keys, sort=False)
            fill = grouped.transform("median" if method == "median" else "mean")
            out[col] = out[col].where(out[col].notna(), fill)
        return out

    @staticmethod
    def handle_missing_value_rules(
        panel: pd.DataFrame,
        rules: tuple[MissingValueRule, ...],
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        for rule in rules:
            out = FactorBuildTools.handle_missing_values(out, rule)
        return out

    @staticmethod
    def standardize(
        panel: pd.DataFrame,
        rule: StandardizationRule,
    ) -> pd.DataFrame:
        """Standardize columns, defaulting to cross-sectional z-scores by date."""
        out = FactorBuildTools.ensure_panel_index(panel).copy()
        method = rule.method.lower()
        if method not in {"zscore", "z_score"}:
            raise ValueError(f"Unknown standardization method: {rule.method}")

        keys = FactorBuildTools._group_keys_for_panel(out, rule.by)
        for col in rule.cols:
            if col not in out.columns:
                continue
            target_col = col if rule.overwrite else f"{col}{rule.suffix}"
            x = pd.to_numeric(out[col], errors="coerce")
            grouped = x.groupby(keys, sort=False)
            mean = grouped.transform("mean")
            std = grouped.transform("std")
            std = std.where(std.abs() > rule.min_std)
            out[target_col] = (x - mean) / std
        return out

    @staticmethod
    def standardize_rules(
        panel: pd.DataFrame,
        rules: tuple[StandardizationRule, ...],
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        for rule in rules:
            out = FactorBuildTools.standardize(out, rule)
        return out

    @staticmethod
    def neutralize(
        panel: pd.DataFrame,
        rule: NeutralizationRule,
    ) -> pd.DataFrame:
        """Neutralize a signal by cross-sectional OLS residualization."""
        out = FactorBuildTools.ensure_panel_index(panel).copy()
        if rule.y_col not in out.columns:
            raise ValueError(f"Missing neutralization target column: {rule.y_col}")
        missing_x = [col for col in rule.x_cols if col not in out.columns]
        missing_cat = [col for col in rule.category_cols if col not in out.columns]
        if missing_x or missing_cat:
            raise ValueError(f"Missing neutralization columns: {missing_x + missing_cat}")

        output_col = rule.output_col or f"{rule.y_col}_neutral"
        codes, n_groups = group_codes(*FactorBuildTools._group_keys_for_panel(out, rule.by))
        x = (
            np.column_stack([pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) for col in rule.x_cols])
            if rule.x_cols
            else None
        )
        # Rows with a missing category are excluded rather than pooled into a baseline category.
        categories = [pd.factorize(out[col].astype("string"), sort=True)[0].astype(np.int64) for col in rule.category_cols]
        out[output_col] = residualize(
            pd.to_numeric(out[rule.y_col], errors="coerce").to_numpy(dtype=float),
            x,
            codes,
            n_groups,
            category_codes=categories,
            add_intercept=rule.add_intercept,
            min_obs=rule.min_obs,
        )
        return out

    @staticmethod
    def neutralize_rules(
        panel: pd.DataFrame,
        rules: tuple[NeutralizationRule, ...],
    ) -> pd.DataFrame:
        out = FactorBuildTools.ensure_panel_index(panel)
        for rule in rules:
            out = FactorBuildTools.neutralize(out, rule)
        return out

    @staticmethod
    def first_available(*values: pd.Series | float | None) -> pd.Series | float:
        out = None
        for value in values:
            if value is None:
                continue
            if out is None:
                out = value.copy() if isinstance(value, pd.Series) else value
            elif isinstance(out, pd.Series):
                out = out.where(out.notna(), value)
        return out

    @staticmethod
    def market_equity(price: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
        """Compute market equity from unadjusted price and shares outstanding."""
        return price.abs() * shares_outstanding

    @staticmethod
    def book_equity(
        seq: pd.Series | None = None,
        ceq: pd.Series | None = None,
        txditc: pd.Series | None = None,
        pstkrv: pd.Series | None = None,
        pstkl: pd.Series | None = None,
        pstk: pd.Series | None = None,
        at: pd.Series | None = None,
        lt: pd.Series | None = None,
    ) -> pd.Series:
        """Compute book equity using the standard Compustat fallback convention."""
        shareholders_equity = FactorBuildTools.first_available(seq, ceq)
        if at is not None and lt is not None:
            asset_liability_equity = at - lt
            shareholders_equity = FactorBuildTools.first_available(shareholders_equity, asset_liability_equity)
        if shareholders_equity is None:
            raise ValueError("Need seq, ceq, or at and lt to compute book equity.")

        deferred_taxes = txditc.fillna(0.0) if txditc is not None else 0.0
        preferred_stock = FactorBuildTools.first_available(pstkrv, pstkl, pstk)
        if isinstance(preferred_stock, pd.Series):
            preferred_stock = preferred_stock.fillna(0.0)
        elif preferred_stock is None:
            preferred_stock = 0.0

        be = shareholders_equity + deferred_taxes - preferred_stock
        return be.where(be > 0)

    @staticmethod
    def value_weighted_return(
        returns: pd.Series,
        weights: pd.Series,
    ) -> float:
        mask = returns.notna() & weights.notna() & weights.gt(0)
        if not mask.any():
            return np.nan
        return float(np.average(returns[mask], weights=weights[mask]))

    def size_breakpoint(
        self,
        characteristics: pd.DataFrame,
        me_col: str = "me",
        exchange_col: str = "exchange_code",
    ) -> float:
        nyse = characteristics[characteristics[exchange_col].isin(self.nyse_exchange_codes)]
        return float(nyse[me_col].median())

    def bm_breakpoints(
        self,
        characteristics: pd.DataFrame,
        bm_col: str = "bm",
        exchange_col: str = "exchange_code",
    ) -> tuple[float, float]:
        nyse = characteristics[characteristics[exchange_col].isin(self.nyse_exchange_codes)]
        return tuple(nyse[bm_col].quantile([0.3, 0.7]).astype(float))

    def assign_2x3_buckets(
        self,
        characteristics: pd.DataFrame,
        me_col: str = "me",
        bm_col: str = "bm",
        exchange_col: str = "exchange_code",
    ) -> pd.DataFrame:
        out = characteristics.copy()
        size_cut = self.size_breakpoint(out, me_col=me_col, exchange_col=exchange_col)
        bm_low, bm_high = self.bm_breakpoints(out, bm_col=bm_col, exchange_col=exchange_col)

        out["size_bucket"] = np.where(out[me_col] <= size_cut, "S", "B")
        out["bm_bucket"] = pd.cut(
            out[bm_col],
            bins=[-np.inf, bm_low, bm_high, np.inf],
            labels=["L", "M", "H"],
        ).astype("string")
        return out

    def portfolio_returns_2x3(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
        ret_col: str = "ret",
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Value-weight daily/monthly returns for six size-BM portfolios."""
        required = {date_col, ret_col, weight_col, "size_bucket", "bm_bucket"}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        ret = pd.to_numeric(panel[ret_col], errors="coerce")
        weight = pd.to_numeric(panel[weight_col], errors="coerce")
        usable = ret.notna() & weight.notna() & weight.gt(0)
        parts = pd.DataFrame(
            {
                "num": (ret * weight).where(usable),
                "den": weight.where(usable),
                "n_assets": ret.notna().astype(int),
            }
        )
        keys = [panel[date_col], panel["size_bucket"], panel["bm_bucket"]]
        agg = parts.groupby(keys, observed=True).agg(["sum", "count"])
        out = pd.DataFrame(
            {
                "ret": np.where(agg[("den", "count")].gt(0), agg[("num", "sum")] / agg[("den", "sum")], np.nan),
                "n_assets": agg[("n_assets", "sum")].astype(int),
            },
            index=agg.index,
        ).reset_index()
        out.columns = ["date", "size_bucket", "bm_bucket", "ret", "n_assets"]
        out["portfolio"] = out["size_bucket"].astype(str) + out["bm_bucket"].astype(str)
        return out[["date", "portfolio", "ret", "n_assets"]].sort_values(["date", "portfolio"]).reset_index(drop=True)

    def assign_2x3_by_date(
        self,
        characteristics: pd.DataFrame,
        date_col: str = "rebalance_date",
        me_col: str = "me",
        bm_col: str = "bm",
        exchange_col: str = "exchange_code",
    ) -> pd.DataFrame:
        """assign_2x3_buckets applied within every date at once (NYSE breakpoints per date)."""
        out = characteristics.copy()
        codes, n_groups = group_codes(out[date_col])
        nyse = out[exchange_col].isin(self.nyse_exchange_codes).to_numpy()
        me = pd.to_numeric(out[me_col], errors="coerce").to_numpy(dtype=float)
        bm = pd.to_numeric(out[bm_col], errors="coerce").to_numpy(dtype=float)
        size_cut = group_median(np.where(nyse, me, np.nan), codes, n_groups)[codes]
        bm_low, bm_high = group_quantiles(np.where(nyse, bm, np.nan), codes, n_groups, (0.3, 0.7))
        bm_low, bm_high = bm_low[codes], bm_high[codes]
        out["size_bucket"] = np.where(me <= size_cut, "S", "B")
        bucket = np.select([bm <= bm_low, bm <= bm_high, bm > bm_high], ["L", "M", "H"], default="")
        out["bm_bucket"] = pd.array(np.where(bucket == "", None, bucket), dtype="string")
        return out
