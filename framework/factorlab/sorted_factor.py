from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FactorBuilder, FactorConfig, WideFactor
from .engine import WeightBook
from .grouped import group_codes, group_qcut


class SortedFactorBuilder(FactorBuilder):
    """Base implementation for single-signal sorted factors."""

    config: FactorConfig
    signal_source_col: str
    required_inputs: tuple[str, ...] = ("panel",)

    def prepare_universe(self, data: dict[str, object]) -> pd.DataFrame:
        panel = data["panel"]
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a DataFrame")
        return panel.dropna(subset=[self.config.rebalance_col, self.config.asset_col]).copy()

    def compute_signal(self, universe: pd.DataFrame) -> pd.DataFrame:
        if self.signal_source_col not in universe.columns:
            raise ValueError(f"Missing signal source column: {self.signal_source_col}")
        out = universe.copy()
        out[self.config.signal_col] = pd.to_numeric(out[self.signal_source_col], errors="coerce")
        return out.dropna(subset=[self.config.signal_col])

    def assign_groups(self, signal_panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.n_groups or cfg.n_groups < 2:
            raise ValueError("SortedFactor requires n_groups >= 2")

        out = signal_panel.sort_values(cfg.rebalance_col, kind="stable").copy()
        codes, n_groups = group_codes(out[cfg.rebalance_col])
        bins = group_qcut(out[cfg.signal_col].to_numpy(dtype=float), codes, n_groups, cfg.n_groups, min_count=1)
        out[cfg.group_col] = np.char.add("G", (bins + 1).astype(str)).astype(object)
        out[cfg.leg_col] = "neutral"
        out.loc[out[cfg.group_col].isin(cfg.long_groups), cfg.leg_col] = "long"
        out.loc[out[cfg.group_col].isin(cfg.short_groups), cfg.leg_col] = "short"
        return out

    def build_wide(
        self,
        signal: pd.DataFrame,
        weight_base: pd.DataFrame | None = None,
        start_dates: pd.Series | pd.Index | None = None,
        end_dates: pd.Series | pd.Index | None = None,
    ) -> WideFactor:
        """Long/short weights from a rebalance-date x asset signal matrix (one WeightBook column).

        Each leg is weighted across all of its groups together, as compute_weights does.
        """
        cfg = self.config
        if not cfg.n_groups or cfg.n_groups < 2:
            raise ValueError("SortedFactor requires n_groups >= 2")
        signal = signal.sort_index()
        signal.index = pd.DatetimeIndex(signal.index, name="date")
        x = signal.to_numpy(dtype=float)
        rows = np.repeat(np.arange(x.shape[0], dtype=np.int64), x.shape[1])
        bins = group_qcut(x.ravel(), rows, x.shape[0], cfg.n_groups, min_count=1).reshape(x.shape)
        if cfg.weighting == "equal":
            base = np.ones(x.shape)
        else:
            if weight_base is None:
                raise ValueError(f"weighting={cfg.weighting!r} requires weight_base.")
            base = weight_base.reindex(index=signal.index, columns=signal.columns).to_numpy(dtype=float)
        base = np.where(np.isfinite(base) & (base > 0), base, 0.0)
        label_of = {f"G{i + 1}": i for i in range(cfg.n_groups)}
        weights = np.zeros(x.shape)
        for groups, sign in ((cfg.long_groups, 1.0), (cfg.short_groups, -1.0)):
            leg = np.isin(bins, [label_of[g] for g in groups if g in label_of])
            leg_base = np.where(leg, base, 0.0)
            total = leg_base.sum(axis=1, keepdims=True)
            weights += sign * np.divide(leg_base, total, out=np.zeros_like(leg_base), where=total > 0)
        book = WeightBook(
            {cfg.name: pd.DataFrame(weights, index=signal.index, columns=signal.columns)},
            start_dates=start_dates,
            end_dates=end_dates,
        )
        return WideFactor(name=cfg.name, book=book, signal=signal, metadata={"n_groups": cfg.n_groups, "weighting": cfg.weighting})

    def compute_weights(self, grouped_panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        active = grouped_panel[grouped_panel[cfg.leg_col].isin(["long", "short"])].copy()
        if active.empty:
            return pd.DataFrame(columns=[cfg.rebalance_col, cfg.hold_start_col, cfg.hold_end_col, cfg.asset_col, cfg.group_col, cfg.leg_col, cfg.weight_col, cfg.signal_col])

        if cfg.hold_start_col not in active.columns:
            active[cfg.hold_start_col] = active[cfg.rebalance_col]
        if cfg.hold_end_col not in active.columns:
            active[cfg.hold_end_col] = pd.NaT

        weight_base_col = "weight_base"
        if cfg.weighting == "equal":
            active[weight_base_col] = 1.0
        elif cfg.weighting == "value":
            if "market_equity" not in active.columns:
                raise ValueError("Value weighting requires market_equity column.")
            active[weight_base_col] = pd.to_numeric(active["market_equity"], errors="coerce")
        else:
            custom_col = f"{cfg.weighting}_weight"
            if custom_col not in active.columns:
                raise ValueError(f"Custom weighting requires {custom_col} column.")
            active[weight_base_col] = pd.to_numeric(active[custom_col], errors="coerce")

        active = active.dropna(subset=[weight_base_col])
        active = active[active[weight_base_col] > 0].copy()
        active["side_sign"] = np.where(active[cfg.leg_col].eq("short"), -1.0, 1.0)
        gross_by_side = active.groupby([cfg.rebalance_col, cfg.leg_col])[weight_base_col].transform("sum")
        active[cfg.weight_col] = active["side_sign"] * active[weight_base_col] / gross_by_side

        cols = [
            cfg.rebalance_col,
            cfg.hold_start_col,
            cfg.hold_end_col,
            cfg.asset_col,
            cfg.group_col,
            cfg.leg_col,
            cfg.weight_col,
            cfg.signal_col,
        ]
        optional = [c for c in ["market_equity", self.signal_source_col] if c in active.columns and c not in cols]
        return active[cols + optional].sort_values([cfg.rebalance_col, cfg.leg_col, cfg.group_col, cfg.asset_col])
