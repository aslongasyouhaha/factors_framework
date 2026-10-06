from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Factor, FactorBuilder, FactorConfig, WideFactor
from .engine import LayerBook
from .grouped import group_codes, group_qcut, group_quantiles
from .panel import quantile_codes, winsorize_rows


@dataclass
class QuantileSignalFactorBuilder(FactorBuilder):
    """Build full quantile layer holdings and a configurable spread for one signal."""

    signal_source_col: str = "signal_value"
    config: FactorConfig = field(
        default_factory=lambda: FactorConfig(
            name="QuantileSignal",
            frequency="monthly",
            n_groups=10,
            weighting="value",
            long_groups=("D10",),
            short_groups=("D01",),
        )
    )
    winsorize: bool = True
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    required_inputs: tuple[str, ...] = ("panel",)

    @property
    def group_labels(self) -> list[str]:
        n = self.config.n_groups or 10
        return [f"D{i:02d}" for i in range(1, n + 1)]

    def prepare_universe(self, data: dict[str, object]) -> pd.DataFrame:
        panel = data["panel"]
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a DataFrame")
        cfg = self.config
        required = [cfg.rebalance_col, cfg.asset_col, self.signal_source_col]
        missing = [col for col in required if col not in panel.columns]
        if missing:
            raise ValueError(f"Missing quantile panel columns: {missing}")
        out = panel.copy()
        out[cfg.rebalance_col] = pd.to_datetime(out[cfg.rebalance_col])
        if cfg.hold_start_col not in out.columns:
            out[cfg.hold_start_col] = out[cfg.rebalance_col]
        if cfg.hold_end_col not in out.columns:
            out[cfg.hold_end_col] = out[cfg.hold_start_col]
        out[cfg.hold_start_col] = pd.to_datetime(out[cfg.hold_start_col])
        out[cfg.hold_end_col] = pd.to_datetime(out[cfg.hold_end_col])
        return out.dropna(subset=required + [cfg.hold_start_col, cfg.hold_end_col])

    def compute_signal(self, universe: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = universe.copy()
        out[cfg.signal_col] = pd.to_numeric(out[self.signal_source_col], errors="coerce")
        out = out.dropna(subset=[cfg.signal_col])
        if self.winsorize:
            out = out.sort_values(cfg.rebalance_col, kind="stable").reset_index(drop=True)
            out[f"{cfg.signal_col}_raw"] = out[cfg.signal_col]
            codes, n_groups = group_codes(out[cfg.rebalance_col])
            lo, hi = group_quantiles(out[cfg.signal_col].to_numpy(dtype=float), codes, n_groups, (self.winsor_lower, self.winsor_upper))
            out[cfg.signal_col] = out[cfg.signal_col].clip(lo[codes], hi[codes])
        return out

    def assign_groups(self, signal_panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.n_groups or cfg.n_groups < 2:
            raise ValueError("QuantileSignalFactorBuilder requires n_groups >= 2")
        if signal_panel.empty:
            return pd.DataFrame()
        out = signal_panel.sort_values(cfg.rebalance_col, kind="stable").reset_index(drop=True)
        codes, n_groups = group_codes(out[cfg.rebalance_col])
        bins = group_qcut(out[cfg.signal_col].to_numpy(dtype=float), codes, n_groups, cfg.n_groups)
        out = out[bins >= 0].reset_index(drop=True)
        if out.empty:
            return pd.DataFrame()
        out[cfg.group_col] = pd.array(np.asarray(self.group_labels, dtype=object)[bins[bins >= 0]], dtype="string")
        out[cfg.leg_col] = "neutral"
        out.loc[out[cfg.group_col].isin(cfg.long_groups), cfg.leg_col] = "long"
        out.loc[out[cfg.group_col].isin(cfg.short_groups), cfg.leg_col] = "short"
        return out

    def compute_weights(self, grouped_panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if grouped_panel.empty:
            return pd.DataFrame(columns=[cfg.rebalance_col, cfg.hold_start_col, cfg.hold_end_col, cfg.asset_col, cfg.group_col, cfg.leg_col, cfg.weight_col, cfg.signal_col])
        out = grouped_panel.copy()
        weight_base_col = "weight_base"
        if cfg.weighting == "equal":
            out[weight_base_col] = 1.0
        elif cfg.weighting == "value":
            if "market_equity" not in out.columns:
                raise ValueError("Value weighting requires market_equity column.")
            out[weight_base_col] = pd.to_numeric(out["market_equity"], errors="coerce")
        else:
            custom_col = f"{cfg.weighting}_weight"
            if custom_col not in out.columns:
                raise ValueError(f"Custom weighting requires {custom_col} column.")
            out[weight_base_col] = pd.to_numeric(out[custom_col], errors="coerce")
        out = out.dropna(subset=[weight_base_col])
        out = out[out[weight_base_col].gt(0)].copy()
        denom = out.groupby([cfg.rebalance_col, cfg.group_col], observed=True)[weight_base_col].transform("sum")
        out[cfg.weight_col] = out[weight_base_col] / denom
        cols = [cfg.rebalance_col, cfg.hold_start_col, cfg.hold_end_col, cfg.asset_col, cfg.group_col, cfg.leg_col, cfg.weight_col, cfg.signal_col]
        optional = [c for c in [f"{cfg.signal_col}_raw", self.signal_source_col, "market_equity"] if c in out.columns and c not in cols]
        return out[cols + optional].sort_values([cfg.rebalance_col, cfg.group_col, cfg.asset_col])

    def spread_holdings(self, layer_holdings: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        h = layer_holdings[layer_holdings[cfg.group_col].isin((*cfg.long_groups, *cfg.short_groups))].copy()
        if h.empty:
            return h
        h["side_sign"] = np.where(h[cfg.group_col].isin(cfg.short_groups), -1.0, 1.0)
        h[cfg.leg_col] = np.where(h["side_sign"].gt(0), "long", "short")
        side_denom = h.groupby([cfg.rebalance_col, cfg.leg_col], observed=True)[cfg.weight_col].transform("sum")
        h[cfg.weight_col] = h["side_sign"] * h[cfg.weight_col] / side_denom
        return h.drop(columns=["side_sign"]).sort_values([cfg.rebalance_col, cfg.leg_col, cfg.group_col, cfg.asset_col])

    def build_wide(
        self,
        signal: pd.DataFrame,
        weight_base: pd.DataFrame | None = None,
        start_dates: pd.Series | pd.Index | None = None,
        end_dates: pd.Series | pd.Index | None = None,
    ) -> WideFactor:
        """Build layers and the long/short spread from a rebalance-date x asset signal matrix.

        weight_base supplies market equity (weighting="value") or custom weights; it is
        aligned to `signal`, and assets without a positive base are ranked but not held.
        """
        cfg = self.config
        if not cfg.n_groups or cfg.n_groups < 2:
            raise ValueError("QuantileSignalFactorBuilder requires n_groups >= 2")
        signal = signal.sort_index()
        signal.index = pd.DatetimeIndex(signal.index, name="date")
        if self.winsorize:
            signal = winsorize_rows(signal, self.winsor_lower, self.winsor_upper)
        codes = quantile_codes(signal, cfg.n_groups)
        if cfg.weighting == "equal":
            base = np.ones(codes.shape)
        else:
            if weight_base is None:
                raise ValueError(f"weighting={cfg.weighting!r} requires weight_base.")
            base = weight_base.reindex(index=signal.index, columns=signal.columns).to_numpy(dtype=float)
        spreads = {cfg.name: (cfg.long_groups, cfg.short_groups)} if cfg.long_groups and cfg.short_groups else None
        book = LayerBook(
            signal.index,
            signal.columns,
            codes,
            base,
            self.group_labels,
            spreads=spreads,
            start_dates=start_dates,
            end_dates=end_dates,
        )
        metadata = {
            "frequency": cfg.frequency,
            "n_groups": cfg.n_groups,
            "weighting": cfg.weighting,
            "winsorize": self.winsorize,
            "winsor_lower": self.winsor_lower,
            "winsor_upper": self.winsor_upper,
            "long_groups": cfg.long_groups,
            "short_groups": cfg.short_groups,
        }
        return WideFactor(name=cfg.name, book=book, signal=signal, metadata=metadata)

    def build(self, data: dict[str, object]) -> Factor:
        factor = super().build(data)
        factor.metadata.update(
            {
                "signal_source_col": self.signal_source_col,
                "winsorize": self.winsorize,
                "winsor_lower": self.winsor_lower,
                "winsor_upper": self.winsor_upper,
                "long_groups": self.config.long_groups,
                "short_groups": self.config.short_groups,
            }
        )
        return factor
