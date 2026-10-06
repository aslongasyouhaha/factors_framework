from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FactorConfig:
    name: str
    frequency: str = "monthly"
    n_groups: int | None = None
    weighting: str = "value"
    long_groups: tuple[str, ...] = ()
    short_groups: tuple[str, ...] = ()
    asset_col: str = "asset_id"
    date_col: str = "date"
    rebalance_col: str = "rebalance_date"
    hold_start_col: str = "hold_start"
    hold_end_col: str = "hold_end"
    signal_col: str = "signal_value"
    group_col: str = "group"
    leg_col: str = "leg"
    weight_col: str = "target_weight"


@dataclass
class Factor:
    """Constructed factor artifact consumed by backtests and analysis."""

    name: str
    holdings: pd.DataFrame
    returns: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_holdings(self) -> None:
        required = {
            "rebalance_date",
            "hold_start",
            "hold_end",
            "asset_id",
            "group",
            "leg",
            "target_weight",
        }
        missing = required.difference(self.holdings.columns)
        if missing:
            raise ValueError(f"{self.name} holdings missing columns: {sorted(missing)}")

    def to_parquet(self, path: str) -> None:
        self.holdings.to_parquet(path, index=False)


@dataclass
class WideFactor:
    """Factor built on date x asset matrices: a weight book plus the signal used to rank."""

    name: str
    book: Any
    signal: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class FactorBuilder(ABC):
    """Abstract construction process for factor-specific builders."""

    config: FactorConfig
    required_inputs: tuple[str, ...] = ()

    def validate_inputs(self, data: dict[str, Any]) -> None:
        missing = [name for name in self.required_inputs if name not in data]
        if missing:
            raise ValueError(f"{self.config.name} missing inputs: {missing}")

    @abstractmethod
    def prepare_universe(self, data: dict[str, Any]) -> pd.DataFrame:
        """Create the security-period panel used for signal ranking."""

    @abstractmethod
    def compute_signal(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Add the factor ranking signal to the universe."""

    @abstractmethod
    def assign_groups(self, signal_panel: pd.DataFrame) -> pd.DataFrame:
        """Assign assets to groups and legs."""

    @abstractmethod
    def compute_weights(self, grouped_panel: pd.DataFrame) -> pd.DataFrame:
        """Compute target holdings and weights for each rebalance date."""

    def diagnostics(self, holdings: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if holdings.empty:
            return pd.DataFrame()
        return (
            holdings.groupby([cfg.rebalance_col, cfg.group_col], dropna=False)
            .agg(
                n_assets=(cfg.asset_col, "nunique"),
                gross_weight=(cfg.weight_col, lambda x: x.abs().sum()),
                net_weight=(cfg.weight_col, "sum"),
            )
            .reset_index()
        )

    def build(self, data: dict[str, Any]) -> Factor:
        self.validate_inputs(data)
        universe = self.prepare_universe(data)
        signal_panel = self.compute_signal(universe)
        grouped = self.assign_groups(signal_panel)
        holdings = self.compute_weights(grouped)
        factor = Factor(
            name=self.config.name,
            holdings=holdings,
            diagnostics=self.diagnostics(holdings),
            metadata={
                "frequency": self.config.frequency,
                "n_groups": self.config.n_groups,
                "weighting": self.config.weighting,
            },
        )
        factor.validate_holdings()
        return factor
