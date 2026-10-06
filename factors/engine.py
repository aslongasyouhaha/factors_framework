from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# Timing convention shared by every book: weights decided at the close of rebalance date d_k are
# held over the return rows in (d_k, d_{k+1}], optionally narrowed to [start_k, end_k].
# Holdings live only at rebalance dates (K x N); they are never expanded to a daily long table.


def _as_index(dates: Sequence | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates))


def holding_segments(
    index: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    start_dates: pd.DatetimeIndex | None = None,
    end_dates: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """Return a (K, 2) array of [first_row, stop_row) positions in `index` for each rebalance."""
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Return index must be sorted and unique.")
    k = len(rebalance_dates)
    first = index.searchsorted(rebalance_dates, side="right")
    if start_dates is not None:
        first = np.maximum(first, index.searchsorted(start_dates, side="left"))
    stop = np.append(index.searchsorted(rebalance_dates[1:], side="right"), len(index))
    if end_dates is not None:
        stop = np.minimum(stop, index.searchsorted(end_dates, side="right"))
    stop = np.maximum(stop, first)
    return np.column_stack([first, stop]).astype(np.int64).reshape(k, 2)


class Book:
    """Target weights at rebalance dates for one or more portfolios over a common asset axis."""

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        assets: pd.Index,
        start_dates: pd.DatetimeIndex | None = None,
        end_dates: pd.DatetimeIndex | None = None,
    ):
        self.dates = dates
        self.assets = assets
        self.start_dates = start_dates
        self.end_dates = end_dates

    @property
    def names(self) -> list[str]:
        raise NotImplementedError

    def weights_at(self, k: int) -> np.ndarray:
        """(N, P) target weights for rebalance k; NaN-free."""
        raise NotImplementedError

    def segments(self, index: pd.DatetimeIndex) -> np.ndarray:
        return holding_segments(index, self.dates, self.start_dates, self.end_dates)

    def to_long(self, extras: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """Long holdings table (non-zero weights only), for inspection or saving.

        extras: name -> rebalance-date x asset matrix (e.g. the signal or market equity) whose
        value for each holding is added as a column.
        """
        ks, assets, ports, weights = [], [], [], []
        for k in range(len(self.dates)):
            w = self.weights_at(k)
            rows, cols = np.nonzero(w)
            ks.append(np.full(len(rows), k))
            assets.append(rows)
            ports.append(cols)
            weights.append(w[rows, cols])
        cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
        k_idx, a_idx, p_idx = cat(ks, int), cat(assets, int), cat(ports, int)
        out = pd.DataFrame(
            {
                "rebalance_date": self.dates[k_idx],
                "portfolio": np.asarray(self.names, dtype=object)[p_idx],
                "asset_id": self.assets[a_idx],
                "target_weight": cat(weights, float),
            }
        )
        for name, mat in (extras or {}).items():
            values = mat.reindex(index=self.dates, columns=self.assets).to_numpy()
            out[name] = values[k_idx, a_idx]
        return out


class WeightBook(Book):
    """Explicit weight matrices, one (K, N) matrix per portfolio."""

    def __init__(
        self,
        weights: pd.DataFrame | dict[str, pd.DataFrame],
        start_dates: Sequence | None = None,
        end_dates: Sequence | None = None,
    ):
        frames = weights if isinstance(weights, dict) else {"portfolio": weights}
        first = next(iter(frames.values()))
        dates = _as_index(first.index)
        assets = pd.Index(first.columns)
        for name, frame in frames.items():
            if not _as_index(frame.index).equals(dates) or not pd.Index(frame.columns).equals(assets):
                raise ValueError(f"Weight matrix {name!r} is not aligned with the others.")
        if not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError("Rebalance dates must be sorted and unique.")
        super().__init__(
            dates=dates,
            assets=assets,
            start_dates=None if start_dates is None else _as_index(start_dates),
            end_dates=None if end_dates is None else _as_index(end_dates),
        )
        self._names = list(frames)
        stacked = np.stack([f.to_numpy(dtype=float) for f in frames.values()], axis=2)
        self._weights = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def names(self) -> list[str]:
        return self._names

    def weights_at(self, k: int) -> np.ndarray:
        return self._weights[k]

    @classmethod
    def from_holdings(
        cls,
        holdings: pd.DataFrame,
        weight_col: str = "target_weight",
        rebalance_col: str = "rebalance_date",
        asset_col: str = "asset_id",
        hold_start_col: str | None = "hold_start",
        hold_end_col: str | None = "hold_end",
        name: str = "portfolio",
    ) -> "WeightBook":
        h = holdings.copy()
        h[rebalance_col] = pd.to_datetime(h[rebalance_col])
        wide = h.pivot_table(index=rebalance_col, columns=asset_col, values=weight_col, aggfunc="sum", fill_value=0.0)
        wide = wide.sort_index()
        start, end = _holding_window(h, wide.index, rebalance_col, hold_start_col, hold_end_col)
        return cls({name: wide}, start_dates=start, end_dates=end)


class LayerBook(Book):
    """Quantile layers stored compactly as group codes plus a weighting base.

    Layer p holds assets with code == p, weighted by `base` and normalised to sum to one.
    Optional spreads are added as extra portfolios: mean(long layers) - mean(short layers).
    """

    def __init__(
        self,
        dates: Sequence,
        assets: Sequence,
        codes: np.ndarray,
        base: np.ndarray,
        labels: Sequence[str],
        spreads: dict[str, tuple[Sequence[str], Sequence[str]]] | None = None,
        start_dates: Sequence | None = None,
        end_dates: Sequence | None = None,
    ):
        dates = _as_index(dates)
        if not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError("Rebalance dates must be sorted and unique.")
        super().__init__(
            dates=dates,
            assets=pd.Index(assets),
            start_dates=None if start_dates is None else _as_index(start_dates),
            end_dates=None if end_dates is None else _as_index(end_dates),
        )
        codes = np.asarray(codes)
        base = np.nan_to_num(np.asarray(base, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if codes.shape != (len(dates), len(self.assets)) or base.shape != codes.shape:
            raise ValueError("codes and base must both be (n_dates, n_assets).")
        self.labels = [str(x) for x in labels]
        self._codes = np.where(base > 0, codes, -1).astype(np.int16)
        self._base = np.where(self._codes >= 0, base, 0.0)
        self._spreads: list[tuple[str, np.ndarray]] = []
        for name, (long_groups, short_groups) in (spreads or {}).items():
            combo = np.zeros(len(self.labels))
            for groups, sign in ((long_groups, 1.0), (short_groups, -1.0)):
                missing = set(groups).difference(self.labels)
                if missing:
                    raise ValueError(f"Spread {name!r} uses unknown groups: {sorted(missing)}")
                for g in groups:
                    combo[self.labels.index(g)] += sign / len(groups)
            self._spreads.append((name, combo))

    @property
    def names(self) -> list[str]:
        return self.labels + [name for name, _ in self._spreads]

    @property
    def codes(self) -> pd.DataFrame:
        return pd.DataFrame(self._codes, index=self.dates, columns=self.assets)

    @property
    def spread_matrix(self) -> np.ndarray:
        """(n_layers, n_spreads) coefficients turning layer returns into spread returns."""
        if not self._spreads:
            return np.zeros((len(self.labels), 0))
        return np.column_stack([c for _, c in self._spreads])

    def membership(self, k: int) -> np.ndarray:
        """(N, n_layers) one-hot layer membership at rebalance k."""
        codes = self._codes[k]
        rows = np.flatnonzero(codes >= 0)
        m = np.zeros((len(self.assets), len(self.labels)))
        m[rows, codes[rows]] = 1.0
        return m

    def weights_at(self, k: int) -> np.ndarray:
        w = self.membership(k) * self._base[k][:, None]
        total = w.sum(axis=0)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        if not self._spreads:
            return w
        # A spread leg with an empty layer would be unbalanced; zero the whole spread instead.
        combos = self.spread_matrix
        combos = combos * np.array([(total[c != 0] > 0).all() for c in combos.T], dtype=float)
        return np.hstack([w, w @ combos])

    @classmethod
    def from_holdings(
        cls,
        holdings: pd.DataFrame,
        group_col: str = "group",
        weight_col: str = "target_weight",
        rebalance_col: str = "rebalance_date",
        asset_col: str = "asset_id",
        hold_start_col: str | None = "hold_start",
        hold_end_col: str | None = "hold_end",
    ) -> "LayerBook":
        h = holdings.copy()
        h[rebalance_col] = pd.to_datetime(h[rebalance_col])
        h[group_col] = h[group_col].astype(str)
        labels = sorted(h[group_col].unique())
        h["_code"] = h[group_col].map({g: i for i, g in enumerate(labels)})
        h["_base"] = h[weight_col].abs()
        if h.duplicated([rebalance_col, asset_col]).any():
            raise ValueError("Each asset may belong to only one layer per rebalance date.")
        codes = h.pivot(index=rebalance_col, columns=asset_col, values="_code").sort_index()
        base = h.pivot(index=rebalance_col, columns=asset_col, values="_base").reindex_like(codes)
        start, end = _holding_window(h, codes.index, rebalance_col, hold_start_col, hold_end_col)
        return cls(
            codes.index,
            codes.columns,
            codes.fillna(-1).to_numpy(dtype=np.int16),
            base.fillna(0.0).to_numpy(dtype=float),
            labels,
            start_dates=start,
            end_dates=end,
        )


def _holding_window(
    h: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rebalance_col: str,
    hold_start_col: str | None,
    hold_end_col: str | None,
) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex | None]:
    out = []
    for col, agg in ((hold_start_col, "min"), (hold_end_col, "max")):
        if col and col in h.columns:
            s = pd.to_datetime(h[col]).groupby(h[rebalance_col]).agg(agg).reindex(dates)
            out.append(pd.DatetimeIndex(s))
        else:
            out.append(None)
    return out[0], out[1]


@dataclass
class SimulationResult:
    returns: pd.DataFrame
    gross: pd.DataFrame
    net: pd.DataFrame
    n_assets: pd.DataFrame
    turnover: pd.DataFrame
    cost: pd.DataFrame

    def to_long(self, portfolios: Sequence[str] | None = None, name: str = "portfolio") -> pd.DataFrame:
        """Long table (date, portfolio, ret, gross_exposure, net_exposure, n_assets) for saving;
        rows where a portfolio held no asset with a return are dropped."""
        cols = list(self.returns.columns if portfolios is None else portfolios)
        parts = {
            "ret": self.returns[cols],
            "gross_exposure": self.gross[cols],
            "net_exposure": self.net[cols],
            "n_assets": self.n_assets[cols],
        }
        out = pd.concat({k: v.stack(dropna=False) for k, v in parts.items()}, axis=1)
        out = out.rename_axis(["date", name]).reset_index()
        out["n_assets"] = out["n_assets"].fillna(0).astype(int)
        return out[out["n_assets"].gt(0)].sort_values(["date", name]).reset_index(drop=True)


def _aligned_values(frame: pd.DataFrame, columns: pd.Index) -> np.ndarray:
    """Matrix values in `columns` order, without copying when already aligned (keeps float32)."""
    if not pd.Index(frame.columns).equals(columns):
        frame = frame.reindex(columns=columns)
    return frame.to_numpy()


def simulate(
    book: Book,
    returns: pd.DataFrame,
    drift: bool = False,
    cost_bps: float = 0.0,
    weight_base: pd.DataFrame | None = None,
) -> SimulationResult:
    """Run every portfolio in `book` against a date x asset return matrix.

    drift=False holds target weights constant within a holding period (rebalanced each row).
    drift=True buys and holds: weights drift with prices until the next rebalance, and
    returns are NAV-based with NAV = 1 + sum_i w_i (G_i - 1).
    weight_base (LayerBook only): a date x asset matrix aligned with `returns`, e.g. lagged
    market equity; each row re-weights layer members by it, as Fama-French portfolios do.
    Turnover is 0.5 * sum|w_target - w_pre_trade| measured on rebalance-date targets;
    cost = sum|dw| * cost_bps is charged on the first return row of each holding period.
    Missing returns contribute zero; fold delisting returns into `returns` beforehand.
    float32 inputs are read in place; each holding period is computed in float64.
    """
    index = pd.DatetimeIndex(returns.index)
    r_all = _aligned_values(returns, book.assets)
    if weight_base is not None:
        if not isinstance(book, LayerBook):
            raise TypeError("weight_base requires a LayerBook.")
        if drift:
            raise ValueError("weight_base and drift are mutually exclusive.")
        b_all = _aligned_values(weight_base.reindex(index=index), book.assets)
    segs = book.segments(index)
    names = book.names
    t, p, k_total = len(index), len(names), len(book.dates)

    ret = np.full((t, p), np.nan)
    gross = np.full((t, p), np.nan)
    net = np.full((t, p), np.nan)
    count = np.full((t, p), np.nan)
    turnover = np.zeros((k_total, p))
    cost = np.zeros((k_total, p))
    prev = np.zeros((len(book.assets), p))

    for k, (lo, hi) in enumerate(segs):
        w = book.weights_at(k)
        turnover[k] = 0.5 * np.abs(w - prev).sum(axis=0)
        if hi <= lo:
            prev = w
            continue
        seg = r_all[lo:hi].astype(float)
        finite = np.isfinite(seg)
        seg = np.where(finite, seg, 0.0)
        count[lo:hi] = finite.astype(float) @ (w != 0)
        if weight_base is not None:
            ret[lo:hi], count[lo:hi], gross[lo:hi], net[lo:hi] = _dynamic_layers(book, k, seg, b_all[lo:hi])
            prev = w
        elif drift:
            growth = np.cumprod(1.0 + seg, axis=0)
            nav = 1.0 + growth @ w - w.sum(axis=0)
            nav_prev = np.vstack([np.ones((1, p)), nav[:-1]])
            growth_prev = np.vstack([np.ones((1, seg.shape[1])), growth[:-1]])
            ret[lo:hi] = nav / nav_prev - 1.0
            gross[lo:hi] = (growth_prev @ np.abs(w)) / nav_prev
            net[lo:hi] = (growth_prev @ w) / nav_prev
            prev = w * growth[-1][:, None] / nav[-1]
        else:
            ret[lo:hi] = seg @ w
            gross[lo:hi] = np.abs(w).sum(axis=0)
            net[lo:hi] = w.sum(axis=0)
            prev = w
        if cost_bps:
            cost[k] = 2.0 * turnover[k] * cost_bps / 10_000.0
            ret[lo] -= cost[k]

    covered = np.zeros(t, dtype=bool)
    for lo, hi in segs:
        covered[lo:hi] = True
    idx = index[covered]
    idx.name = "date"
    frame = lambda a: pd.DataFrame(a[covered], index=idx, columns=names)
    reb = book.dates.rename("rebalance_date")
    return SimulationResult(
        returns=frame(ret),
        gross=frame(gross),
        net=frame(net),
        n_assets=frame(count),
        turnover=pd.DataFrame(turnover, index=reb, columns=names),
        cost=pd.DataFrame(cost, index=reb, columns=names),
    )


def _dynamic_layers(book: LayerBook, k: int, seg: np.ndarray, base: np.ndarray):
    """Layer and spread returns when members are re-weighted every row by `base`."""
    member = book.membership(k)
    base = base.astype(float)
    base = np.where(np.isfinite(base) & (base > 0), base, 0.0)
    den = base @ member
    with np.errstate(invalid="ignore", divide="ignore"):
        layer = np.where(den > 0, ((base * seg) @ member) / den, np.nan)
    n_layer = (base > 0).astype(float) @ member
    combos = book.spread_matrix
    used = (combos != 0).astype(float)
    empty = (den <= 0).astype(float) @ used > 0
    spread = np.where(empty, np.nan, np.nan_to_num(layer) @ combos)
    alive = (den > 0).astype(float)
    ret = np.hstack([layer, spread])
    count = np.hstack([n_layer, n_layer @ used])
    gross = np.hstack([alive, np.where(empty, np.nan, alive @ np.abs(combos))])
    net = np.hstack([alive, np.where(empty, np.nan, alive @ combos)])
    return ret, count, gross, net


def forward_returns(
    returns: pd.DataFrame,
    rebalance_dates: Sequence,
    start_dates: Sequence | None = None,
    end_dates: Sequence | None = None,
) -> pd.DataFrame:
    """Compounded asset returns over each holding period, aligned to rebalance dates (K x N).

    An asset with no finite return inside a period gets NaN.
    """
    index = pd.DatetimeIndex(returns.index)
    dates = _as_index(rebalance_dates)
    segs = holding_segments(
        index,
        dates,
        None if start_dates is None else _as_index(start_dates),
        None if end_dates is None else _as_index(end_dates),
    )
    r = returns.to_numpy()
    out = np.full((len(dates), r.shape[1]), np.nan)
    for k, (lo, hi) in enumerate(segs):
        if hi <= lo:
            continue
        seg = r[lo:hi].astype(float)
        finite = np.isfinite(seg)
        growth = np.prod(np.where(finite, 1.0 + seg, 1.0), axis=0) - 1.0
        out[k] = np.where(finite.any(axis=0), growth, np.nan)
    return pd.DataFrame(out, index=dates.rename("date"), columns=returns.columns)
