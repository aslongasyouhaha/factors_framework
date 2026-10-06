from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .grouped import group_bounds, group_codes, group_moments, group_qcut, group_quantiles, residualize

# Cross-sectional operations on date x asset matrices. Each row is one cross-section; passing
# `groups` (a date x asset matrix of category labels, e.g. industry) splits every row further.
# All of them run through the flat kernels in grouped.py, without a Python loop over dates.


def to_wide(
    frame: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    asset_col: str = "asset_id",
    dtype: str | type = float,
) -> pd.DataFrame:
    """Pivot a long panel to a date x asset matrix; duplicate (date, asset) keys are an error."""
    if isinstance(frame.index, pd.MultiIndex) and date_col not in frame.columns:
        frame = frame.reset_index()
    data = frame[[date_col, asset_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    dupes = data.duplicated([date_col, asset_col])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate ({date_col}, {asset_col}) rows in {value_col!r}.")
    wide = data.pivot(index=date_col, columns=asset_col, values=value_col).sort_index()
    wide.index.name = "date"
    wide.columns.name = "asset_id"
    return wide.astype(dtype)


def to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = wide.stack().rename(value_name).reset_index()
    return out.rename(columns={out.columns[0]: "date", out.columns[1]: "asset_id"})


def _like(wide: pd.DataFrame, flat: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(flat.reshape(wide.shape), index=wide.index, columns=wide.columns)


def _codes(wide: pd.DataFrame, groups=None) -> tuple[np.ndarray, int]:
    """Group code per flattened cell: its row, or (row, category) when groups are given."""
    t, n = wide.shape
    rows = np.repeat(np.arange(t, dtype=np.int64), n)
    cats = _category_codes(wide, groups)
    if not cats:
        return rows, t
    return group_codes(rows, *cats)


def _category_codes(wide: pd.DataFrame, groups) -> list[np.ndarray]:
    if groups is None:
        return []
    mats = [groups] if isinstance(groups, pd.DataFrame) else list(groups)
    out = []
    for g in mats:
        g = g.reindex(index=wide.index, columns=wide.columns)
        codes, _ = pd.factorize(pd.Series(g.to_numpy().ravel()), sort=True)
        out.append(codes.astype(np.int64))
    return out


def row_quantiles(wide: pd.DataFrame | np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """Per-row quantiles identical to pandas Series.quantile; shape (len(qs), n_rows)."""
    x = wide.to_numpy(dtype=float) if isinstance(wide, pd.DataFrame) else np.asarray(wide, dtype=float)
    rows = np.repeat(np.arange(x.shape[0], dtype=np.int64), x.shape[1])
    return group_quantiles(x.ravel(), rows, x.shape[0], qs)


def outlier_mask_and_clip(
    wide: pd.DataFrame,
    method: str = "quantile",
    lower: float | None = 0.01,
    upper: float | None = 0.99,
    n_std: float = 3.0,
    n_mad: float = 3.5,
    iqr_multiplier: float = 1.5,
    groups=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (values clipped to per-cross-section bounds, boolean outlier flags)."""
    x = wide.to_numpy(dtype=float).ravel()
    codes, n_groups = _codes(wide, groups)
    lo, hi = group_bounds(x, codes, n_groups, method, lower, upper, n_std, n_mad, iqr_multiplier)
    idx = np.maximum(codes, 0)
    lo_x = np.where(codes >= 0, lo[idx], -np.inf)
    hi_x = np.where(codes >= 0, hi[idx], np.inf)
    with np.errstate(invalid="ignore"):
        flags = (x < lo_x) | (x > hi_x)
    clipped = np.where(np.isfinite(x), np.clip(x, lo_x, hi_x), x)
    return _like(wide, clipped), _like(wide, flags).astype(bool)


def winsorize_rows(wide: pd.DataFrame, lower: float = 0.01, upper: float = 0.99, groups=None) -> pd.DataFrame:
    clipped, _ = outlier_mask_and_clip(wide, "quantile", lower, upper, groups=groups)
    return clipped


def zscore_rows(wide: pd.DataFrame, min_std: float = 1e-12, groups=None) -> pd.DataFrame:
    x = wide.to_numpy(dtype=float).ravel()
    codes, n_groups = _codes(wide, groups)
    _, mean, std = group_moments(x, codes, n_groups)
    std = np.where(np.abs(std) > min_std, std, np.nan)
    idx = np.maximum(codes, 0)
    z = np.where(codes >= 0, (x - mean[idx]) / std[idx], np.nan)
    return _like(wide, z)


def fill_missing(
    wide: pd.DataFrame,
    method: str,
    universe: pd.DataFrame | None = None,
    limit: int | None = None,
    fill_value: float | None = None,
    groups=None,
) -> pd.DataFrame:
    """Fill NaN cells inside the universe.

    method: "mean" / "median" (cross-sectional, per row or row x group), "zero", "value",
    "ffill" / "bfill" (along time per asset; `limit` counts universe rows, as a long-panel
    groupby(asset).ffill(limit) would). Cells outside `universe` stay NaN.
    """
    x = wide.to_numpy(dtype=float)
    uni = np.ones(x.shape, dtype=bool) if universe is None else universe.reindex_like(wide).fillna(False).to_numpy(dtype=bool)
    valid = np.isfinite(x) & uni
    method = method.lower()
    if method in {"zero", "value"}:
        fill = np.full(x.shape, 0.0 if method == "zero" else fill_value, dtype=float)
    elif method in {"mean", "median"}:
        codes, n_groups = _codes(wide, groups)
        flat = np.where(valid, x, np.nan).ravel()
        if method == "mean":
            _, stat, _ = group_moments(flat, codes, n_groups)
        else:
            stat = group_quantiles(flat, codes, n_groups, (0.5,))[0]
        fill = np.where(codes >= 0, stat[np.maximum(codes, 0)], np.nan).reshape(x.shape)
    elif method in {"ffill", "bfill"}:
        step = slice(None) if method == "ffill" else slice(None, None, -1)
        xs, us, vs = x[step], uni[step], valid[step]
        filled = pd.DataFrame(np.where(vs, xs, np.nan)).ffill().to_numpy()
        if limit is not None:
            cu = np.cumsum(us, axis=0)
            last = pd.DataFrame(np.where(vs, cu, np.nan)).ffill().to_numpy()
            filled = np.where(cu - last <= limit, filled, np.nan)
        fill = filled[step]
    else:
        raise ValueError(f"Unknown missing-value method: {method}")
    out = np.where(valid, x, np.where(uni, fill, np.nan))
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def neutralize_rows(
    y: pd.DataFrame,
    x: dict[str, pd.DataFrame] | Sequence[pd.DataFrame] | None = None,
    categories=None,
    add_intercept: bool = True,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Cross-sectional OLS residuals of y on [intercept] + x + category dummies, per row."""
    mats = list(x.values()) if isinstance(x, dict) else list(x or [])
    xs = np.column_stack([m.reindex_like(y).to_numpy(dtype=float).ravel() for m in mats]) if mats else None
    t, n = y.shape
    rows = np.repeat(np.arange(t, dtype=np.int64), n)
    resid = residualize(
        y.to_numpy(dtype=float).ravel(),
        xs,
        rows,
        t,
        category_codes=_category_codes(y, categories),
        add_intercept=add_intercept,
        min_obs=min_obs,
    )
    return _like(y, resid)


def quantile_codes(wide: pd.DataFrame, n_groups: int, min_assets: int | None = None) -> np.ndarray:
    """Equal-count bucket codes 0..n_groups-1 per row; -1 for missing or thin rows.

    Matches pd.qcut(s.rank(method="first"), n_groups) applied to each row.
    """
    x = wide.to_numpy(dtype=float)
    rows = np.repeat(np.arange(x.shape[0], dtype=np.int64), x.shape[1])
    min_count = max(n_groups, min_assets or 0)
    return group_qcut(x.ravel(), rows, x.shape[0], n_groups, min_count).reshape(x.shape)
