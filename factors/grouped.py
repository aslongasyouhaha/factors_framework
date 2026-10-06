from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Grouped kernels on flat arrays. A long panel passes group codes built from its `by` columns;
# a date x asset matrix is flattened with one group per row (optionally x category). Both
# layouts therefore share one implementation and give identical numbers.
# Code -1 means "not in any group": the element is ignored and its output is NaN.


def group_codes(*keys: Sequence | pd.Series | np.ndarray) -> tuple[np.ndarray, int]:
    """Dense, sorted group codes for one or more aligned key arrays; any missing key -> -1."""
    if not keys:
        raise ValueError("Need at least one key.")
    if len(keys) == 1:
        codes, uniques = pd.factorize(pd.Series(np.asarray(keys[0])), sort=True)
        return codes.astype(np.int64), len(uniques)
    frame = pd.DataFrame({i: np.asarray(k) for i, k in enumerate(keys)})
    codes = frame.groupby(list(frame.columns), sort=True, dropna=True).ngroup()
    codes = codes.fillna(-1).to_numpy(dtype=np.int64)
    return codes, int(codes.max()) + 1 if len(codes) else 0


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    # numpy's interpolation formula for the "linear" method, kept bit-identical.
    diff = b - a
    return np.where(t >= 0.5, b - diff * (1 - t), a + diff * t)


def _row_width(codes: np.ndarray, n_groups: int) -> int | None:
    """Width w if codes are exactly [0]*w + [1]*w + ... (a flattened matrix, one group per row)."""
    if n_groups == 0 or len(codes) % n_groups:
        return None
    w = len(codes) // n_groups
    if not (codes[::w] == np.arange(n_groups)).all() or not (np.diff(codes) >= 0).all():
        return None
    return w if (codes[w - 1 :: w] == np.arange(n_groups)).all() else None


def _sorted_groups(values: np.ndarray, codes: np.ndarray, n_groups: int):
    """Values sorted within groups: group g occupies srt[starts[g] : starts[g] + counts[g]].

    One-group-per-row matrices use a row-wise sort; otherwise sort by value, then stably by group.
    """
    valid = np.isfinite(values) & (codes >= 0)
    w = _row_width(codes, n_groups)
    if w is not None:
        x = np.sort(np.where(valid, values, np.nan).reshape(n_groups, w), axis=1)  # NaN sorts last
        return x.ravel(), valid.reshape(n_groups, w).sum(axis=1), np.arange(n_groups) * w
    idx = np.flatnonzero(valid)
    v, c = values[idx], codes[idx]
    order = np.argsort(v)
    order = order[np.argsort(c[order], kind="stable")]
    counts = np.bincount(c, minlength=n_groups)
    return v[order], counts, np.cumsum(counts) - counts


def _group_ranks(values: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(counts, rank0): 0-based position of each element within its group, ties in input order
    (i.e. rank(method="first") - 1); -1 for missing elements."""
    valid = np.isfinite(values) & (codes >= 0)
    rank0 = np.full(len(values), -1, dtype=np.int64)
    w = _row_width(codes, n_groups)
    if w is not None:
        order = np.argsort(np.where(valid, values, np.nan).reshape(n_groups, w), axis=1, kind="stable")
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.arange(w)[None, :], axis=1)
        return valid.reshape(n_groups, w).sum(axis=1), np.where(valid, pos.ravel(), -1)
    idx = np.flatnonzero(valid)
    c = codes[idx]
    order = np.argsort(values[idx], kind="stable")
    order = order[np.argsort(c[order], kind="stable")]
    counts = np.bincount(c, minlength=n_groups)
    sorted_pos = np.empty(len(idx), dtype=np.int64)
    sorted_pos[order] = np.arange(len(idx))
    rank0[idx] = sorted_pos - (np.cumsum(counts) - counts)[c]
    return counts, rank0


def group_quantiles(values: np.ndarray, codes: np.ndarray, n_groups: int, qs: Sequence[float]) -> np.ndarray:
    """Per-group quantiles identical to pandas Series.quantile (linear); shape (len(qs), n_groups)."""
    srt, counts, starts = _sorted_groups(np.asarray(values, dtype=float), codes, n_groups)
    out = np.full((len(qs), n_groups), np.nan)
    has = counts > 0
    for i, q in enumerate(qs):
        q_eff = np.true_divide(q * 100.0, 100)  # pandas routes through np.percentile(q * 100)
        pos = (counts[has] - 1) * q_eff
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, counts[has] - 1)
        out[i, has] = _lerp(srt[starts[has] + lo], srt[starts[has] + hi], pos - lo)
    return out


def group_median(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group median identical to pandas Series.median."""
    srt, counts, starts = _sorted_groups(np.asarray(values, dtype=float), codes, n_groups)
    out = np.full(n_groups, np.nan)
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    out[has] = (srt[lo] + srt[hi]) / 2
    return out


def group_moments(values: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group count, mean and sample std (ddof=1), ignoring NaN."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (codes >= 0)
    c, v = codes[valid], values[valid]
    n = np.bincount(c, minlength=n_groups).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(c, weights=v, minlength=n_groups) / n
        dev = v - mean[c]
        var = np.bincount(c, weights=dev * dev, minlength=n_groups) / (n - 1)
    mean[n == 0] = np.nan
    var[n < 2] = np.nan
    return n, mean, np.sqrt(var)


def group_bounds(
    values: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    method: str = "quantile",
    lower: float | None = 0.01,
    upper: float | None = 0.99,
    n_std: float = 3.0,
    n_mad: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group outlier bounds; groups where a bound is undefined get (-inf, inf)."""
    values = np.asarray(values, dtype=float)
    method = method.lower()
    if method in {"std", "mean_std", "zscore"}:
        _, mean, std = group_moments(values, codes, n_groups)
        ok = np.isfinite(std) & (std != 0)
        lo, hi = mean - n_std * std, mean + n_std * std
    elif method in {"mad", "median_mad", "absolute_median"}:
        med = group_median(values, codes, n_groups)
        dev = np.abs(values - np.where(codes >= 0, med[np.maximum(codes, 0)], np.nan))
        mad = group_median(dev, codes, n_groups)
        ok = np.isfinite(mad) & (mad != 0)
        scaled = 1.4826 * mad
        lo, hi = med - n_mad * scaled, med + n_mad * scaled
    elif method in {"iqr", "box", "boxplot", "box_plot"}:
        q1, q3 = group_quantiles(values, codes, n_groups, (0.25, 0.75))
        iqr = q3 - q1
        ok = np.isfinite(iqr) & (iqr != 0)
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    elif method in {"quantile", "percentile"}:
        lo, hi = group_quantiles(values, codes, n_groups, (0.0 if lower is None else lower, 1.0 if upper is None else upper))
        ok = np.isfinite(lo) & np.isfinite(hi)
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    return np.where(ok, lo, -np.inf), np.where(ok, hi, np.inf)


def group_qcut(values: np.ndarray, codes: np.ndarray, n_groups: int, n_bins: int, min_count: int | None = None) -> np.ndarray:
    """Equal-count bin per element within its group, 0..n_bins-1; -1 if missing or group too small.

    Identical to pd.qcut(s.rank(method="first"), n_bins) applied group by group. Groups with
    fewer than `min_count` members (default n_bins) are left unassigned.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    values = np.asarray(values, dtype=float)
    counts, rank0 = _group_ranks(values, codes, n_groups)
    valid = rank0 >= 0
    ranks = rank0 + 1.0  # rank(method="first")
    g = np.maximum(codes, 0)
    # qcut edges on ranks 1..n are np.percentile(1..n, 100*q); reproduce their rounding exactly.
    # They depend only on the group size, so compute them per group and broadcast.
    n = counts.astype(float)
    out = np.zeros(len(values), dtype=np.int16)
    for q in np.linspace(0.0, 1.0, n_bins + 1)[1:-1]:
        pos = (n - 1) * np.true_divide(q * 100.0, 100)
        lo = np.floor(pos)
        edge = _lerp(lo + 1, np.minimum(lo + 2, np.maximum(n, 1)), pos - lo)
        out += ranks > edge[g]
    enough = counts >= (n_bins if min_count is None else min_count)
    return np.where(valid & enough[g], out, -1).astype(np.int16)


def demean(
    values: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    category_codes: Sequence[np.ndarray] = (),
    tol: float = 1e-13,
    max_iter: int = 1000,
) -> np.ndarray:
    """Remove group means, or (group x category) fixed effects for each category key.

    With one category this is exact in one pass; with several it alternates projections
    until convergence, which equals OLS on intercept + dummies for every category.
    """
    x = np.asarray(values, dtype=float).copy()
    keys = [codes] if not category_codes else [
        group_codes(codes, cat)[0] for cat in category_codes
    ]
    for _ in range(max_iter if len(keys) > 1 else 1):
        change = 0.0
        for key in keys:
            ok = key >= 0
            n_key = int(key.max()) + 1 if ok.any() else 0
            sums = np.bincount(key[ok], weights=x[ok], minlength=n_key)
            cnt = np.bincount(key[ok], minlength=n_key)
            means = sums / np.maximum(cnt, 1)
            x[ok] -= means[key[ok]]
            change = max(change, float(np.abs(means).max(initial=0.0)))
        if change < tol:
            break
    return x


def residualize(
    y: np.ndarray,
    x: np.ndarray | None,
    codes: np.ndarray,
    n_groups: int,
    category_codes: Sequence[np.ndarray] = (),
    add_intercept: bool = True,
    min_obs: int = 20,
) -> np.ndarray:
    """Per-group OLS residuals of y on [intercept] + x + category dummies.

    Category fixed effects are swept out by demeaning (Frisch-Waugh-Lovell), so dummies are
    never materialised; the remaining small regression is solved for all groups at once.
    Rows with any missing input, or groups with fewer than min_obs rows, get NaN.
    """
    y = np.asarray(y, dtype=float)
    x = np.zeros((len(y), 0)) if x is None else np.asarray(x, dtype=float).reshape(len(y), -1)
    if category_codes and not add_intercept:
        raise ValueError("Category neutralization requires add_intercept=True.")
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1) & (codes >= 0)
    for cat in category_codes:
        mask &= np.asarray(cat) >= 0
    g = np.where(mask, codes, -1)
    counts = np.bincount(g[mask], minlength=n_groups)
    mask &= counts[np.maximum(g, 0)] >= min_obs
    g = np.where(mask, codes, -1)
    cats = [np.where(mask, np.asarray(cat), -1) for cat in category_codes]

    if add_intercept or cats:
        yd = demean(np.where(mask, y, 0.0), g, n_groups, cats)
        xd = np.column_stack([demean(np.where(mask, x[:, j], 0.0), g, n_groups, cats) for j in range(x.shape[1])]) if x.shape[1] else x
    else:
        yd, xd = np.where(mask, y, 0.0), np.where(mask[:, None], x, 0.0)

    resid = yd
    p = xd.shape[1]
    if p:
        gm = g[mask]
        xm, ym = xd[mask], yd[mask]
        xtx = np.zeros((n_groups, p, p))
        xty = np.zeros((n_groups, p))
        for i in range(p):
            xty[:, i] = np.bincount(gm, weights=xm[:, i] * ym, minlength=n_groups)
            for j in range(i, p):
                s = np.bincount(gm, weights=xm[:, i] * xm[:, j], minlength=n_groups)
                xtx[:, i, j] = xtx[:, j, i] = s
        # Scale columns per group so regressors on very different scales stay well conditioned.
        d = np.sqrt(np.einsum("gii->gi", xtx))
        d = np.where(d > 0, d, 1.0)
        scaled = xtx / (d[:, :, None] * d[:, None, :])
        beta = np.einsum("gij,gj->gi", np.linalg.pinv(scaled, rcond=1e-12, hermitian=True), xty / d) / d
        resid = yd - np.einsum("np,np->n", xd, beta[np.maximum(g, 0)])
    return np.where(mask, resid, np.nan)
