from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, panels


def rolling_compound(ret: pd.Series, asset: pd.Series, skip: int, window: int, min_periods: int | None = None) -> pd.Series:
    """Compounded return by asset, ending ``skip`` observations before formation."""
    min_periods = window if min_periods is None else min_periods
    gross = (1.0 + pd.to_numeric(ret, errors="coerce")).groupby(asset).shift(skip)
    log_gross = np.log(gross.where(gross.gt(0)))
    rolled = log_gross.groupby(asset).rolling(window, min_periods=min_periods).sum().reset_index(level=0, drop=True)
    counts = gross.notna().groupby(asset).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    return (np.exp(rolled) - 1.0).where(counts.ge(min_periods))


def price_momentum() -> pd.DataFrame:
    m = panels.crsp_monthly()
    m["exposure"] = rolling_compound(m["ret"], m["asset_id"], skip=2, window=11, min_periods=8)
    return m[m["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date", "asset_id", "exposure"]]


def residual_momentum() -> pd.DataFrame:
    """FF3 residual momentum: 12-2 cumulative residual divided by its residual volatility."""
    m = panels.crsp_monthly()
    ff = data.load_base("french_factors_monthly")
    ff["date"] = pd.to_datetime(ff["date"])
    d = m[["date", "asset_id", "ret"]].merge(ff[["date", "MKT_RF", "SMB", "HML", "RF"]], on="date", how="left")
    d["excess"] = d["ret"] - d["RF"]
    # Rolling regressions produce time-varying FF3 betas. Residuals are only used after a
    # two-month shift, so the formation signal contains no current-month return.
    g = d.groupby("asset_id", sort=False)
    residual = pd.Series(np.nan, index=d.index, dtype=float)
    for _, idx in g.indices.items():
        idx = np.asarray(idx)
        y = d.loc[idx, "excess"].to_numpy(float)
        x = d.loc[idx, ["MKT_RF", "SMB", "HML"]].to_numpy(float)
        out = np.full(len(idx), np.nan)
        for i in range(23, len(idx)):
            lo = max(0, i - 35)
            yy, xx = y[lo : i + 1], x[lo : i + 1]
            good = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
            if good.sum() < 24:
                continue
            design = np.column_stack([np.ones(good.sum()), xx[good]])
            beta, *_ = np.linalg.lstsq(design, yy[good], rcond=None)
            if np.isfinite(y[i]) and np.isfinite(x[i]).all():
                out[i] = y[i] - np.r_[1.0, x[i]] @ beta
        residual.loc[idx] = out
    shifted = residual.groupby(d["asset_id"]).shift(2)
    mean = shifted.groupby(d["asset_id"]).rolling(11, min_periods=8).mean().reset_index(level=0, drop=True)
    std = shifted.groupby(d["asset_id"]).rolling(11, min_periods=8).std(ddof=1).reset_index(level=0, drop=True)
    d["exposure"] = mean / std.replace(0, np.nan)
    return d[d["date"].between(panels.SAMPLE_START, panels.SAMPLE_END)][["date", "asset_id", "exposure"]]


def cross_sectional_zscore(frame: pd.DataFrame, value: str = "exposure") -> pd.Series:
    """Winsorised (1%, 99%) cross-sectional z-score by date."""
    def one(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 20:
            return pd.Series(np.nan, index=s.index)
        lo, hi = valid.quantile([0.01, 0.99])
        clipped = s.clip(lo, hi)
        scale = clipped.std(ddof=1)
        return (clipped - clipped.mean()) / scale if scale and np.isfinite(scale) else pd.Series(np.nan, index=s.index)

    return frame.groupby("date", group_keys=False)[value].apply(one)
