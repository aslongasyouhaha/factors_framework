from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .panel import to_wide

# Shared data layout (never committed):
#   data/source/   downloaded files exactly as received (CRSP / Compustat zips, FRED csv, ...)
#   data/base/     cleaned datasets built from source by pipeline/, shared by every project
#   data/factors/  factor data built by each project: <project>/exposure.parquet + meta.json
# The data root is <repo>/data, or the FACTORLAB_DATA environment variable when set.

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FACTORLAB_DATA", REPO_ROOT / "data"))
SOURCE = DATA_ROOT / "source"
BASE = DATA_ROOT / "base"
FACTORS = DATA_ROOT / "factors"


def load_base(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read data/base/<name>.parquet."""
    path = BASE / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with the matching script in pipeline/.")
    return pd.read_parquet(path, columns=columns)


def save_base(frame: pd.DataFrame, name: str) -> Path:
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / f"{name}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def factor_dir(project: str) -> Path:
    return FACTORS / project


def save_factor(
    project: str,
    exposure: pd.DataFrame | None,
    meta: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Store a project's factor data.

    exposure: long table (date, asset_id, exposure) or a date x asset matrix, giving the
    signal at each rebalance date; larger values mean a more positive (long) view. Infinite
    values are treated as missing. None for projects that only publish tables.
    meta: description and report settings, saved as meta.json.
    tables: extra outputs such as factor return series, saved as <name>.parquet.
    """
    out = factor_dir(project)
    out.mkdir(parents=True, exist_ok=True)
    if exposure is not None:
        if isinstance(exposure.index, pd.DatetimeIndex):
            exposure = exposure.stack().rename("exposure").rename_axis(["date", "asset_id"]).reset_index()
        exposure = exposure[["date", "asset_id", "exposure"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["exposure"])
        exposure.to_parquet(out / "exposure.parquet", index=False, compression="zstd")
    for name, table in (tables or {}).items():
        table.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def load_factor(project: str, wide: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """(exposure, meta) of a project; exposure as a date x asset matrix unless wide=False."""
    path = factor_dir(project)
    if not (path / "exposure.parquet").exists():
        raise FileNotFoundError(f"No factor data for {project!r}; run factors/{project}/build.py first.")
    exposure = pd.read_parquet(path / "exposure.parquet")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return (to_wide(exposure, "exposure") if wide else exposure), meta


def load_factor_table(project: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(factor_dir(project) / f"{name}.parquet")
