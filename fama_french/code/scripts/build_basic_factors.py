from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(CODE_ROOT))

from factors import FactorBacktester, FactorConfig, QuantileSignalFactorBuilder, forward_returns, to_wide


RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "basic_factors"
START_YEAR = 2017
END_YEAR = 2025


FACTOR_SPECS = [
    {"name": "SIZE", "signal_col": "size_signal", "label": "Small minus big; signal = -log(ME)", "panel": "monthly"},
    {"name": "BM", "signal_col": "bm", "label": "High BM minus low BM", "panel": "annual_chars"},
    {"name": "MOM_12_2", "signal_col": "mom_12_2", "label": "Prior 12-to-2 month winner minus loser", "panel": "monthly"},
    {"name": "REV_1M", "signal_col": "rev_1m", "label": "Prior one-month loser minus winner", "panel": "monthly"},
    {"name": "LOW_VOL_12M", "signal_col": "low_vol_12m", "label": "Low trailing 12-month volatility minus high volatility", "panel": "monthly"},
    {"name": "ILLIQ", "signal_col": "amihud_illiq", "label": "High Amihud illiquidity minus low illiquidity", "panel": "monthly"},
    {"name": "GROSS_PROFIT", "signal_col": "gross_profitability", "label": "Gross profitability = TTM gross profit / assets", "panel": "financial"},
    {"name": "OP_PROFIT", "signal_col": "operating_profitability", "label": "Operating profitability = TTM operating income / assets", "panel": "financial"},
    {"name": "ROA", "signal_col": "roa", "label": "Return on assets = TTM net income / assets", "panel": "financial"},
    {"name": "EARNINGS_YIELD", "signal_col": "earnings_yield", "label": "Earnings yield = TTM net income / December market equity", "panel": "financial"},
    {"name": "ASSET_GROWTH", "signal_col": "asset_growth_signal", "label": "Conservative minus aggressive investment; signal = -asset growth", "panel": "financial"},
    {"name": "SALES_GROWTH", "signal_col": "sales_growth", "label": "High TTM sales growth minus low sales growth", "panel": "financial"},
    {"name": "LOW_LEVERAGE", "signal_col": "low_leverage", "label": "Low debt/assets minus high debt/assets; signal = -leverage", "panel": "financial"},
    {"name": "CASH", "signal_col": "cash_to_assets", "label": "High cash/assets minus low cash/assets", "panel": "financial"},
    {"name": "LOW_ACCRUALS", "signal_col": "low_accruals", "label": "Low accruals minus high accruals; signal = -accruals/assets", "panel": "financial"},
    {"name": "LOW_CAPEX", "signal_col": "low_capex_to_assets", "label": "Low capex/assets minus high capex/assets; signal = -capex/assets", "panel": "financial"},
]


def load_monthly() -> pd.DataFrame:
    monthly = pd.read_parquet(RESULTS / "crsp_monthly_permco_2017_2025.parquet")
    monthly = monthly[monthly["ff_primary_share"]].copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["ret"] = pd.to_numeric(monthly["ret"], errors="coerce")
    monthly["market_equity"] = pd.to_numeric(monthly["market_equity"], errors="coerce")
    monthly["dollar_volume"] = pd.to_numeric(monthly["dollar_volume"], errors="coerce")
    return monthly.sort_values(["asset_id", "date"])


def rolling_compound(ret: pd.Series, asset: pd.Series, skip: int, window: int, min_periods: int) -> pd.Series:
    """Per-asset compounded return over `window` rows ending `skip` rows back.

    Same result as rolling(window, min_periods).apply(np.prod) on (1 + ret).shift(skip), i.e.
    NaN whenever the window holds a missing return, but computed as a rolling log-sum.
    """
    gross = (1.0 + ret).groupby(asset).shift(skip)
    missing = gross.isna().astype(float)
    wiped = gross.le(0).astype(float)
    log_gross = np.log(gross.where(gross.gt(0), 1.0)).where(gross.notna())

    def roll(s: pd.Series, mp: int) -> pd.Series:
        return s.groupby(asset).rolling(window, min_periods=mp).sum().reset_index(level=0, drop=True)

    out = np.exp(roll(log_gross, min_periods)) - 1.0
    out = out.where(roll(wiped, 1).eq(0), -1.0).where(out.notna())
    return out.where(roll(missing, 1).eq(0))


def monthly_signal_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    g = m.groupby("asset_id", group_keys=False)
    m["mom_12_2"] = rolling_compound(m["ret"], m["asset_id"], skip=2, window=11, min_periods=8)
    m["rev_1m"] = -m["ret"]
    m["low_vol_12m"] = -g["ret"].transform(lambda x: x.rolling(12, min_periods=8).std())
    m["size_signal"] = -np.log(m["market_equity"].where(m["market_equity"].gt(0)))
    m["amihud_illiq"] = m["ret"].abs() / m["dollar_volume"].where(m["dollar_volume"].gt(0))
    m["rebalance_date"] = m["date"]
    m["hold_start"] = m["date"] + pd.offsets.MonthEnd(1)
    m["hold_end"] = m["hold_start"]
    start_rebalance = pd.Timestamp(f"{START_YEAR}-06-30")
    end_rebalance = pd.Timestamp(f"{END_YEAR}-11-30")
    m = m[m["rebalance_date"].between(start_rebalance, end_rebalance)].copy()
    cols = ["rebalance_date", "hold_start", "hold_end", "asset_id", "market_equity", "mom_12_2", "rev_1m", "low_vol_12m", "size_signal", "amihud_illiq"]
    return m[cols]


def annual_characteristic_panel() -> pd.DataFrame:
    chars = pd.read_parquet(RESULTS / "ff3_characteristics_2017_2025.parquet")
    chars = chars.copy()
    chars["rebalance_date"] = pd.to_datetime(chars["rebalance_date"])
    chars["hold_start"] = chars["rebalance_date"] + pd.offsets.Day(1)
    chars["hold_end"] = chars["rebalance_date"] + pd.DateOffset(years=1)
    return chars[["rebalance_date", "hold_start", "hold_end", "asset_id", "market_equity", "bm", "dec_market_equity"]]


def quarterly_financial_descriptors(chunksize: int = 500_000) -> pd.DataFrame:
    cache = OUT / f"compustat_q4_financial_descriptors_{START_YEAR}_{END_YEAR}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    qzip = RAW / "Comp_Quarterly6126.csv.zip"
    cols = [
        "gvkey",
        "datadate",
        "fyearq",
        "fqtr",
        "indfmt",
        "consol",
        "datafmt",
        "atq",
        "ltq",
        "seqq",
        "saleq",
        "cogsq",
        "niq",
        "ibq",
        "oiadpq",
        "cheq",
        "dlttq",
        "dlcq",
        "actq",
        "lctq",
        "dpq",
        "capxy",
    ]
    parts = []
    with zipfile.ZipFile(qzip) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")][0]
        with z.open(csv_name) as f:
            for chunk in pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False):
                chunk = chunk[
                    chunk["fyearq"].between(START_YEAR - 3, END_YEAR)
                    & chunk["indfmt"].eq("INDL")
                    & chunk["consol"].eq("C")
                    & chunk["datafmt"].eq("STD")
                ].copy()
                if chunk.empty:
                    continue
                parts.append(chunk)
    if not parts:
        raise RuntimeError("No Compustat quarterly rows for financial descriptors.")

    q = pd.concat(parts, ignore_index=True)
    q["gvkey"] = q["gvkey"].astype("string").str.zfill(6)
    q["datadate"] = pd.to_datetime(q["datadate"])
    numeric_cols = [c for c in cols if c not in {"gvkey", "datadate", "indfmt", "consol", "datafmt"}]
    for col in numeric_cols:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.sort_values(["gvkey", "datadate"])

    q["gross_profit_q"] = q["saleq"] - q["cogsq"]
    q["net_income_q"] = q["niq"].combine_first(q["ibq"])
    flow_cols = ["saleq", "gross_profit_q", "net_income_q", "oiadpq", "dpq"]
    for col in flow_cols:
        q[f"{col}_ttm"] = q.groupby("gvkey")[col].transform(lambda x: x.rolling(4, min_periods=4).sum())

    q4 = q[q["fqtr"].eq(4)].copy()
    q4["lag_atq"] = q4.groupby("gvkey")["atq"].shift(1)
    q4["lag_saleq_ttm"] = q4.groupby("gvkey")["saleq_ttm"].shift(1)
    q4["lag_actq"] = q4.groupby("gvkey")["actq"].shift(1)
    q4["lag_cheq"] = q4.groupby("gvkey")["cheq"].shift(1)
    q4["lag_lctq"] = q4.groupby("gvkey")["lctq"].shift(1)
    q4["lag_dlcq"] = q4.groupby("gvkey")["dlcq"].shift(1)
    avg_assets = (q4["atq"] + q4["lag_atq"]) / 2.0
    debt = q4["dlttq"].fillna(0.0) + q4["dlcq"].fillna(0.0)
    delta_current_assets = q4["actq"] - q4["lag_actq"]
    delta_cash = q4["cheq"] - q4["lag_cheq"]
    delta_current_liab = q4["lctq"] - q4["lag_lctq"]
    delta_short_debt = q4["dlcq"] - q4["lag_dlcq"]
    accruals = delta_current_assets - delta_cash - (delta_current_liab - delta_short_debt) - q4["dpq_ttm"].fillna(0.0)

    out = pd.DataFrame(
        {
            "gvkey": q4["gvkey"],
            "datadate": q4["datadate"],
            "fyear": q4["fyearq"].astype("Int64"),
            "gross_profitability": q4["gross_profit_q_ttm"] / q4["atq"],
            "operating_profitability": q4["oiadpq_ttm"] / q4["atq"],
            "roa": q4["net_income_q_ttm"] / q4["atq"],
            "asset_growth": q4["atq"] / q4["lag_atq"] - 1.0,
            "asset_growth_signal": -(q4["atq"] / q4["lag_atq"] - 1.0),
            "sales_growth": q4["saleq_ttm"] / q4["lag_saleq_ttm"] - 1.0,
            "leverage": debt / q4["atq"],
            "low_leverage": -(debt / q4["atq"]),
            "cash_to_assets": q4["cheq"] / q4["atq"],
            "accruals_to_assets": accruals / avg_assets,
            "low_accruals": -(accruals / avg_assets),
            "capex_to_assets": q4["capxy"] / q4["atq"],
            "low_capex_to_assets": -(q4["capxy"] / q4["atq"]),
            "net_income_ttm": q4["net_income_q_ttm"],
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[out["fyear"].notna()].copy()
    out["fyear"] = out["fyear"].astype(int)
    out = out.sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    out.to_parquet(cache, index=False, compression="zstd")
    return out


def load_link_table() -> pd.DataFrame:
    link = pd.read_parquet(RESULTS / "ccm_link_filtered.parquet")
    link = link.copy()
    link["gvkey"] = link["gvkey"].astype("string").str.zfill(6)
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"])
    return link


def financial_factor_panel(chars: pd.DataFrame) -> pd.DataFrame:
    desc = quarterly_financial_descriptors()
    link = load_link_table()
    linked = desc.merge(link, on="gvkey", how="inner")
    linked = linked[linked["datadate"].between(linked["linkdt"], linked["linkenddt"])]
    linked = linked.sort_values(["asset_id", "fyear", "datadate"]).drop_duplicates(["asset_id", "fyear"], keep="last")

    c = chars.copy()
    c["be_year"] = pd.to_datetime(c["rebalance_date"]).dt.year - 1
    out = c.merge(linked.rename(columns={"fyear": "be_year"}), on=["asset_id", "be_year"], how="inner")
    out["earnings_yield"] = out["net_income_ttm"] * 1000.0 / out["dec_market_equity"].where(out["dec_market_equity"].gt(0))
    signal_cols = [spec["signal_col"] for spec in FACTOR_SPECS if spec["panel"] == "financial"]
    cols = ["rebalance_date", "hold_start", "hold_end", "asset_id", "market_equity", *signal_cols]
    return out[cols].replace([np.inf, -np.inf], np.nan)


def build_one(spec: dict[str, str], panel: pd.DataFrame, returns: pd.DataFrame, bt: FactorBacktester) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one decile factor on date x asset matrices; `returns` is the monthly return matrix."""
    name = spec["name"]
    cfg = FactorConfig(
        name=name,
        frequency="monthly" if spec["panel"] == "monthly" else "annual",
        n_groups=10,
        weighting="value",
        long_groups=("D10",),
        short_groups=("D01",),
    )
    builder = QuantileSignalFactorBuilder(signal_source_col=spec["signal_col"], config=cfg, winsorize=True, winsor_lower=0.01, winsor_upper=0.99)
    p = panel.dropna(subset=[spec["signal_col"]])
    signal = to_wide(p, spec["signal_col"], date_col="rebalance_date")
    market_equity = to_wide(p, "market_equity", date_col="rebalance_date")
    window = p.groupby("rebalance_date").agg(start=("hold_start", "min"), end=("hold_end", "max")).reindex(signal.index)
    factor = builder.build_wide(signal, market_equity, start_dates=window["start"], end_dates=window["end"])
    book = factor.book

    sim = bt.simulate(book, returns)
    layer_returns = sim.to_long(builder.group_labels, name="group")[["date", "group", "ret", "n_assets"]]
    spread_returns = sim.to_long([name]).drop(columns="portfolio")
    turnover = bt.compute_turnover(book, name)
    turnover_ppy = 12 if spec["panel"] == "monthly" else 1
    summary = bt.summary(spread_returns["ret"], turnover, turnover_periods_per_year=turnover_ppy).rename(name)

    # IC of the winsorised signal of held stocks against the following month's return.
    held = book.codes >= 0
    ic_signal = factor.signal.where(held)[held.any(axis=1)]
    next_month = forward_returns(returns, ic_signal.index, end_dates=ic_signal.index + pd.offsets.MonthEnd(1))
    ic = bt.ic_wide(ic_signal, next_month).reset_index()
    r2 = bt.r2_wide(ic_signal, next_month).reset_index()
    ic_summary = pd.Series(
        {
            "mean_ic": ic["ic"].mean(),
            "mean_rank_ic": ic["rank_ic"].mean(),
            "mean_cos_ic": ic["cos_ic"].mean(),
            "ic_ir": ic["ic"].mean() / ic["ic"].std(ddof=1),
            "rank_ic_ir": ic["rank_ic"].mean() / ic["rank_ic"].std(ddof=1),
            "cos_ic_ir": ic["cos_ic"].mean() / ic["cos_ic"].std(ddof=1),
            "mean_r2": r2["r2"].mean(),
        },
        name=name,
    )

    holdings = book.to_long(extras={"signal_value": factor.signal, "market_equity": market_equity})
    decile_holdings = holdings[holdings["portfolio"].ne(name)].rename(columns={"portfolio": "group"})
    spread_holdings = holdings[holdings["portfolio"].eq(name)].drop(columns="portfolio")
    spread_holdings.insert(1, "leg", np.where(spread_holdings["target_weight"].gt(0), "long", "short"))

    stem = name.lower()
    decile_holdings.to_parquet(OUT / f"{stem}_decile_holdings.parquet", index=False, compression="zstd")
    spread_holdings.to_parquet(OUT / f"{stem}_spread_holdings.parquet", index=False, compression="zstd")
    layer_returns.assign(factor=name).to_csv(OUT / f"{stem}_layer_returns.csv", index=False)
    spread_returns.assign(factor=name).to_csv(OUT / f"{stem}_spread_returns.csv", index=False)
    turnover.to_csv(OUT / f"{stem}_turnover.csv", index=False)
    ic.assign(factor=name).to_csv(OUT / f"{stem}_ic.csv", index=False)
    r2.assign(factor=name).to_csv(OUT / f"{stem}_r2.csv", index=False)
    return summary, ic_summary, spread_returns.assign(factor=name), layer_returns.assign(factor=name), ic.assign(factor=name)


def write_plots(spread_returns: pd.DataFrame, layer_returns: pd.DataFrame, ic: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plotting.") from exc

    fig = go.Figure()
    for factor, g in spread_returns.groupby("factor", sort=True):
        g = g.sort_values("date").copy()
        g["cum_return"] = (1.0 + g["ret"]).cumprod() - 1.0
        fig.add_trace(go.Scatter(x=pd.to_datetime(g["date"]).to_numpy(dtype="datetime64[ns]"), y=g["cum_return"].to_numpy(), name=factor))
    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(template="plotly_white", height=650, title="Basic factor spread cumulative returns")
    fig.write_html(OUT / "basic_factor_spread_curves.html", include_plotlyjs="cdn")

    for factor, g in layer_returns.groupby("factor", sort=True):
        fig = go.Figure()
        for group, gg in g.groupby("group", observed=True):
            gg = gg.sort_values("date").copy()
            gg["cum_return"] = (1.0 + gg["ret"]).cumprod() - 1.0
            fig.add_trace(go.Scatter(x=pd.to_datetime(gg["date"]).to_numpy(dtype="datetime64[ns]"), y=gg["cum_return"].to_numpy(), name=str(group)))
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(template="plotly_white", height=560, title=f"{factor} decile layer returns")
        fig.write_html(OUT / f"{factor.lower()}_layer_curves.html", include_plotlyjs="cdn")

    for factor, g in ic.groupby("factor", sort=True):
        fig = go.Figure()
        for col in ["cum_ic", "cum_rank_ic", "cum_cos_ic"]:
            fig.add_trace(go.Scatter(x=pd.to_datetime(g["date"]).to_numpy(dtype="datetime64[ns]"), y=g[col].to_numpy(), name=col))
        fig.update_layout(template="plotly_white", height=500, title=f"{factor} cumulative IC")
        fig.write_html(OUT / f"{factor.lower()}_cumulative_ic.html", include_plotlyjs="cdn")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bt = FactorBacktester(periods_per_year=12)
    monthly = load_monthly()
    monthly_panel = monthly_signal_panel(monthly)
    annual_panel = annual_characteristic_panel()
    financial_panel = financial_factor_panel(annual_panel)
    returns = to_wide(monthly, "ret")
    panels = {"monthly": monthly_panel, "annual_chars": annual_panel, "financial": financial_panel}

    summaries = []
    ic_summaries = []
    spread_parts = []
    layer_parts = []
    ic_parts = []
    for spec in FACTOR_SPECS:
        summary, ic_summary, spread_returns, layer_returns, ic = build_one(spec, panels[spec["panel"]], returns, bt)
        summary["description"] = spec["label"]
        summaries.append(summary)
        ic_summaries.append(ic_summary)
        spread_parts.append(spread_returns)
        layer_parts.append(layer_returns)
        ic_parts.append(ic)

    summary = pd.concat(summaries, axis=1).T
    ic_summary = pd.concat(ic_summaries, axis=1).T
    spread_returns = pd.concat(spread_parts, ignore_index=True)
    layer_returns = pd.concat(layer_parts, ignore_index=True)
    ic = pd.concat(ic_parts, ignore_index=True)

    summary.to_csv(OUT / "basic_factor_summary.csv")
    ic_summary.to_csv(OUT / "basic_factor_ic_r2_summary.csv")
    spread_returns.to_csv(OUT / "basic_factor_spread_returns.csv", index=False)
    layer_returns.to_csv(OUT / "basic_factor_layer_returns.csv", index=False)
    ic.to_csv(OUT / "basic_factor_ic.csv", index=False)
    write_plots(spread_returns, layer_returns, ic)

    cols = ["ann_return", "ann_vol", "sharpe", "total_return", "max_drawdown", "hit_rate", "avg_turnover"]
    print("Wrote basic factors under", OUT)
    print(summary[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nIC/R2")
    print(ic_summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
