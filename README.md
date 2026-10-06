# factors

美股（以及后续 A 股）横截面因子研究框架：因子构建、截面预处理、分层回测与因子评价。

所有计算都以 **date × asset 宽矩阵**为核心：持仓只在调仓日存一份 K×N 矩阵，回测按持有期切片做矩阵乘法，截面处理（去极值、标准化、中性化、分位分组）对所有日期一次算完，不按日期写 Python 循环。日频 6,500 天 × 3,000 只、每日调仓的完整流程约 1 分钟。

## 安装

```bash
pip install -e ".[plot,dev]"
pytest
```

## 快速开始

```python
import numpy as np
from factors import (FactorBacktester, FactorConfig, QuantileSignalFactorBuilder,
                     forward_returns, neutralize_rows, outlier_mask_and_clip, to_wide, zscore_rows)

R  = to_wide(daily, "ret")                 # 收益：行 = 日期，列 = 股票
me = to_wide(signals, "market_equity")     # 调仓日市值
x  = to_wide(signals, "mom")               # 调仓日的原始信号
industry = to_wide(signals, "industry", dtype=object)   # 行业标签矩阵

x, _ = outlier_mask_and_clip(x, "mad")     # 截面去极值
x = neutralize_rows(x, {"size": np.log(me)}, categories=industry)   # 市值 + 行业中性化
x = zscore_rows(x)

cfg = FactorConfig(name="MOM", n_groups=10, weighting="value", long_groups=("D10",), short_groups=("D01",))
wf  = QuantileSignalFactorBuilder(config=cfg, winsorize=False).build_wide(x, me)

res = FactorBacktester(periods_per_year=252).run(
    wf.book, R, signal_panel=wf.signal, forward=forward_returns(R, wf.book.dates), drift=True)
res.summary         # 各层与多空组合的年化收益、Sharpe、回撤、换手
res.layer_returns   # D01–D10 日收益
res.ic              # 每期 IC / Rank IC / cos IC
```

原有的长表接口（`FactorBuildTools.handle_outliers`、`FactorBacktester.portfolio_returns(holdings, returns)` 等）仍然可用，内部走同一套计算。

## 结构

| 模块 | 内容 |
|---|---|
| `factors/engine.py` | 持仓簿 `WeightBook` / `LayerBook`，回测引擎 `simulate()`，`forward_returns()` |
| `factors/grouped.py` | 分组计算核心：分位数、中位数、均值方差、去极值边界、分位分组、中性化回归 |
| `factors/panel.py` | 宽矩阵截面操作：`to_wide`、`outlier_mask_and_clip`、`zscore_rows`、`fill_missing`、`neutralize_rows`、`quantile_codes` |
| `factors/builders.py` | 长表预处理与 Fama-French 工具（`FactorBuildTools`） |
| `factors/quantile.py` / `sorted_factor.py` | 分位因子、排序因子构建器（长表 `build()`，宽矩阵 `build_wide()`） |
| `factors/backtest.py` | `FactorBacktester`：回测、IC、R²、汇总统计、绘图、因子回归 |
| `fama_french/` | FF3 复现与 16 个基础因子（CRSP / Compustat） |
| `embedded_leverage/` | Frazzini–Pedersen embedded leverage 复现（指数期权、杠杆 ETF） |

## 约定

- **时序**：调仓日 d 收盘决定权重，持有 (d, 下一个调仓日] 内的收益。
- **`drift`**：`False` 时持有期内维持目标权重；`True` 时买入持有，权重随价格漂移。
- **`weight_base`**：每期按给定矩阵（如滞后市值）重新加权分层组合，用于 Fama-French 口径。
- **交易成本**：成本 = Σ|Δw| × bps，扣在每个持有期第一天；换手报告为 0.5 × Σ|Δw|。
- **缺失收益**按 0 处理，退市收益需事先并入收益矩阵。
- 分位分组与 `pd.qcut(rank(method="first"))` 逐位一致；分位数、中位数与 pandas 逐位一致。

## 数据

仓库不包含任何数据、结果或论文。CRSP、Compustat、OptionMetrics 数据来自 WRDS，受许可协议限制，需自行获取并放到各项目的 `data/raw/` 下。
