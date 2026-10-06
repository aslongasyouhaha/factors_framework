# factors_framework

美股横截面因子研究：每篇论文是一个复现项目，所有项目共用同一套数据和同一个框架 `factorlab`。

## 目录结构

```
framework/                 统一框架 factorlab（所有项目都调用它）
  factorlab/               回测引擎、截面处理、因子构建、评价、数据读写、标准报告
  tests/                   框架测试
pipeline/                  把 data/source 的原始文件处理成 data/base（各项目共用）
factors/                   复现项目，每篇论文一个文件夹
  <作者_年份_主题>/
    doc/                   论文 PDF（不进 git）和 notes.md（定义、与原文差异、待办）
    build.py               读 data/base → 构建因子暴露 → 写入 data/factors/<项目>/
    report.py              生成 result/：分层、多空、换手、IC、IC 衰减、分年度
    result/                结果（summary.md 进 git，其余为生成文件）
data/                      本地数据，不进 git
  source/                  下载的原始文件，原样保存（CRSP、Compustat、CCM、FRED、Yahoo……）
  base/                    清洗后的共用数据（crsp_monthly、compustat_quarterly、ccm_link、rf_monthly、french_factors_monthly）
  factors/                 各项目构建好的因子数据（exposure.parquet + meta.json）
```

## 工作流程

```bash
# 1. 原始数据 -> data/base（数据更新时才需要重跑）
python pipeline/build_crsp_monthly.py --start-year 2006 --end-year 2025   # 约 35 分钟
python pipeline/build_compustat.py --start-year 2004 --end-year 2026
python pipeline/build_rf.py                                              # French 的 RF 和因子

# 2. 构建一个因子 -> data/factors/<项目>/
python factors/jegadeesh_titman_1993_momentum/build.py

# 3. 生成结果 -> factors/<项目>/result/
python factors/jegadeesh_titman_1993_momentum/report.py
```

## 约定

- **因子暴露**：每个调仓日每只股票一个值，**值越大越看多**；存成长表 `(date, asset_id, exposure)`，`asset_id` 为 CRSP permno。
- **项目之间不互相 import**：只调用 `factorlab`、只读 `data/`。需要别的因子（如 FF3 做风险调整）就读 `data/factors/<那个项目>/`。
- **数据清洗只在 `pipeline/` 做一次**：`build.py` 不直接读 `data/source`。
- **结果统一由 `factorlab.report` 生成**，所有项目的指标口径一致。
- **存储用长表，计算用宽矩阵**：`data/` 里都是长表 parquet，框架读取时转成 date × asset 矩阵计算。
- 时序：调仓日 d 收盘定权重，持有 (d, 下一个调仓日]；IC 是信号对下一期收益。
- **样本期**统一在 `factorlab.panels`：月度调仓 2008-01 至 2025-11（CRSP 截至 2025-12），年度 6 月形成 2008–2025。
- **财务数据按公布日期使用（时点数据）**：每个季度有 `available_date` = 财报公布日 `rdq`；缺失时用季度末 + 90 天（CRSP 可链接的季度中约 10%）。月度财务因子在每个月末只用当时已公布的最新季度（超过一年的视为过期）；年度 6 月形成只用 6 月 30 日前已公布的财年。
- **财务因子剔除金融股**：按当月 CRSP 历史行业代码 SIC 6000–6999 剔除（银行、保险等资产负债表不可比）；价格类因子和 FF3 保留金融股。

## 项目

| 项目 | 因子 | 状态 |
|---|---|---|
| fama_french_1993_three_factors | MKT_RF、SMB、HML | 复现，与 French 数据对比见 result/summary.md |
| banz_1981_size | 规模 | 统一口径基准版 |
| fama_french_1992_book_to_market | 账面市值比 | 统一口径基准版 |
| jegadeesh_titman_1993_momentum | 动量 12-2 | 统一口径基准版 |
| jegadeesh_1990_short_term_reversal | 短期反转 | 统一口径基准版 |
| ang_hodrick_xing_zhang_2006_volatility | 低波动 | 基准版，定义与原文不同 |
| amihud_2002_illiquidity | 非流动性 | 基准版，定义与原文不同 |
| basu_1977_earnings_yield | 盈利收益率 | 月度时点版 |
| novy_marx_2013_gross_profitability | 毛利率 | 月度时点版 |
| ball_et_al_2015_operating_profitability | 营业利润率 | 月度时点版，分母与原文不同 |
| haugen_baker_1996_roa | ROA | 月度时点版 |
| cooper_gulen_schill_2008_asset_growth | 资产增长 | 月度时点版 |
| lakonishok_shleifer_vishny_1994_sales_growth | 销售增长 | 月度时点版，**方向与原文相反** |
| george_hwang_2010_leverage | 低杠杆 | 月度时点版 |
| palazzo_2012_cash_holdings | 现金持有 | 月度时点版 |
| sloan_1996_accruals | 应计 | 月度时点版 |
| titman_wei_xie_2004_capital_investment | 资本支出 | 月度时点版，定义与原文不同 |
| frazzini_pedersen_2012_embedded_leverage | 期权、杠杆 ETF 的 BAB | 独立脚本，尚未接入框架 |
| bollerslev_li_zhao_2020_good_bad_volatility | RSJ | 代码完成，缺日内数据 |
| benichou_et_al_2016_agnostic_risk_parity 等 5 个 | — | 只有论文 |

"统一口径基准版" / "月度时点版"：同样的样本期和市值加权十分位多空，便于横向比较；财务因子每月用已公布的最新季度数据。与原文的差异写在各项目的 `doc/notes.md`。

## 安装与测试

```bash
pip install -e "framework[plot,dev]"
pytest            # 框架、pipeline 和各项目的测试
```
