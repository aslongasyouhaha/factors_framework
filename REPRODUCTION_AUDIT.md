# Reproduction audit (2026-10-07)

This audit applies a strict rule: a project may carry a paper's name only when the available data support the paper's signal timing and portfolio construction. Weak realized returns are not a reason for deletion; a materially different definition is.

## Rebuilt as paper portfolios

| Project | Correction |
|---|---|
| `daniel_hirshleifer_sun_2020_pead` | Four-day `[-2,+1]` CAR, two-trading-day month-end eligibility, six-month freshness, monthly NYSE size and 20/80 CAR 2x3 portfolios. |
| `fama_french_2015_rmw` | Annual operating profit uses revenue minus COGS, SG&A and interest; June NYSE median and 30/70 2x3 portfolios. |
| `fama_french_2015_cma` | Annual asset growth, June NYSE median and 30/70 2x3 portfolios. |
| `hirshleifer_et_al_2004_net_operating_assets` | Original NOA balance-sheet reconstruction including minority interest, scaled by lagged assets and formed annually in June. |
| `frazzini_pedersen_2014_bab` | One-year volatility/five-year three-day-correlation beta, shrinkage toward one, rank weights and separate unit-beta scaling of both legs. |

`bali_cakici_whitelaw_2011_max`, `de_bondt_thaler_1985_long_term_reversal`, `novy_marx_2012_intermediate_momentum`, and `blitz_huij_martens_2011_residual_momentum` retain definitions supported by the local data.

## Added from previously unextracted raw fields

| Project | Construction |
|---|---|
| `hou_xue_zhang_2015_q_factor` | Monthly 2x3x3 size, investment-to-assets and point-in-time quarterly ROE portfolios. |
| `piotroski_2000_fscore` | Nine annual financial-strength signals within the highest book-to-market quintile. |
| `jegadeesh_livnat_2006_sue` | Seasonal change in split-adjusted EPS using observed report dates. |
| `pontiff_woodgate_2008_share_issuance` | Real share growth from month t-18 to t-6 using ex-dividend CRSP returns. |
| `amaya_et_al_2015_realized_skewness` | Weekly realized skewness from cleaned five-minute intraday returns. |
| `amaya_et_al_2015_realized_kurtosis` | Weekly realized kurtosis from cleaned five-minute intraday returns. |

## Factor selection and portfolio allocation

| Project | Scope |
|---|---|
| `bryzgalova_huang_julliard_2023_factor_zoo` | Official continuous spike-and-slab B-SDF equations ported from the authors' package; applied to 37 local factors and locally reconstructed FF49 industry test portfolios. This reproduces the method, not the authors' original 51-factor input panel. |
| `benichou_et_al_2016_agnostic_risk_parity` | ARP `C^(-1/2)p` applied to the 37-factor return panel and compared with equal-risk-signal and Markowitz rules. Ledoit-Wolf cleaning replaces the paper's RIE; this is an algorithm experiment rather than a 110-futures data replication. |

## Removed

| Project | Reason |
|---|---|
| `pastor_stambaugh_2003_liquidity` | The earlier implementation was not the paper's aggregate innovation and loading procedure. |
| `daniel_titman_2006_net_stock_issues` | Total-return market-cap-growth proxy did not identify the requested one-year net issuance measure. |
| `boyer_mitton_vorkink_2010_idiosyncratic_skewness` | Realized one-month residual skewness was substituted for expected idiosyncratic skewness. |
| `harvey_siddique_2000_coskewness` | One-month realized daily coskewness was substituted for the conditional three-moment model. |
| `asness_frazzini_pedersen_2019_qmj` | Missing components and size-neutral multistage construction made the seven-component proxy materially different. |
| `ohlson_1980_distress` | Missing price-level adjustment and original funds-from-operations input made the score non-comparable. |
| `baltussen_et_al_2025_risk_managed_momentum` | Stock momentum divided by volatility was not the paper's portfolio-level volatility overlay. |
| `george_hwang_2004_52_week_high` | Monthly total-return wealth was not the split-adjusted daily price-to-high signal. |
| `moskowitz_grinblatt_1999_industry_momentum` | The local 12-month peer-return construction and changing monthly SIC assignment did not match the paper. |
| `baltussen_et_al_2025_multidimensional_momentum` | The five-signal local composite omitted material signal families from the paper. |
