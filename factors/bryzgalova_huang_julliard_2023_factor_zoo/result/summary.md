# Bayesian Solutions for the Factor Zoo — local application

- Candidate factors: 37; FF49 industry test portfolios: 49; months: 180.
- MCMC: 12000 draws, 3000 burn-in; prior model Sharpe ratio 0.10.
- Estimator: official continuous spike-and-slab B-SDF equations, OLS and common intercept.
- Scope: the authors' method is reproduced on local inputs; this does not claim to match their original 51-factor empirical tables.

## Highest posterior inclusion probabilities

| factor         |   inclusion_probability |   lambda_mean |     p05 |     p50 |    p95 |
|:---------------|------------------------:|--------------:|--------:|--------:|-------:|
| BM             |                  0.5871 |       -0.0061 | -0.0382 | -0.0005 | 0.0175 |
| HML            |                  0.5512 |       -0.0068 | -0.0416 | -0.0005 | 0.0170 |
| EARNINGS_YIELD |                  0.5436 |        0.0019 | -0.0169 |  0.0001 | 0.0236 |
| LTREV          |                  0.5407 |       -0.0030 | -0.0262 | -0.0002 | 0.0154 |
| BAB            |                  0.5380 |        0.0013 | -0.0183 |  0.0001 | 0.0232 |
| RMW            |                  0.5343 |        0.0005 | -0.0195 |  0.0000 | 0.0218 |
| LOW_NOA        |                  0.5333 |        0.0055 | -0.0156 |  0.0004 | 0.0367 |
| SIZE           |                  0.5296 |       -0.0024 | -0.0250 | -0.0002 | 0.0159 |
| MOM_12_2       |                  0.5290 |        0.0024 | -0.0124 |  0.0002 | 0.0219 |
| MKT_RF         |                  0.5272 |        0.0025 | -0.0204 |  0.0002 | 0.0295 |
| SMB            |                  0.5269 |       -0.0050 | -0.0387 | -0.0003 | 0.0187 |
| LOW_MAX        |                  0.5231 |        0.0005 | -0.0260 |  0.0000 | 0.0281 |
| ROA            |                  0.5227 |        0.0012 | -0.0173 |  0.0001 | 0.0221 |
| LOW_LEVERAGE   |                  0.5223 |        0.0034 | -0.0203 |  0.0002 | 0.0328 |
| REV_1M         |                  0.5209 |       -0.0007 | -0.0164 | -0.0000 | 0.0139 |

## Pricing diagnostics

|                                |   value |
|:-------------------------------|--------:|
| raw_ann_rmse_ex_intercept      |  0.0424 |
| bma_ann_rmse_ex_intercept      |  0.0403 |
| pricing_error_reduction        |  0.0497 |
| mean_model_size                | 19.0112 |
| bma_sdf_monthly_vol            |  0.0379 |
| max_split_inclusion_difference |  0.1111 |

## Posterior model size

|   model_size |   probability |
|-------------:|--------------:|
|            7 |        0.0001 |
|            8 |        0.0001 |
|            9 |        0.0007 |
|           10 |        0.0006 |
|           11 |        0.0032 |
|           12 |        0.0072 |
|           13 |        0.0168 |
|           14 |        0.0382 |
|           15 |        0.0527 |
|           16 |        0.0781 |
|           17 |        0.1099 |
|           18 |        0.1301 |
|           19 |        0.1281 |
|           20 |        0.1239 |
|           21 |        0.1057 |
|           22 |        0.0833 |
|           23 |        0.0563 |
|           24 |        0.0320 |
|           25 |        0.0208 |
|           26 |        0.0070 |
|           27 |        0.0034 |
|           28 |        0.0013 |
|           29 |        0.0003 |
|           31 |        0.0001 |
