# Agnostic Risk Parity on the local factor zoo

- Universe: 37 locally constructed factor return series.
- Signal: trailing 12-month mean of volatility-normalized factor returns.
- Covariance: trailing 60 months (minimum 36); Ledoit-Wolf cleaning.
- All strategies use a 10% ex-ante annual volatility target and 10 bp per unit turnover.
- ARP uses cleaned C^(-1/2)p; Markowitz uses C^(-1)p.
- This is an algorithm application, not a reproduction of the paper's proprietary 110-futures dataset.

## Net performance

|                   |        n |   ann_return |   ann_vol |   sharpe |   max_drawdown |   gross_ann_return |   avg_monthly_turnover |   ann_cost |
|:------------------|---------:|-------------:|----------:|---------:|---------------:|-------------------:|-----------------------:|-----------:|
| ARP               | 144.0000 |       0.0815 |    0.1617 |   0.5042 |        -0.2887 |             0.0977 |                 1.3483 |     0.0162 |
| equal_risk_signal | 144.0000 |       0.0682 |    0.1580 |   0.4319 |        -0.2149 |             0.0799 |                 0.9740 |     0.0117 |
| markowitz_clean   | 144.0000 |       0.0955 |    0.1532 |   0.6237 |        -0.3508 |             0.1171 |                 1.8009 |     0.0216 |
| markowitz_raw     | 144.0000 |       0.0591 |    0.1219 |   0.4848 |        -0.2618 |             0.0899 |                 2.5671 |     0.0308 |

## Eigenrisk diagnostic

| strategy          |   mean |   median |    max |
|:------------------|-------:|---------:|-------:|
| ARP               | 0.2091 |   0.1873 | 0.5565 |
| equal_risk_signal | 0.4465 |   0.4379 | 0.8472 |
| markowitz_clean   | 0.0902 |   0.0834 | 0.2390 |
| markowitz_raw     | 0.3229 |   0.2404 | 0.9906 |

Lower HHI means realized risk is spread across more covariance eigenmodes.
