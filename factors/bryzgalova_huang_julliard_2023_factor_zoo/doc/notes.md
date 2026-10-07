# Bayesian Solutions for the Factor Zoo

**Paper:** Bryzgalova, Huang & Julliard (2023), *Journal of Finance* 78(1), 487–557.

## Implementation

The project ports the paper's official continuous spike-and-slab B-SDF sampler to
Python and applies it to the local factor library. Locally reconstructed, lagged-value-
weighted FF49 industry portfolios provide independent test assets and avoid mechanically
including a candidate long-short factor in its own test-asset span. The prior is calibrated to a monthly model Sharpe
ratio of 0.10 using equation (27). Outputs include posterior factor inclusion
probabilities, risk-price credible intervals, posterior model size and the BMA-SDF.

The econometric method follows the paper, but the local input panel does not equal the
authors' original 51 factors. Official R source used for validation is kept under
`vendor/BayesianFactorZoo`.
