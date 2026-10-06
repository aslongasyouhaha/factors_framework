# Low volatility (LOW_VOL_12M)

**Paper**: Ang, Hodrick, Xing & Zhang (2006), The cross-section of volatility and expected returns, Journal of Finance 61(1); Blitz & van Vliet (2007), The volatility effect, Journal of Portfolio Management

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Minus the standard deviation of the last 12 monthly returns (at least 8).

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing, signal at each month end (January 2008 - November 2025), held over the next month.
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- AHXZ use idiosyncratic volatility of daily FF3 residuals over the past month; this total-volatility version is closer to Blitz & van Vliet. TODO: idiosyncratic volatility from CRSP daily once the daily pipeline exists.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
