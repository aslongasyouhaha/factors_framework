# Short-term reversal (REV_1M)

**Paper**: Jegadeesh (1990), Evidence of predictable behavior of security returns, Journal of Finance 45(3); Lehmann (1990), Quarterly Journal of Economics

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Minus the stock's return in the formation month.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing, signal at each month end (January 2008 - November 2025), held over the next month.
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Jegadeesh uses predicted returns from a regression on lagged returns; here the plain one-month reversal.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
