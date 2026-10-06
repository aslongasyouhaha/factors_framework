# Illiquidity (ILLIQ)

**Paper**: Amihud (2002), Illiquidity and stock returns: cross-section and time-series effects, Journal of Financial Markets 5(1)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
|monthly return| / average daily dollar volume in the month.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing, signal at each month end (January 2008 - November 2025), held over the next month.
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Amihud averages daily |r| / dollar volume over the previous year; here a single month of monthly data. TODO: daily version once the daily pipeline exists.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
