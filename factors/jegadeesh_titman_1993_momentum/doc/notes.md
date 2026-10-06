# Momentum (MOM_12_2)

**Paper**: Jegadeesh & Titman (1993), Returns to buying winners and selling losers, Journal of Finance 48(1); 12-2 convention from Carhart (1997)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Compounded return from month t-12 to t-2 (11 months, skipping the most recent month); at least 8 observed months and no missing month in the window.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing, signal at each month end (January 2008 - November 2025), held over the next month.
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- J&T study J-month / K-month overlapping portfolios (e.g. 6-6) with equal weights; here one 12-2 signal, value-weighted deciles, monthly rebalancing.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
