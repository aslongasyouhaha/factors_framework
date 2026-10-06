# Size effect (SIZE)

**Paper**: Banz (1981), The relationship between return and market value of common stocks, Journal of Financial Economics 9(1)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
-log(company market equity) at each month end; value-weighted decile spread rebalanced monthly.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing, signal at each month end (January 2008 - November 2025), held over the next month.
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Banz sorts on market value within NYSE stocks over 1926-1975 with beta-adjusted returns; here all NYSE/AMEX/NASDAQ common stocks, raw returns.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
