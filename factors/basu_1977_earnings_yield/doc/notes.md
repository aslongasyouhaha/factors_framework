# Earnings yield (EARNINGS_YIELD)

**Paper**: Basu (1977), Investment performance of common stocks in relation to their price-earnings ratios, Journal of Finance 32(3)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Trailing-four-quarter net income / market equity at the month end. Latest fiscal quarter announced by the month end.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing with point-in-time accounting data: at each month end, the latest fiscal quarter already public (announcement date rdq, or fiscal quarter end + 90 days when rdq is missing) whose fiscal period ended within the past year; flows are trailing four quarters, growth compares with the same quarter a year earlier. Universe: CRSP common stocks with such data, excluding financial firms (historical CRSP SIC 6000-6999) (January 2008 - November 2025).
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Basu forms P/E quintiles on NYSE stocks; negative earnings are kept here (they rank lowest).

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
