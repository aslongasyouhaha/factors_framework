# Operating profitability (OP_PROFIT)

**Paper**: Ball, Gerakos, Linnainmaa & Nikolaev (2015), Deflating profitability, Journal of Financial Economics 117(2); Fama & French (2015), A five-factor asset pricing model, Journal of Financial Economics 116(1)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Operating income after depreciation over the last four quarters / total assets. Latest fiscal quarter announced by the month end.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing with point-in-time accounting data: at each month end, the latest fiscal quarter already public (announcement date rdq, or fiscal quarter end + 90 days when rdq is missing) whose fiscal period ended within the past year; flows are trailing four quarters, growth compares with the same quarter a year earlier. Universe: CRSP common stocks with such data, excluding financial firms (historical CRSP SIC 6000-6999) (January 2008 - November 2025).
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Fama & French (2015) divide (revenue - COGS - SG&A - interest) by book equity; Ball et al. use operating profit before R&D over assets. This is operating income after depreciation over assets.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
