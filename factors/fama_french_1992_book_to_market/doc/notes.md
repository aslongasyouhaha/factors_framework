# Book-to-market (BM)

**Paper**: Fama & French (1992), The cross-section of expected stock returns, Journal of Finance 47(2); Rosenberg, Reid & Lanstein (1985)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Book equity of the fiscal year ending in calendar year t-1 (announced by June t) over December t-1 market equity, formed each June.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- June formation (2008 - 2025): accounting data of the fiscal year ending in the previous calendar year, used only if announced by June 30, held July to June; Fama-French universe (positive book equity, listed at least 180 days).
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Book equity from fiscal-Q4 quarterly Compustat items rather than the annual file.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
