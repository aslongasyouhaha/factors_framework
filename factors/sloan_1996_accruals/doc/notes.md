# Accruals (LOW_ACCRUALS)

**Paper**: Sloan (1996), Do stock prices fully reflect information in accruals and cash flows about future earnings?, The Accounting Review 71(3)

The PDF is not committed; put it in this folder as `paper.pdf`.

## Our definition
Minus balance-sheet accruals (change over the same quarter a year earlier in non-cash current assets minus change in current liabilities excluding short-term debt, minus trailing-four-quarter depreciation) / average total assets. Latest fiscal quarter announced by the month end.

- Universe: CRSP US common stocks (NYSE / AMEX / NASDAQ), one share class per company with company-level market equity.
- Monthly rebalancing with point-in-time accounting data: at each month end, the latest fiscal quarter already public (announcement date rdq, or fiscal quarter end + 90 days when rdq is missing) whose fiscal period ended within the past year; flows are trailing four quarters, growth compares with the same quarter a year earlier. Universe: CRSP common stocks with such data, excluding financial firms (historical CRSP SIC 6000-6999) (January 2008 - November 2025).
- Portfolios: value-weighted deciles on the winsorised (1%, 99%) signal; factor = D10 - D01 (high exposure minus low).

## Differences from the paper
- Sloan also subtracts the change in income taxes payable (not in this extract). A blank debt in current liabilities is treated as zero.

## Status
Shared baseline implementation of the original 16-factor set, sample 2008-2025. Not yet a
faithful replication; see the differences above before drawing conclusions.
