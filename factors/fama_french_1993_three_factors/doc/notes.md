# Fama-French three factors (FF3)

**Paper**: Fama & French (1993), Common risk factors in the returns on stocks and bonds,
Journal of Financial Economics 33(1). The PDF is not committed; put it here as `paper.pdf`.

## Construction
- Universe: CRSP US common stocks on NYSE / AMEX / NASDAQ; market equity summed across a
  company's share classes and assigned to its largest one.
- Each June: size = June market equity; book-to-market = book equity of the fiscal year ending
  in the previous calendar year / December market equity. Stocks need positive book-to-market
  and at least 180 days of listing history.
- Breakpoints from NYSE stocks: median size; 30th / 70th percentile book-to-market.
- Six portfolios (SL, SM, SH, BL, BM, BH) held July to June, value-weighted each month by the
  previous month's market equity.
- SMB = mean(SL, SM, SH) - mean(BL, BM, BH); HML = mean(SH, BH) - mean(SL, BL);
  MKT_RF = value-weighted market return - RF.

## Differences from the paper / French's data library
- Book equity comes from fiscal-Q4 quarterly Compustat items, not the annual file.
- RF is French's one-month T-bill return (data/base/rf_monthly), so RF matches his data exactly.
- Point in time: a stock enters the June formation only if the fiscal year's results were
  announced (Compustat rdq, or fiscal year end + 90 days when unknown) by June 30.

## Replication quality
Sample July 2008 - December 2025; see `result/summary.md` for the comparison with French's factors.
