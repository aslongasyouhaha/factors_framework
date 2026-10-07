# PEAD

The event signal is the market-adjusted CRSP return over trading days [-2,+1] around an observed Compustat `rdq`, requiring at least two valid daily returns. An announcement enters a month-end sort only after two trading days and remains eligible for six months. Imputed announcement dates and implausible reporting lags are excluded.

Each month, common stocks excluding financials are independently assigned using the NYSE median size breakpoint and NYSE 20/80 CAR breakpoints. The six next-month portfolios are value weighted. PEAD is `(SH+BH)/2 - (SL+BL)/2`, matching Daniel, Hirshleifer and Sun's stated construction.
