# Good Volatility, Bad Volatility (RSJ)

Paper: Bollerslev, Li, and Zhao (2020), *Good Volatility, Bad Volatility,
and the Cross Section of Stock Returns*, JFQA 55(3), 751–781.

## Definition and direction

For five-minute intraday log returns `r`:

```text
RV+ = sum(r^2 where r > 0)
RV- = sum(r^2 where r < 0)
RSJ = (RV+ - RV-) / (RV+ + RV-)
```

The weekly raw signal is the arithmetic mean of five consecutive daily RSJ
observations and is formed at Tuesday's close. The paper buys low RSJ and shorts
high RSJ. Because this repository requires "larger exposure means long", the
stored exposure is `-weekly_RSJ`.

## Local source and point-in-time universe

The one-minute source currently lives at `F:\temp` and is not copied into the
repository. It contains 72 quarterly Parquet files, 2008Q1–2025Q4, with about
2.37 billion rows (27.9 GB). Fields are UTC Unix timestamp, UTC date and minute,
symbol, and unadjusted one-minute OHLCV.

The vendor symbol is not a permanent identifier. `pipeline/build_crsp_daily.py`
therefore creates a year-partitioned point-in-time bridge containing historical
CRSP TradingSymbol, PERMNO, total return, price and market equity. It retains
active US common equities on NYSE, AMEX and NASDAQ. During the minute build the
universe is further restricted to CRSP prices from $5 through $1,000. Ambiguous
symbol-date mappings are dropped instead of guessed.

## Intraday treatment

`pipeline/build_rsj_daily.py`:

1. interprets `timestamp` in UTC and uses a DST-aware 09:30 New York session
   opening minute;
2. keeps session minutes 0–390 and drops pre/post-market observations;
3. aggregates minutes 0–389 into 78 five-minute intervals;
4. carries the last observed price across an empty interval;
5. uses the **open** of the 16:00 vendor bar as the final endpoint, capturing
   the closing auction without using its post-market close;
6. removes demonstrable vendor bad ticks: open and close must be within
   0.5x–2.0x the point-in-time CRSP daily close, and isolated observations
   more than 1.5x away from a centered five-observation median are dropped;
7. requires at least 60 valid five-minute returns and positive realized
   variance.

The first return of a day starts from the first observed intraday `open`, so no
overnight return or split gap enters RSJ.

## Build and report

```powershell
python pipeline/build_crsp_daily.py --start-year 2008 --end-year 2025
python pipeline/build_rsj_daily.py --source F:\temp --start-year 2008 --end-year 2025
python factors/bollerslev_li_zhao_2020_good_bad_volatility/build.py
python factors/bollerslev_li_zhao_2020_good_bad_volatility/report.py
```

Outputs:

- `data/base/crsp_daily/YYYY.parquet`: point-in-time daily CRSP data;
- `data/base/rsj_daily/YYYYQn.parquet`: daily realized measures;
- `data/base/rsj_daily/inventory.csv`: quarter-level row counts and validation
  diagnostics;
- `data/factors/bollerslev_li_zhao_2020_good_bad_volatility/`: weekly exposure
  and raw weekly realized measures;
- `factors/bollerslev_li_zhao_2020_good_bad_volatility/result/`: value-weighted
  quintiles, low-minus-high return, turnover, IC and yearly results.

The report compounds CRSP total returns over Tuesday-to-Tuesday holding periods
and uses formation-date CRSP market equity. Its main spread is D05 minus D01 in
the stored `-RSJ` exposure, which is economically low raw RSJ minus high raw
RSJ.

## Local replication result

The completed 2008–2025 build contains 7,714,327 stock-day realized-measure
rows and 1,509,393 weekly exposures for 3,923 PERMNOs. In the standard
value-weighted quintile report, low raw RSJ minus high raw RSJ earns -0.68% per
year (Newey-West t = -0.23; 931 weekly observations). Thus this local vendor
sample does **not** reproduce the paper's positive premium. This is a result,
not a direction flip: D05 in stored `-RSJ` is the low-raw-RSJ portfolio.

## Differences from the paper

- Available sample is 2008–2025 rather than 1993–2013.
- Source is vendor one-minute aggregate OHLCV rather than cleaned TAQ trades.
- A one-minute bar cannot isolate the exact transaction at each five-minute
  grid point; the construction above uses interval endpoints and the 16:00 bar
  open as a documented approximation.

