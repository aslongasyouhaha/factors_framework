# Good Volatility, Bad Volatility (RSJ)

This project implements the relative signed jump variation factor in Bollerslev,
Li, and Zhao (JFQA, 2020) using the shared `factorlab` framework.

## Definition

For stock `i` on day `t`, using intraday log returns `r`:

```text
RV+ = sum(r^2 where r > 0)
RV- = sum(r^2 where r < 0)
RSJ = (RV+ - RV-) / (RV+ + RV-)
```

The paper's weekly signal is the mean of the five daily RSJ observations. It is
formed at Tuesday's close and held until the next Tuesday close. The traded
factor is **low RSJ minus high RSJ**: buy stocks dominated by negative jumps and
short stocks dominated by positive jumps.

## Layout

```text
bollerslev_li_zhao_2020_good_bad_volatility/
  rsj.py          factor construction (realized measures, weekly signal)
  build.py        command-line replication: signals -> data/factors/<project>/, backtest -> result/
  tests/          unit tests
  doc/            this note (put the paper PDF here as paper.pdf)
  result/         generated tables and portfolios
```

Intraday input data goes in `data/source/` (not committed).

## Input data

The intraday input must be a CSV or Parquet long table with:

- `timestamp`: exchange-local timestamp;
- `asset_id`: permanent security identifier;
- `price`: positive price sampled on a regular intraday grid, normally 5-minute
  bars; alternatively supply `intraday_ret`, containing intraday log returns.

The constructor does **not** manufacture missing bars. Prepare a regular grid
upstream and carry the last valid transaction price forward within the trading
session if replicating the paper's TAQ procedure. By default a stock-day needs
at least 60 valid intraday returns.

For value weighting, provide a separate long table with `date`, `asset_id`, and
`market_equity`. Daily returns used for the backtest must contain `date`,
`asset_id`, and `ret` (decimal total return, including delisting returns where
available).

## Run

Signal and holdings only, equal weighted:

```powershell
python good_bad_volatility/scripts/build_rsj_factor.py `
  --bars data/intraday_5min.parquet `
  --output good_bad_volatility/results `
  --weighting equal
```

Full value-weighted backtest:

```powershell
python good_bad_volatility/scripts/build_rsj_factor.py `
  --bars data/intraday_5min.parquet `
  --market-equity data/market_equity.parquet `
  --returns data/daily_returns.parquet `
  --output good_bad_volatility/results `
  --weighting value --cost-bps 10
```

Outputs follow the repository convention: daily realized measures, weekly
signals, quintile holdings, low-minus-high holdings, return series, turnover,
IC, and summary statistics.

The two large root ZIP files currently contain daily OHLCV data. They cannot
replicate RSJ because the factor requires intraday returns; using daily returns
would create a different sign-of-return proxy rather than the paper's factor.

