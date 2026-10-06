# Embedded leverage

**Paper**: Frazzini & Pedersen, Embedded Leverage, NBER Working Paper 18558 (2012) — `paper.pdf`.

Assets with embedded leverage (options, leveraged ETFs) should offer lower risk-adjusted returns
because leverage-constrained investors bid them up. The betting-against-leverage (BAB) factor is
long low-embedded-leverage and short high-embedded-leverage exposure, scaled to be market neutral.

## Scripts
- `build_index_options.py`: SPX / NDX option returns by maturity and delta bucket, daily
  delta-hedged, and the option BAB factor. Option and index-close data are read from the external
  store `D:/Projects/data` (OptionMetrics; not in this repository). Period returns are written to
  `data/factors/<project>/`, tables to `result/index_options/`.
- `build_levered_etfs.py`: downloads 1x / 2x index ETF pairs from Yahoo (cached in
  `data/source/yahoo/`) and builds the ETF BAB factor before and after fees; risk-free rate from
  `data/source/fred/`. Tables go to `result/yahoo_etfs/`.
- `report.ipynb`: tables and figures; run it from this folder.

## Status
Standalone scripts, not yet ported to `factorlab`.
