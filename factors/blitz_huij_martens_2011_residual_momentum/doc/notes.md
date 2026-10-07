# Residual momentum

For every stock-month, excess returns are regressed on MKT-RF, SMB and HML over a rolling 36-month window (minimum 24 months). The score is the mean residual over months t-12 through t-2 divided by its residual standard deviation; at least eight residuals are required.

The implementation uses Kenneth French factor returns already stored in `data/base`. The sample is shorter near listings because rolling factor loadings require history.
