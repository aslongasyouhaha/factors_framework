# Betting against beta

The ex-ante beta estimator combines one-year daily volatility (minimum 120 observations) with five-year correlation of overlapping three-day log excess returns (minimum 750 observations), then shrinks the estimate 60% toward its time-series value and 40% toward one.

Each month, stocks are rank weighted into below- and above-median beta legs. Each leg is separately scaled to unit formation beta and BAB is the leveraged low-beta excess return minus the de-leveraged high-beta excess return. The short local history means the valid sample starts only after the correlation estimator has enough observations.
