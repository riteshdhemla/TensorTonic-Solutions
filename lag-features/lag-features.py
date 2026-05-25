def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    lag_features = []
    # Write code here
    for i in range(len(series)):
        lag_features.append([series[i - lag] if (i - lag) >= 0 else None for lag in lags])
    lag_features = list(filter(lambda x: None not in x, lag_features))
    return lag_features
        