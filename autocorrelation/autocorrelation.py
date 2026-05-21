def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    sum = 0
    for e in series:
        sum += e
    mean = sum / len(series)

    var = 0
    for e in series:
        var += (e - mean)**2

    autocorr = []
    for lag in range(max_lag + 1):
        autocovariance_at_lag = 0.0
        for t in range(len(series) - lag):
            autocovariance_at_lag += (series[t] - mean)*(series[t + lag] - mean)
        if var == 0:
            if lag == 0:
                autocorr.append(1)
            else:
                autocorr.append(0)
        else:
            autocorr.append(autocovariance_at_lag / var)   
    assert autocorr[0] == 1
    return autocorr
