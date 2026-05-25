def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    new_series = []
    n = len(series)
    series_ip = series
    for d in range(order):
        new_series = [0 for _ in range(n - d - 1)]
        for i in range(n - d - 1):
            new_series[i] = (series_ip[i + 1] - series_ip[i])
        series_ip = new_series
    return series_ip
        