def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here, indicates long term trends
    assert len(values) >= window_size
    sma : list[float] = []
    leading_prefix_sum, lagging_prefix_sum = 0.0, 0.0
    for i in range(window_size):
        leading_prefix_sum += values[i]

    for i in range(window_size, len(values)):
        sma.append((leading_prefix_sum - lagging_prefix_sum) / window_size)
        leading_prefix_sum += values[i]
        lagging_prefix_sum += values[i - window_size]
    sma.append((leading_prefix_sum - lagging_prefix_sum) / window_size)
    return sma
        