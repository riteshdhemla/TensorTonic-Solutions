def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    res = []
    k = len(weights)
    for i in range(len(values) - k + 1):
        num = 0
        deno = 0
        for j in range(k):
            num += weights[j] * values[i + j]
            deno += weights[j]
        res.append(num / deno)
    return res