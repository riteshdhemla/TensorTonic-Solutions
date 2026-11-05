import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    shape_len = len(np.array(x).shape)
    if shape_len == 1:
        x = np.array(x).reshape(1, np.array(x).shape[0])
    max_x = np.max(x, axis = 1)
    x = np.array(x) - max_x[:, np.newaxis]
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x, axis = 1)
    res = exp_x / sum_exp[:, np.newaxis]
    return res if shape_len > 1 else res[0]