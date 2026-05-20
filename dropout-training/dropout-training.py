import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x, dtype = np.float64)
    dropout_pattern = rng.random(size = x.shape) if rng else np.random.random(size = x.shape)
    dropout_pattern[dropout_pattern < p] = 0
    dropout_pattern[dropout_pattern >= p] = 1.0 / (1 - p)
    output = np.multiply(x, dropout_pattern)
    return output, dropout_pattern
