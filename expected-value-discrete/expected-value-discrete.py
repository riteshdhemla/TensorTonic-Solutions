import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype = np.float64)
    p = np.array(p, dtype = np.float64)
    if x.shape != p.shape:
        raise ValueError("shapes doesn't match")
    if not np.isclose(np.sum(p), 1.0, atol = 1e-6):
        raise ValueError("Probabilities don't sum up to 1")
    return np.sum(x * p)
