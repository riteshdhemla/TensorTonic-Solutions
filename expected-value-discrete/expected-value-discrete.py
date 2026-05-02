import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    if not np.isclose(np.sum(p), 1.0, rtol=1e-6):
        raise ValueError("probabilities should sum to 1")
    exp_x = np.dot(x, p)
    return float(exp_x)
