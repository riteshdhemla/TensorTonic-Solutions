import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    x = np.array(X)
    Y = np.array(y)
    weight = np.linalg.inv(x.T@x)@x.T@Y
    return weight.tolist()