import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    squared_error = (y_true - y_pred)**2
    return np.mean(squared_error)