import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    e = np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))
    l = np.zeros_like(e)
    less_cond = (e <= delta)
    great_cond = (e > delta)
    l[less_cond] = 0.5*(e[e <= delta]**2)
    l[great_cond]  = delta*(e[e > delta] - delta/2)
    return np.mean(l)