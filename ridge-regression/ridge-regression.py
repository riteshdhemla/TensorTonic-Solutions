def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    _, d = X.shape
    w = np.linalg.inv(X.T@X + lam * np.eye(d, d))@X.T@y
    return w
