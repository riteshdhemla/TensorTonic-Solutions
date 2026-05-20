import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param, dtype = np.float64)
    grad = np.array(grad, dtype = np.float64)
    m = np.array(m, dtype = np.float64)
    v = np.array(v, dtype = np.float64)
    m_new = beta1*m + (1 - beta1)*grad
    v_new = beta2*v + (1 - beta2)*(np.multiply(grad, grad))
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    param_new = param - lr * (np.divide(m_hat, (np.sqrt(v_hat) + eps)))
    return (param_new, m_new, v_new)
