import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    s_tm1 = np.array(s)
    g_t = np.array(g)
    w_tm1 = np.array(w)
    s_t = beta * s_tm1 + (1 - beta) * g_t * g_t
    w_t = w_tm1 - (lr / (np.sqrt(s_t) + eps)) * g_t
    return (w_t, s_t)
    