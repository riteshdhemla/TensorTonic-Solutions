import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    output = []
    (batch, seq_len, input_dim) = X.shape
    h_prev = h_0
    for i in range(seq_len):
        h = np.tanh(X[:,i,:]@W_xh.T + h_prev@W_hh.T + b_h)
        output.append(h)
        h_prev = h
    h_final = h_prev
    output = np.stack(np.array(output), axis = 1)
    return output, h_final
        