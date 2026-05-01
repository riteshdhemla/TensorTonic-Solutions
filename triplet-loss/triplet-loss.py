import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.array(anchor, dtype=np.float32)
    positive = np.array(positive, dtype=np.float32)
    negative = np.array(negative, dtype=np.float32)
    axis = 1 if len(anchor.shape) > 1 else 0
    dist_p = np.linalg.norm(anchor - positive, axis = axis)
    dist_n = np.linalg.norm(anchor - negative, axis = axis)
    return float(np.mean(np.maximum(0.0, dist_p**2 - dist_n**2 + margin)))