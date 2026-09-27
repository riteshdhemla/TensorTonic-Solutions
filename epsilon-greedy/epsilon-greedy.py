import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    gen = np.random.default_rng(seed=seed)
    idx = None
    rn = gen.random()
    if rn < epsilon:
        idx = gen.integers(0, high = len(q_values))
    else:
        idx = np.argmax(q_values)
    return int(idx)