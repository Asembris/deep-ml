import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float],
        W2: list[list[float]], b2: list[float],
        dropout_p: float = 0.1, seed: int = 42) -> list[float]:

    x = np.array(x)
    W1 = np.array(W1)
    b1 = np.array(b1)
    W2 = np.array(W2)
    b2 = np.array(b2)

    rng = np.random.RandomState(seed)

    z1 = W1 @ x + b1
    a1 = np.maximum(z1, 0.0)

    z2 = W2 @ a1 + b2

    mask = (rng.rand(*z2.shape) > dropout_p).astype(float)
    mask /= (1 - dropout_p)

    z2 *= mask

    res = z2 + x

    return np.round(res, 4).tolist()

