import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    d=x.shape[-1]//2
    x1=x[:,0:d]
    x2=x[:,d:2*d]

    scores= x1 * x2 /(1+np.exp(-x2))
    scores=np.round(scores,4)
    return scores