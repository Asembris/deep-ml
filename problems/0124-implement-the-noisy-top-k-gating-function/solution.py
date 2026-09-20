import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    def softplus(x):
        return np.log(1+np.exp(x))

    H_base=X @ W_g
    H_noise= X @ W_noise
    H=H_base + N * softplus(H_noise)

    masked_H=np.full_like(H, -np.inf, dtype=float)
    top_k_indices=np.argsort(H,axis=-1)[:,-k:]
    rows=np.arange(H.shape[0])[:, None]
    
    masked_H[rows,top_k_indices]=H[rows,top_k_indices]

    G=np.exp(masked_H)/np.sum(np.exp(masked_H),axis=-1,keepdims=True)

    return G






