import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""

    T = attn_weights.shape[0]

    if method == 'tril':
        mask = np.tril(np.ones((T, T)))
        weights = attn_weights * mask
        result = weights / np.sum(weights, axis=-1, keepdims=True)

    else:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        scores = np.log(attn_weights)
        scores = np.where(mask, -np.inf, scores)

        mx = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - mx)
        result = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


    return result.tolist()