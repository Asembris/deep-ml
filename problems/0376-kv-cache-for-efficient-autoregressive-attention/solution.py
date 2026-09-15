import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_k = W_K.shape[-1]

    Q = x_new @ W_Q
    K = x_new @ W_K
    V = x_new @ W_V

    if cache is None:
        k_cache, v_cache = None, None
    else:
        k_cache, v_cache = cache

    k_cache = (
        K[None, :]
        if k_cache is None
        else np.concatenate((k_cache, K[None, :]), axis=0)
    )

    v_cache = (
        V[None, :]
        if v_cache is None
        else np.concatenate((v_cache, V[None, :]), axis=0)
    )

    scores = (Q @ k_cache.T) / np.sqrt(d_k)

    mx = np.max(scores)
    exp_scores = np.exp(scores - mx)
    attn_weights = exp_scores / np.sum(exp_scores)

    output = attn_weights @ v_cache

    return output, (k_cache, v_cache)








