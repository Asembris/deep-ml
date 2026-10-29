import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.

    Args:
        X: input array of shape (B, S, d_in)
        W_q, W_k, W_v: projection matrices of shape (d_in, d_out)
        W_o: output projection matrix of shape (d_out, d_out)
        num_heads: number of attention heads (must divide d_out)
        causal: if True, apply a causal mask so position t only attends to <= t

    Returns:
        Nested list of shape (B, S, d_out) representing the attention output.
    """
    X = np.asarray(X, dtype=float)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)
    W_o = np.asarray(W_o, dtype=float)
    B, S, _ = X.shape
    d_out = W_q.shape[1]
    d_head = d_out // num_heads

    Q = np.einsum('bsd,df->bsf', X, W_q)
    K = np.einsum('bsd,df->bsf', X, W_k)
    V = np.einsum('bsd,df->bsf', X, W_v)

    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = np.einsum('bhid,bhjd->bhij', Q, K) / np.sqrt(d_head)

    if causal:
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    out = np.einsum('bhij,bhjd->bhid', weights, V)

    out = out.transpose(0, 2, 1, 3).reshape(B, S, d_out)
    out = np.einsum('bsd,df->bsf', out, W_o)

    return out.tolist()
