import numpy as np

def additive_attention(query, keys, W_q, W_k, v):
    """
    query: (d_q,), keys: (n, d_k), W_q: (d_a, d_q), W_k: (d_a, d_k), v: (d_a,)
    Returns (weights (n,), context (d_k,)).
    """
    # Your code here
    e = v @ np.tanh((W_q @ query)[:, None] + W_k @ keys.T)

    mx = np.max(e, keepdims=True)

    a = np.exp(e - mx) / np.sum(np.exp(e - mx))

    context = a @ keys

    return a, context
