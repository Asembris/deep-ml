import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    d_k=K.shape[-1]
    attn_scores= (Q @ K.T)/np.sqrt(d_k)
    mx=np.max(attn_scores,axis=-1,keepdims=True)
    att_weights=np.exp(attn_scores-mx)/np.sum(np.exp(attn_scores-mx),axis=-1)
    return np.round(att_weights,4).tolist()
   

