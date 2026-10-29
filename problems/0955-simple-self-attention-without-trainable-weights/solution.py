import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    
    X=np.array(X)
    attn_scores= X @ X.T
    mx=np.max(attn_scores,axis=-1,keepdims=True)
    att_weights=np.exp(attn_scores-mx)/np.sum(np.exp(attn_scores-mx),axis=-1,keepdims=True)

    context= att_weights @ X
    return context

